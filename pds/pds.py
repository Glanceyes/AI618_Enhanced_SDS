from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from jaxtyping import Float
from PIL import Image
from typing import List, Dict
from pds.pds_unet import CustomUNet2DConditionModel
from pds.pds_attention import AttentionStore, register_attention_control, aggregate_attention
from pds.utils.free_lunch_utils import register_free_upblock2d_in, register_free_crossattn_upblock2d_in
from pds.utils.gaussian_smoothing import GaussianSmoothing

@dataclass
class PDSConfig:
    sd_pretrained_model_or_path: str = "runwayml/stable-diffusion-v1-5"

    num_inference_steps: int = 500
    min_step_ratio: float = 0.2
    max_step_ratio: float = 0.9

    src_prompt: str = "a photo of a sks man"
    tgt_prompt: str = "a photo of a sks Spider Man"
    
    src_token_indices_list: List[int] = field(default_factory=lambda: [0])
    tgt_token_indices_list: List[int] = field(default_factory=lambda: [0])
    
    cross_attn_res: int = 16
    cross_attn_layer_list: list = field(default_factory=lambda: ["up"])

    use_freeu: bool = False
    use_pds: bool = False
    timestep_annealing: bool = False

    log_step: int = 10
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    lambda_identity: float = 0.1
    device: torch.device = torch.device("cuda")


class PDS(object):
    def __init__(self, config: PDSConfig, use_wandb=True):
        self.config = config
        self.device = torch.device(config.device)

        self.pipe = DiffusionPipeline.from_pretrained(config.sd_pretrained_model_or_path).to(self.device)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.set_timesteps(config.num_inference_steps)
        self.pipe.scheduler = self.scheduler

        self.unet = CustomUNet2DConditionModel.from_pretrained(
            config.sd_pretrained_model_or_path,
            subfolder="unet"
        ).to(self.device)
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae

        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        ## construct text features beforehand.
        self.src_prompt = self.config.src_prompt
        self.tgt_prompt = self.config.tgt_prompt

        self.update_text_features(src_prompt=self.src_prompt, tgt_prompt=self.tgt_prompt)
        self.null_text_feature = self.encode_text("")
    
        # self.get_attn = self.config.tgt_token_indices_list and self.config.src_token_indices_list
        self.get_attn = False
        self.use_wandb = use_wandb
        self.use_freeu = self.config.use_freeu
        self.use_pds = self.config.use_pds
        self.timestep_annealing = self.config.timestep_annealing
        self.guidance_scale = self.config.guidance_scale
        
        self.check = 0
        
        self.iteration = 0
        self.max_iteration = 4000
        
    def compute_posterior_mean(self, xt, noise_pred, t, t_prev):
        """
        Computes an estimated posterior mean \mu_\phi(x_t, y; \epsilon_\phi).
        """
        device = self.device
        beta_t = self.scheduler.betas[t].to(device)
        alpha_t = self.scheduler.alphas[t].to(device)
        alpha_bar_t = self.scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = self.scheduler.alphas_cumprod[t_prev].to(device)

        pred_x0 = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        c0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
        c1 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        mean_func = c0 * pred_x0 + c1 * xt
        
        return mean_func, pred_x0

    def encode_image(self, img_tensor: Float[torch.Tensor, "B C H W"]):
        x = img_tensor
        x = 2 * x - 1
        x = x.float()
        return self.vae.encode(x).latent_dist.sample() * 0.18215

    def encode_text(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

    def decode_latent(self, latent):
        x = self.vae.decode(latent / 0.18215).sample
        x = (x / 2 + 0.5).clamp(0, 1)
        return x

    def update_text_features(self, src_prompt=None, tgt_prompt=None):
        if getattr(self, "src_text_feature", None) is None:
            assert src_prompt is not None
            self.src_prompt = src_prompt
            self.src_text_feature = self.encode_text(src_prompt)
        else:
            if src_prompt is not None and src_prompt != self.src_prompt:
                self.src_prompt = src_prompt
                self.src_text_feature = self.encode_text(src_prompt)

        if getattr(self, "tgt_text_feature", None) is None:
            assert tgt_prompt is not None
            self.tgt_prompt = tgt_prompt
            self.tgt_text_feature = self.encode_text(tgt_prompt)
        else:
            if tgt_prompt is not None and tgt_prompt != self.tgt_prompt:
                self.tgt_prompt = tgt_prompt
                self.tgt_text_feature = self.encode_text(tgt_prompt)

    def pds_timestep_sampling(self, batch_size):
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)

        min_step = 1 if self.config.min_step_ratio <= 0 else int(len(timesteps) * self.config.min_step_ratio)
        max_step = (
            len(timesteps) if self.config.max_step_ratio >= 1 else int(len(timesteps) * self.config.max_step_ratio)
        )
        max_step = max(max_step, min_step + 1)
        idx = torch.randint(
            min_step,
            max_step,
            [batch_size],
            dtype=torch.long,
            device="cpu",
        )
        t = timesteps[idx].cpu()
        t_prev = timesteps[idx - 1].cpu()

        return t, t_prev
    
    def pds_timestep_anneal_sampling(self, batch_size):
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)

        min_step = 1 if self.config.min_step_ratio <= 0 else int(len(timesteps) * self.config.min_step_ratio)
        max_step = (
            len(timesteps) if self.config.max_step_ratio >= 1 else int(len(timesteps) * self.config.max_step_ratio)
        )
        max_step = max(max_step, min_step + 1)

        idx = torch.full((batch_size,), (max_step-min_step)*((self.max_iteration-self.iteration)/self.max_iteration) + min_step, dtype=torch.long, device="cpu")

        t = timesteps[idx].cpu()
        t_prev = timesteps[idx - 1].cpu()
        #t_tau = timesteps[idx + tau].cpu()
        return t, t_prev
    
    def select_attention(
            self,
            attention_maps: torch.Tensor,
            prompt: str,
            indices_list: List[int] = None,
            normalize_eot_token: bool = False
        ) -> Dict[int, torch.Tensor]:
        last_token_index = -1
        
        if normalize_eot_token:
            if isinstance(prompt, list):
                prompt = prompt[0]
            last_token_index = len(self.tokenizer(prompt)["input_ids"]) - 1
            
        attention_maps = attention_maps[:, :, 1:last_token_index]
        attention_maps *= 100
        attention_maps = torch.nn.functional.softmax(attention_maps, dim=-1)
        
        attention_for_indices = {}
        
        if indices_list is not None:
            indices_list = [index - 1 for index in indices_list]
            for index in indices_list:
                attention_for_indices[index + 1] = attention_maps[:, :, index]
        else:
            for index in range(attention_maps.shape[-1]):
                attention_for_indices[index + 1] = attention_maps[:, :, index]            
    
        return attention_for_indices
    
    def get_cross_attn(self, h, w, attn_store, prompt, token_indices_list):
        l = min(h, w)
        
        token_attn_maps = {}
        
        agg_attns = aggregate_attention(
            attention_store=attn_store,
            h = round(self.config.cross_attn_res * h / l),
            w = round(self.config.cross_attn_res * w / l),
            from_where=self.config.cross_attn_layer_list,
            is_cross=True,
            batch_select=0
        )

        selected_token_attn_maps = self.select_attention(agg_attns, prompt, token_indices_list)
        
        for token_index in selected_token_attn_maps.keys():
            token_attn_map = F.interpolate(selected_token_attn_maps[token_index].unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0).squeeze(0)
            
            token_attn_map = (token_attn_map - token_attn_map.min()) / (token_attn_map.max() - token_attn_map.min() + 1e-8)
            token_attn_maps[token_index] = token_attn_map

        return token_attn_maps

    def __call__(
        self,
        tgt_x0,
        src_x0,
        tgt_prompt=None,
        src_prompt=None,
        reduction="mean",
        return_dict=False,
        step=0,
    ):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(src_prompt=src_prompt, tgt_prompt=tgt_prompt)
        tgt_text_embedding, src_text_embedding = (
            self.tgt_text_feature,
            self.src_text_feature,
        )
        uncond_embedding = self.null_text_feature

        batch_size = tgt_x0.shape[0]
        
        if self.timestep_annealing:
            t, t_prev = self.pds_timestep_anneal_sampling(batch_size)
        else:
            t, t_prev = self.pds_timestep_sampling(batch_size)
        
        # t, t_prev = self.pds_timestep_sampling(batch_size)
        
        beta_t = scheduler.betas[t].to(device)
        alpha_t = scheduler.alphas[t].to(device)
        alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = scheduler.alphas_cumprod[t_prev].to(device)
        
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)
        
        gamma_t = (alpha_bar_t_prev ** 0.5) * (1 - alpha_t) / (1 - alpha_bar_t)
        delta_t = (alpha_t ** 0.5) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)

        psi_t = 2 * ((alpha_bar_t_prev ** 0.5) - gamma_t - delta_t * (alpha_bar_t ** 0.5)) ** 2 / (sigma_t ** 2)
        chi_t = 2 * ((alpha_bar_t_prev ** 0.5) - gamma_t - delta_t * (alpha_bar_t ** 0.5)) / (sigma_t ** 2) * gamma_t * (1 / alpha_bar_t - 1).sqrt()

        xi_t = 2 * ((alpha_bar_t_prev ** 0.5) - delta_t * (alpha_bar_t ** 0.5)) ** 2 / (sigma_t ** 2)
        pi_t = 2 * ((alpha_bar_t_prev ** 0.5) - delta_t * (alpha_bar_t ** 0.5)) * gamma_t / (sigma_t ** 2)   

        noise = torch.randn_like(tgt_x0)
        noise_t_prev = torch.randn_like(tgt_x0)

        zts = dict()
        pred_x0s = dict()
        noise_preds = dict()
        # feats = dict()
        attn_maps = dict()
        
        for latent, cond_text_embedding, name in zip(
            [tgt_x0, src_x0], [tgt_text_embedding, src_text_embedding], ["tgt", "src"]
        ):
            # attn_store = AttentionStore()
            latents_noisy = scheduler.add_noise(latent, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            text_embeddings = torch.cat([cond_text_embedding, uncond_embedding], dim=0)
            
            # register_attention_control(self, attn_store)
            
            b1 = 1.1
            b2 = 1.1
            s1 = 0.9 # 0.8
            s2 = 0.2 # 0.1

            if self.check == 0:
                register_free_upblock2d_in(
                    self.unet,
                    b1=b1, 
                    b2=b2, 
                    s1=s1, 
                    s2=s2,
                    check = self.check
                )
                register_free_crossattn_upblock2d_in(
                    self.unet, 
                    b1=b1,
                    b2=b2,
                    s1=s1,
                    s2=s2,
                    check = self.check
                )
                self.check = 1
            else:
                register_free_upblock2d_in(
                    self.unet,
                    b1=b1, 
                    b2=b2, 
                    s1=s1, 
                    s2=s2,
                )
                register_free_crossattn_upblock2d_in(
                    self.unet, 
                    b1=b1,
                    b2=b2,
                    s1=s1,
                    s2=s2,
                )
            
            
            unet_outputs = self.unet.forward(
                latent_model_input,
                torch.cat([t] * 2).to(device),
                encoder_hidden_states=text_embeddings,
            )
            
            noise_pred = unet_outputs.sample
            unet_feats = unet_outputs.features
            
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            x_t_prev = scheduler.add_noise(latent, noise_t_prev, t_prev)
            mu, pred_x0 = self.compute_posterior_mean(latents_noisy, noise_pred, t, t_prev)
            zt = (x_t_prev - mu) / sigma_t
            zts[name] = zt
            pred_x0s[name] = pred_x0
            
            if not self.use_pds:
                if name == "tgt":
                    noise_preds[name] = noise_pred
                else:
                    noise_preds[name] = noise_pred_uncond
            else:
                noise_preds[name] = noise_pred
        
            _, _, h, w= pred_x0.shape
            
            # feat = None
            # for level in unet_feats.keys():
            #     if feat is None:
            #         feat = nn.Upsample(size=(h, w), mode="bilinear")(unet_feats[level].chunk(2)[0])
            #     else:
            #         feat = torch.cat([feat, nn.Upsample(size=(h, w), mode="bilinear")(unet_feats[level][0].unsqueeze(0))], dim=1)
        
            # feats[name] = feat
            
            # if self.get_attn:
            #     prompt = tgt_prompt if name == "tgt" else src_prompt
            #     token_indices_list = self.config.tgt_token_indices_list if name == "tgt" else self.config.src_token_indices_list
            #     attn_maps[name] = self.get_cross_attn(h, w, attn_store, prompt, token_indices_list)
        
        if self.use_pds:
            grad = zts["tgt"] - zts["src"]
        else:
            grad = (noise_preds["tgt"] - noise_preds["src"]) + (0.2 * (((self.max_iteration - self.iteration) / self.max_iteration) ** 2) + 0.02) * (tgt_x0 - src_x0)
        grad = torch.nan_to_num(grad)
        target = (tgt_x0 - grad).detach()
        loss = 0.5 * F.mse_loss(tgt_x0, target, reduction=reduction) / batch_size
    
        self.iteration += 1
    
        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t}
            return dic
        else:
            return loss

    def run_sdedit(self, x0, tgt_prompt=None, num_inference_steps=20, skip=7, eta=0):
        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        reversed_timesteps = reversed(scheduler.timesteps)

        S = num_inference_steps - skip
        t = reversed_timesteps[S - 1]
        noise = torch.randn_like(x0)

        xt = scheduler.add_noise(x0, noise, t)

        self.update_text_features(None, tgt_prompt=tgt_prompt)
        tgt_text_embedding = self.tgt_text_feature
        null_text_embedding = self.null_text_feature
        text_embeddings = torch.cat([tgt_text_embedding, null_text_embedding], dim=0)

        op = timesteps[-S:]

        for t in op:
            xt_input = torch.cat([xt] * 2)
            noise_pred = self.unet.forward(
                xt_input,
                torch.cat([t[None]] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)
            xt = self.reverse_step(noise_pred, t, xt, eta=eta)

        return xt

    def reverse_step(self, model_output, timestep, sample, eta=0, variance_noise=None):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        variance = self.get_variance(timestep)
        model_output_direction = model_output
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn_like(model_output)
            sigma_z = eta * variance ** (0.5) * variance_noise
            prev_sample = prev_sample + sigma_z
        return prev_sample

    def get_variance(self, timestep):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance


def tensor_to_pil(img):
    if img.ndim == 4:
        img = img[0]
    img = img.cpu().permute(1, 2, 0).detach().numpy()
    
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img


def pil_to_tensor(img, device="cpu"):
    device = torch.device(device)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img[None].transpose(0, 3, 1, 2))
    img = img.to(device)
    return img


def resize_image(image, min_size):
    if min(image.size) < min_size:
        image = image.resize((min_size, min_size))
    return image



class PDS_InstructPix2Pix(object):
    def __init__(self, config: PDSConfig, use_wandb=False):
        self.config = config
        self.device = torch.device(config.device)

        self.pipe = DiffusionPipeline.from_pretrained(config.sd_pretrained_model_or_path).to(self.device)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.set_timesteps(config.num_inference_steps)
        self.pipe.scheduler = self.scheduler

        self.unet = CustomUNet2DConditionModel.from_pretrained(
            config.sd_pretrained_model_or_path,
            subfolder="unet"
        ).to(self.device)
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae

        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        ## construct text features beforehand.
        self.src_prompt = self.config.src_prompt
        self.tgt_prompt = self.config.tgt_prompt

        self.update_text_features(src_prompt=self.src_prompt, tgt_prompt=self.tgt_prompt)
        self.null_text_feature = self.encode_text("")
    
        self.get_attn = False
        self.use_wandb = use_wandb
        self.use_freeu = self.config.use_freeu
        self.use_pds = self.config.use_pds
        self.lambda_identity = self.config.lambda_identity
        self.timestep_annealing = self.config.timestep_annealing

        self.iteration = 0
        self.max_iteration = 4000

        self.threshold = 0.2
        self.check = 0
        
    def compute_posterior_mean(self, xt, noise_pred, t, t_prev):
        """
        Computes an estimated posterior mean \mu_\phi(x_t, y; \epsilon_\phi).
        """
        device = self.device
        beta_t = self.scheduler.betas[t].to(device)
        alpha_t = self.scheduler.alphas[t].to(device)
        alpha_bar_t = self.scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = self.scheduler.alphas_cumprod[t_prev].to(device)

        pred_x0 = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        c0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
        c1 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        mean_func = c0 * pred_x0 + c1 * xt
        
        return mean_func, pred_x0

    def encode_image(self, img_tensor: Float[torch.Tensor, "B C H W"]):
        x = img_tensor
        x = 2 * x - 1
        x = x.float()
        return self.vae.encode(x).latent_dist.sample() * 0.18215
    
    def encode_src_image(self, img_tensor: Float[torch.Tensor, "B C H W"]):
        x = img_tensor.float()
        return self.vae.encode(x)

    def encode_text(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

    def decode_latent(self, latent):
        x = self.vae.decode(latent / 0.18215).sample
        x = (x / 2 + 0.5).clamp(0, 1)
        return x

    def update_text_features(self, src_prompt=None, tgt_prompt=None):
        if getattr(self, "src_text_feature", None) is None:
            assert src_prompt is not None
            self.src_prompt = src_prompt
            self.src_text_feature = self.encode_text(src_prompt)
        else:
            if src_prompt is not None and src_prompt != self.src_prompt:
                self.src_prompt = src_prompt
                self.src_text_feature = self.encode_text(src_prompt)

        if getattr(self, "tgt_text_feature", None) is None:
            assert tgt_prompt is not None
            self.tgt_prompt = tgt_prompt
            self.tgt_text_feature = self.encode_text(tgt_prompt)
        else:
            if tgt_prompt is not None and tgt_prompt != self.tgt_prompt:
                self.tgt_prompt = tgt_prompt
                self.tgt_text_feature = self.encode_text(tgt_prompt)

    def pds_timestep_sampling(self, batch_size):
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)

        min_step = 1 if self.config.min_step_ratio <= 0 else int(len(timesteps) * self.config.min_step_ratio)
        max_step = (
            len(timesteps) if self.config.max_step_ratio >= 1 else int(len(timesteps) * self.config.max_step_ratio)
        )
        max_step = max(max_step, min_step + 1)
        #random_int = torch.randint(low=min_step, high=max_step, size=(1,)).item()
        #idx = torch.full((batch_size,), random_int, dtype=torch.long,device='cpu')

        idx = torch.randint(
            min_step,
            max_step,
            [batch_size],
            dtype=torch.long,
            device="cpu",
        )
        t = timesteps[idx].cpu()
        t_prev = timesteps[idx - 1].cpu()

        return t, t_prev
    

    def pds_timestep_anneal_sampling(self, batch_size):
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)

        min_step = 1 if self.config.min_step_ratio <= 0 else int(len(timesteps) * self.config.min_step_ratio)
        max_step = (
            len(timesteps) if self.config.max_step_ratio >= 1 else int(len(timesteps) * self.config.max_step_ratio)
        )
        max_step = max(max_step, min_step + 1)

        idx = torch.full((batch_size,), (max_step-min_step)*((self.max_iteration-self.iteration)/self.max_iteration) + min_step, dtype=torch.long, device="cpu")

        t = timesteps[idx].cpu()
        t_prev = timesteps[idx - 1].cpu()
        #t_tau = timesteps[idx + tau].cpu()
        return t, t_prev
    
    def select_attention(
            self,
            attention_maps: torch.Tensor,
            prompt: str,
            indices_list: List[int] = None,
            normalize_eot_token: bool = False
        ) -> Dict[int, torch.Tensor]:
        last_token_index = -1
        
        if normalize_eot_token:
            if isinstance(prompt, list):
                prompt = prompt[0]
            last_token_index = len(self.tokenizer(prompt)["input_ids"]) - 1
            
        attention_maps = attention_maps[:, :, 1:last_token_index]
        attention_maps *= 100
        attention_maps = torch.nn.functional.softmax(attention_maps, dim=-1)
        
        attention_for_indices = {}
        
        if indices_list is not None:
            indices_list = [index - 1 for index in indices_list]
            for index in indices_list:
                attention_for_indices[index + 1] = attention_maps[:, :, index]
        else:
            for index in range(attention_maps.shape[-1]):
                attention_for_indices[index + 1] = attention_maps[:, :, index]            
    
        return attention_for_indices
    
    def get_cross_attn(self, h, w, attn_store, prompt, token_indices_list):
        l = min(h, w)
        
        token_attn_maps = {}
        
        agg_attns = aggregate_attention(
            attention_store=attn_store,
            h = round(self.config.cross_attn_res * h / l),
            w = round(self.config.cross_attn_res * w / l),
            from_where=self.config.cross_attn_layer_list,
            is_cross=True,
            batch_select=0
        )

        selected_token_attn_maps = self.select_attention(agg_attns, prompt, token_indices_list)
        
        for token_index in selected_token_attn_maps.keys():
            token_attn_map = F.interpolate(selected_token_attn_maps[token_index].unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0).squeeze(0)
            
            token_attn_map = (token_attn_map - token_attn_map.min()) / (token_attn_map.max() - token_attn_map.min() + 1e-8)
            token_attn_maps[token_index] = token_attn_map

        return token_attn_maps

    def __call__(
        self,
        tgt_x0,
        src_x0,
        src_emb,
        tgt_prompt=None,
        src_prompt=None,
        reduction="mean",
        return_dict=False,
        step=0,
    ):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(src_prompt=src_prompt, tgt_prompt=tgt_prompt)
        tgt_text_embedding, src_text_embedding = (
            self.tgt_text_feature,
            self.src_text_feature,
        )
        uncond_embedding = self.null_text_feature

        batch_size = tgt_x0.shape[0]
        if self.timestep_annealing:
            t, t_prev = self.pds_timestep_anneal_sampling(batch_size)
        else:
            t, t_prev = self.pds_timestep_sampling(batch_size)
        
        beta_t = scheduler.betas[t].to(device)
        alpha_t = scheduler.alphas[t].to(device)
        alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = scheduler.alphas_cumprod[t_prev].to(device)
        
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)

        noise = torch.randn_like(tgt_x0)
        noise_t_prev = torch.randn_like(tgt_x0)

        zts = dict()
        pred_x0s = dict()
        noise_preds = dict()
        eps = dict()

        for latent, cond_text_embedding, name in zip(
            [tgt_x0, src_x0], [tgt_text_embedding, src_text_embedding], ["tgt", "src"]
        ):
            latents_noisy = scheduler.add_noise(latent, noise, t)
            

            text_embeddings = torch.cat([cond_text_embedding, uncond_embedding, uncond_embedding], dim=0)
            text_embeddings = torch.cat([text_embeddings, text_embeddings], dim=1)
            max_step = int(len(reversed(self.scheduler.timesteps)) * self.config.max_step_ratio)
            min_step = int(len(reversed(self.scheduler.timesteps)) * self.config.min_step_ratio)
            
            #b1 = (t[0] - min_step) / (max_step - min_step) * 0.4 + 1.1
            #b2 = (t[0] - min_step) / (max_step - min_step) * 0.4 + 1.1

            ################################    FreeU   ################################
            if self.use_freeu:
                b1 = 1.1
                b2 = 1.1
                s1 = 0.9
                s2 = 0.2

                if self.check == 0:
                    register_free_upblock2d_in(
                        self.unet,
                        b1=b1, 
                        b2=b2, 
                        s1=s1, 
                        s2=s2,
                        check = self.check
                    )
                    register_free_crossattn_upblock2d_in(
                        self.unet, 
                        b1=b1,
                        b2=b2,
                        s1=s1,
                        s2=s2,
                        check = self.check
                    )
                    self.check = 1
                else:
                    register_free_upblock2d_in(
                        self.unet,
                        b1=b1, 
                        b2=b2, 
                        s1=s1, 
                        s2=s2,
                    )
                    register_free_crossattn_upblock2d_in(
                        self.unet, 
                        b1=b1,
                        b2=b2,
                        s1=s1,
                        s2=s2,
                    )

            ############################################################################
            src_encoded = src_emb.latent_dist.mode()
            
            uncond_image_latent = torch.zeros_like(src_encoded)
            latent_image = torch.cat([src_encoded, src_encoded, uncond_image_latent], dim=0)
            latent_model_input = torch.cat([latents_noisy] * 3, dim=0)
            latent_model_input = torch.cat([latent_model_input, latent_image], dim=1)
            
            unet_outputs = self.unet.forward(
                latent_model_input,
                torch.cat([t] * 3).to(device),
                encoder_hidden_states=text_embeddings,
            )
            
            noise_pred = unet_outputs.sample
            unet_feats = unet_outputs.features
            
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            
            if not self.use_pds:
                if name == "tgt":
                    noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_image) + \
                        self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_uncond + self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            else:
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)
            eps[name] = noise_pred
        
            x_t_prev = scheduler.add_noise(latent, noise_t_prev, t_prev)
            mu, pred_x0 = self.compute_posterior_mean(latents_noisy, noise_pred, t, t_prev)
            
            zt = (x_t_prev - mu) / sigma_t
            zts[name] = zt
            pred_x0s[name] = pred_x0
            noise_preds[name] = noise_pred

        if self.use_pds:
            grad = zts["tgt"] - zts["src"]
        else:
            if self.timestep_annealing:
                # grad = eps["tgt"] - eps["src"] + (tgt_x0 - src_x0) * self.lambda_identity * (1. - t / self.scheduler.config.num_train_timesteps).to(self.device)  # 0.1
                grad = eps["tgt"] - eps["src"] + (0.2 * (((self.max_iteration - self.iteration) / self.max_iteration) ** 2) + 0.02) * (tgt_x0 - src_x0)
            else:
                grad = eps["tgt"] - eps["src"] + (tgt_x0 - src_x0) * self.lambda_identity
       
        grad = torch.nan_to_num(grad)
        
        target = (tgt_x0 - grad).detach()
        loss = 0.5 * F.mse_loss(tgt_x0, target, reduction=reduction) / batch_size #we could change this to MSE of each term
        
        self.iteration += 1
        
        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t}
            return dic
        else:
            return loss

    def run_sdedit(self, x0, tgt_prompt=None, num_inference_steps=20, skip=7, eta=0):
        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        reversed_timesteps = reversed(scheduler.timesteps)

        S = num_inference_steps - skip
        t = reversed_timesteps[S - 1]
        noise = torch.randn_like(x0)

        xt = scheduler.add_noise(x0, noise, t)

        self.update_text_features(None, tgt_prompt=tgt_prompt)
        tgt_text_embedding = self.tgt_text_feature
        null_text_embedding = self.null_text_feature
        text_embeddings = torch.cat([tgt_text_embedding, null_text_embedding], dim=0)

        op = timesteps[-S:]

        for t in op:
            xt_input = torch.cat([xt] * 2)
            noise_pred = self.unet.forward(
                xt_input,
                torch.cat([t[None]] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)
            xt = self.reverse_step(noise_pred, t, xt, eta=eta)

        return xt

    def reverse_step(self, model_output, timestep, sample, eta=0, variance_noise=None):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        variance = self.get_variance(timestep)
        model_output_direction = model_output
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn_like(model_output)
            sigma_z = eta * variance ** (0.5) * variance_noise
            prev_sample = prev_sample + sigma_z
        return prev_sample

    def get_variance(self, timestep):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
