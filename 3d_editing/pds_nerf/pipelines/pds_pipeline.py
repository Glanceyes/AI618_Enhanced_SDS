from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from PIL import Image
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from pds_nerf.pipelines.base_pipeline import ModifiedVanillaPipeline
from pds_nerf.data.datamanagers.pds_datamanager import PDSDataManagerConfig
from pds.pds import PDS, PDS_InstructPix2Pix, PDSConfig, tensor_to_pil
from pds.utils.imageutil import merge_images
from pds.utils.sysutil import clean_gpu
from pds.utils.free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d

cmap = plt.get_cmap("viridis")


@dataclass
class PDSPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: PDSPipeline)
    datamanager: PDSDataManagerConfig = PDSDataManagerConfig()

    # PDS configs.
    pds: PDSConfig = PDSConfig()
    pds_device: Optional[Union[torch.device, str]] = None

    use_freeu: bool = False
    pds_loss_mult: float = 1.0
    pds_freeu_b1: float = 0.7
    pds_freeu_b2: float = 0.8
    pds_freeu_s1: float = 0.9
    pds_freeu_s2: float = 0.2
    pds_freeu_timestep: int = 500
    pds_freeu_t_b1: float = 0.9
    pds_freeu_t_b2: float = 0.9

    change_view_step: int = 10
    log_step: int = 10
    only_pds: bool = False


class PDSPipeline(ModifiedVanillaPipeline):
    config: PDSPipelineConfig

    def __init__(
        self,
        config: PDSPipelineConfig,
        device: Union[str, torch.device],
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler, **kwargs)

        # Construct PDS
        self.pds_device = (
            torch.device(device) if self.config.pds_device is None else torch.device(self.config.pds_device)
        )
        self.config.pds.device = self.pds_device
        self.use_wandb = kwargs.get("wandb_enabled", True)
        self.only_pds = self.config.only_pds
        
        if self.only_pds:
            self.pds = PDS(self.config.pds, use_wandb=self.use_wandb)
        else:
            self.pds = PDS_InstructPix2Pix(self.config.pds, use_wandb=self.use_wandb)
        
        if self.config.use_freeu:
            register_free_upblock2d(
                self.pds, 
                b1=self.config.pds_freeu_b1, 
                b2=self.config.pds_freeu_b2, 
                s1=self.config.pds_freeu_s1, 
                s2=self.config.pds_freeu_s2,
                timestep=self.config.pds_freeu_timestep,
                t_b1=self.config.pds_freeu_t_b1,
                t_b2=self.config.pds_freeu_t_b2
            )
            register_free_crossattn_upblock2d(
                self.pds, 
                b1=self.config.pds_freeu_b1, 
                b2=self.config.pds_freeu_b2, 
                s1=self.config.pds_freeu_s1, 
                s2=self.config.pds_freeu_s2,
                timestep=self.config.pds_freeu_timestep,
                t_b1=self.config.pds_freeu_t_b1,
                t_b2=self.config.pds_freeu_t_b2
            )

        # Caching source's x0
        self.src_x0s = dict()
        self.current_spot = None
        
    def get_current_rendering(self, step):
        if getattr(self, "current_spot", None) is None or step % self.config.change_view_step == 0:
            self.current_spot = np.random.randint(len(self.datamanager.train_dataparser_outputs.image_filenames))
        current_spot = self.current_spot
        current_index = self.datamanager.image_batch["image_idx"][current_spot]
        current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index:current_index+1].to(self.device)
        camera_outputs = self.model.diff_get_outputs_for_camera(current_camera)
        rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)  # [B,3,H,W]

        # delete to free up memory
        del camera_outputs
        del current_camera
        clean_gpu()

        return rendered_image, current_spot

    def get_train_loss_dict(self, step: int):
        loss_dict = dict()

        rendered_image, current_spot = self.get_current_rendering(step)
        # get original image from dataloader
        original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device)
        original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)

        h, w = original_image.shape[2:]
        l = min(h, w)
        h = int(h * 512 / l)
        w = int(w * 512 / l)  # resize an image such that the smallest length is 512.
        original_image_512 = F.interpolate(original_image, size=(h, w), mode="bilinear")
        rendered_image_512 = F.interpolate(rendered_image, size=(h, w), mode="bilinear")

        if current_spot not in self.src_x0s.keys():
            with torch.no_grad():
                src_x0 = self.pds.encode_image(original_image_512.to(self.pds_device))
                self.src_x0s[current_spot] = src_x0.clone().cpu()
        else:
            src_x0 = self.src_x0s[current_spot].to(self.pds_device)

        x0 = self.pds.encode_image(rendered_image_512.to(self.pds_device))
        if not self.only_pds:
            src_emb = self.pds.encode_src_image(original_image_512.to(self.pds_device))

        del rendered_image_512
        del original_image_512
        clean_gpu()

        if self.only_pds:
            dic = self.pds(tgt_x0=x0, src_x0=src_x0, return_dict=True, step=step)
        else:
            dic = self.pds(tgt_x0=x0, src_x0=src_x0, src_emb=src_emb, return_dict=True, step=step)
        
        grad = dic["grad"].cpu()
        loss = dic["loss"] * self.config.pds_loss_mult
        loss = loss.to(self.device)
        loss_dict["pds_loss"] = loss
        
        vis_grad = self.visualize_grad(grad, w, h)
        
        min_size = 512
        original_image_resized = TF.resize(original_image[0], min_size)
        vis_grad_resized = TF.resize(vis_grad, min_size)
        rendered_image_resized = TF.resize(rendered_image[0], min_size)
        
        if self.use_wandb:
            import wandb
            wandb.log({
                "grad": wandb.Image(vis_grad_resized),
                "original_image": wandb.Image(original_image_resized.permute(1, 2, 0).detach().cpu().numpy()),
                "rendered_image": wandb.Image(rendered_image_resized.permute(1, 2, 0).detach().cpu().numpy()),
            }, step=step, commit=False) if step % self.config.log_step == 0 else None
            
            wandb.log({
                "pds_loss": loss.item(),
            }, step=step, commit=False)

        # logging
        if step % self.config.log_step == 0:
            self.log_images(rendered_image, original_image, grad, step)

        return None, loss_dict, dict()

    @torch.no_grad()
    def log_images(self, rendered_image, original_image, grad, step):
        edit_img = tensor_to_pil(rendered_image)
        orig_img = tensor_to_pil(original_image)

        w, h = edit_img.size
        
        vis_grad = grad if isinstance(grad, Image.Image) else self.visualize_grad(grad, w, h)
        
        img = merge_images([orig_img, edit_img, vis_grad])
        img.save(self.base_dir / f"logging/{step}.png")
        
    def visualize_grad(self, grad, w, h):
        vis_grad = grad.norm(dim=1).clone().detach().cpu()
        vis_grad = vis_grad / vis_grad.max()
        vis_grad = vis_grad.clamp(0, 1).squeeze().numpy()
        vis_grad = cmap(vis_grad)[..., :3]
        vis_grad = Image.fromarray((vis_grad * 255).astype(np.uint8))
        vis_grad = vis_grad.resize((w, h), resample=Image.Resampling.NEAREST)
        
        return vis_grad

    # to enable backprop.
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle):
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.model.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.model.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs