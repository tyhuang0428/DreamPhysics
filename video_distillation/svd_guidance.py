from typing import Union, List
from jaxtyping import Float, Int

import PIL
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from diffusers import DDIMScheduler, StableVideoDiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing
from diffusers.utils.import_utils import is_xformers_available

from utils.threestudio_utils import parse_version, cleanup, get_device, C


class SVDGuidance:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device()
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                print(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                print(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.image_encoder = self.pipe.image_encoder.eval()
        self.image_processor = self.pipe.image_processor
        self.feature_extractor = self.pipe.feature_extractor

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val = None

        # Extra for latents
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # set spatial size
        # to save GPU memory, we use (256, 256) here
        # SVD supports the maximum resolution of 576x1024, so you can set (576, 1024) if you have enough GPU memory
        self.spatial_size = (256, 256)
        
    def get_add_time_ids(
        self,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int = 1,
        fps: int = 6,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        do_classifier_free_guidance: bool = True,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        added_time_ids = self.get_add_time_ids(
            self.weights_dtype, 1
        )
        added_time_ids = added_time_ids.to(self.device)
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            added_time_ids=added_time_ids
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_vae_images(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device]
    ):
        input_dtype = image.dtype
        image = image.to(device).to(self.weights_dtype)  # [25, 3, 576, 1024]
        posterior = self.vae.encode(image).latent_dist#.mode().unsqueeze(0)
        image_latents = posterior.sample() * self.vae.config.scaling_factor
        return image_latents.unsqueeze(0).to(input_dtype)
    
    @torch.cuda.amp.autocast(enabled=False)
    def encode_image(
        self,
        image,
        device: Union[str, torch.device],
    ) -> torch.FloatTensor:
        input_dtype = image.dtype
        dtype = next(self.image_encoder.parameters()).dtype
        
        image = image * 2.0 - 1.0
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0

        # Normalize the image with for CLIP input
        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)
        
        image_embeddings = torch.cat([image_embeddings, image_embeddings])

        return image_embeddings.to(input_dtype)
    
    def pre_process(self, image):
        image = self.image_processor.pil_to_numpy(image)
        image = self.image_processor.numpy_to_pt(image)
        return image

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        image_latents: Float[Tensor, "B 4 64 64"],
        image_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
    ):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latents_noisy = torch.cat([latents_noisy, image_latents], dim=2)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=image_embeddings,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )
        grad = w * (noise_pred - noise)
        return grad

    def __call__(
        self,
        rgb_BCHW: Float[Tensor, "B H W C"],
        image: Union[PIL.Image.Image, np.ndarray, torch.FloatTensor, List[PIL.Image.Image], List[np.ndarray], List[torch.FloatTensor]],
        num_frames: int = 25,
    ):
        image = self.pre_process(image)
        batch_size = rgb_BCHW.shape[0] // num_frames
        rgb_BCHW_256 = F.interpolate(
            rgb_BCHW, self.spatial_size, mode="bilinear", align_corners=False
        )
        image_BCHW_256 = F.interpolate(
            image, self.spatial_size, mode="bilinear", align_corners=False
        )
        latents = self.encode_vae_images(rgb_BCHW_256, self.device)
        image_latents = self.encode_vae_images(image_BCHW_256, self.device)
        image_latents = image_latents.repeat(1, num_frames, 1, 1, 1)
        
        image_embeddings = self.encode_image(image, self.device)
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        grad = self.compute_grad_sds(latents, image_latents, image_embeddings, t)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad

        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds_video": loss_sds
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        # t annealing from ProlificDreamer
        if (
            self.cfg.anneal_start_step is not None
            and global_step > self.cfg.anneal_start_step
        ):
            self.max_step = int(
                self.num_train_timesteps * self.cfg.max_step_percent_annealed
            )
