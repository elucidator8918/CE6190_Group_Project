"""
LEDITS: Real Image Editing with DDPM Inversion and SEGA
Single-file implementation for easy import and use
"""

import uuid
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Callable, List, Optional, Union
from itertools import repeat
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from torch import autocast, inference_mode


class SemanticStableDiffusionPipeline(DiffusionPipeline):
    """Pipeline for SEGA with DDPM inversion support"""
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
        editing_prompt: Optional[Union[str, List[str]]] = None,
        editing_prompt_embeddings: Optional[torch.Tensor] = None,
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
        edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
        edit_warmup_steps: Optional[Union[int, List[int]]] = 10,
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
        edit_threshold: Optional[Union[float, List[float]]] = 0.9,
        edit_momentum_scale: Optional[float] = 0.1,
        edit_mom_beta: Optional[float] = 0.4,
        edit_weights: Optional[List[float]] = None,
        sem_guidance: Optional[List[torch.Tensor]] = None,
        use_ddpm: bool = False,
        wts: Optional[List[torch.Tensor]] = None,
        zs: Optional[List[torch.Tensor]] = None
    ):
        """Generate edited images with SEGA and optional DDPM inversion"""
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        
        # Configure editing
        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            enabled_editing_prompts = len(editing_prompt)
        elif editing_prompt_embeddings is not None:
            enable_edit_guidance = True
            enabled_editing_prompts = editing_prompt_embeddings.shape[0]
        else:
            enabled_editing_prompts = 0
            enable_edit_guidance = False
        
        # Encode prompts
        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(batch_size * num_images_per_prompt, -1, text_embeddings.shape[-1])
        
        # Encode editing prompts
        if enable_edit_guidance and editing_prompt_embeddings is None:
            edit_concepts_input = self.tokenizer(
                [x for item in editing_prompt for x in repeat(item, batch_size)],
                padding="max_length", max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            edit_concepts = self.text_encoder(edit_concepts_input.input_ids.to(self.device))[0]
            edit_concepts = edit_concepts.repeat(1, num_images_per_prompt, 1)
            edit_concepts = edit_concepts.view(-1, edit_concepts.shape[1], edit_concepts.shape[-1])
        elif enable_edit_guidance:
            edit_concepts = editing_prompt_embeddings.to(self.device).repeat(batch_size, 1, 1)
            edit_concepts = edit_concepts.repeat(1, num_images_per_prompt, 1)
            edit_concepts = edit_concepts.view(-1, edit_concepts.shape[1], edit_concepts.shape[-1])
        
        # Classifier-free guidance setup
        do_classifier_free_guidance = guidance_scale > 1.0
        
        if do_classifier_free_guidance:
            uncond_tokens = [""] if negative_prompt is None else (
                [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            )
            uncond_input = self.tokenizer(
                uncond_tokens, padding="max_length", max_length=text_embeddings.shape[1],
                truncation=True, return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            uncond_embeddings = uncond_embeddings.repeat(batch_size, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, -1, uncond_embeddings.shape[-1])
            
            if enable_edit_guidance:
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings, edit_concepts])
            else:
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        if use_ddpm:
            t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}
            timesteps = timesteps[-zs.shape[0]:]
        
        # Prepare latents
        if latents is None:
            shape = (batch_size * num_images_per_prompt, self.unet.config.in_channels,
                    height // self.vae_scale_factor, width // self.vae_scale_factor)
            latents = torch.randn(shape, generator=generator, device=self.device, dtype=text_embeddings.dtype)
            latents = latents * self.scheduler.init_noise_sigma
        else:
            latents = latents.to(self.device)
        
        # Initialize edit momentum
        edit_momentum = None
        
        # Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = (
                torch.cat([latents] * (2 + enabled_editing_prompts)) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Apply guidance
            if do_classifier_free_guidance:
                noise_pred_out = noise_pred.chunk(2 + enabled_editing_prompts)
                noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]
                noise_pred_edit_concepts = noise_pred_out[2:]
                
                noise_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # SEGA guidance
                if enable_edit_guidance:
                    if edit_momentum is None:
                        edit_momentum = torch.zeros_like(noise_guidance)
                    
                    concept_weights = torch.zeros(
                        (len(noise_pred_edit_concepts), noise_guidance.shape[0]),
                        device=self.device, dtype=noise_guidance.dtype
                    )
                    noise_guidance_edit = torch.zeros(
                        (len(noise_pred_edit_concepts), *noise_guidance.shape),
                        device=self.device, dtype=noise_guidance.dtype
                    )
                    
                    warmup_inds = []
                    for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):
                        edit_guidance_scale_c = edit_guidance_scale[c] if isinstance(edit_guidance_scale, list) else edit_guidance_scale
                        edit_threshold_c = edit_threshold[c] if isinstance(edit_threshold, list) else edit_threshold
                        reverse_editing_direction_c = reverse_editing_direction[c] if isinstance(reverse_editing_direction, list) else reverse_editing_direction
                        edit_weight_c = edit_weights[c] if edit_weights else 1.0
                        edit_warmup_steps_c = edit_warmup_steps[c] if isinstance(edit_warmup_steps, list) else edit_warmup_steps
                        edit_cooldown_steps_c = edit_cooldown_steps[c] if isinstance(edit_cooldown_steps, list) else (i + 1 if edit_cooldown_steps is None else edit_cooldown_steps)
                        
                        if i >= edit_warmup_steps_c:
                            warmup_inds.append(c)
                        if i >= edit_cooldown_steps_c:
                            continue
                        
                        noise_guidance_edit_tmp = noise_pred_edit_concept - noise_pred_uncond
                        tmp_weights = torch.full_like(
                            (noise_guidance - noise_pred_edit_concept).sum(dim=(1, 2, 3)),
                            edit_weight_c
                        )
                        
                        if reverse_editing_direction_c:
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1
                        
                        concept_weights[c, :] = tmp_weights
                        noise_guidance_edit_tmp = noise_guidance_edit_tmp * edit_guidance_scale_c
                        
                        # Apply threshold
                        tmp = torch.quantile(
                            torch.abs(noise_guidance_edit_tmp).flatten(start_dim=2).float(),
                            edit_threshold_c, dim=2, keepdim=False
                        ).to(noise_guidance_edit_tmp.dtype)
                        
                        noise_guidance_edit_tmp = torch.where(
                            torch.abs(noise_guidance_edit_tmp) >= tmp[:, :, None, None],
                            noise_guidance_edit_tmp,
                            torch.zeros_like(noise_guidance_edit_tmp)
                        )
                        noise_guidance_edit[c, :, :, :, :] = noise_guidance_edit_tmp
                    
                    # Combine editing guidance
                    concept_weights = torch.where(
                        concept_weights < 0, torch.zeros_like(concept_weights), concept_weights
                    )
                    concept_weights = concept_weights / (concept_weights.sum(dim=0, keepdim=True) + 1e-8)
                    noise_guidance_edit = torch.einsum("cb,cbijk->bijk", concept_weights, noise_guidance_edit)
                    noise_guidance_edit = noise_guidance_edit + edit_momentum_scale * edit_momentum
                    edit_momentum = edit_mom_beta * edit_momentum + (1 - edit_mom_beta) * noise_guidance_edit
                    
                    if len(warmup_inds) == len(noise_pred_edit_concepts):
                        noise_guidance = noise_guidance + noise_guidance_edit
                
                noise_pred = noise_pred_uncond + noise_guidance
            
            # DDPM reverse step
            if use_ddpm:
                idx = t_to_idx[int(t)]
                z = zs[idx] if zs is not None else None
                
                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t
                
                pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
                
                beta_prod_t_prev = 1 - alpha_prod_t_prev
                variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
                
                pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** 0.5 * noise_pred
                prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
                
                if eta > 0:
                    if z is None:
                        z = torch.randn(noise_pred.shape, device=self.device)
                    sigma_z = eta * variance ** 0.5 * z
                    latents = prev_sample + sigma_z
                else:
                    latents = prev_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample
            
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        
        # Decode latents
        if output_type != "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, self.device, text_embeddings.dtype)
        else:
            image = latents
            has_nsfw_concept = None
        
        do_denormalize = [True] * image.shape[0] if has_nsfw_concept is None else [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        
        if not return_dict:
            return (image, has_nsfw_concept)
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept


class LEDITSEditor:
    """LEDITS image editor combining DDPM inversion and SEGA"""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pipelines
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(model_id).to(self.device)
        self.sd_pipe.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
        self.sega_pipe = SemanticStableDiffusionPipeline.from_pretrained(model_id).to(self.device)
        
    def load_image(self, image_path, size=512):
        """Load and preprocess image"""
        if isinstance(image_path, str):
            image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
        else:
            image = np.array(image_path)
            
        h, w, c = image.shape
        
        # Center crop to square
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
            
        image = np.array(Image.fromarray(image).resize((size, size)))
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return image
    
    def encode_text(self, prompts):
        """Encode text prompts"""
        text_input = self.sd_pipe.tokenizer(
            prompts, padding="max_length", max_length=self.sd_pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            text_encoding = self.sd_pipe.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding
    
    def invert(self, x0, prompt_src="", num_inference_steps=100, cfg_scale_src=3.5, eta=1):
        """DDPM inversion"""
        self.sd_pipe.scheduler.set_timesteps(num_inference_steps)
        
        with autocast("cuda"), inference_mode():
            w0 = (self.sd_pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()
        
        wt, zs, wts = self._inversion_forward_process(
            self.sd_pipe, w0, etas=eta, prompt=prompt_src, 
            cfg_scale=cfg_scale_src, num_inference_steps=num_inference_steps
        )
        return zs, wts
    
    def _inversion_forward_process(self, model, x0, etas=None, prompt="", 
                                   cfg_scale=3.5, num_inference_steps=50):
        """Forward inversion process"""
        uncond_embedding = self.encode_text("")
        if prompt:
            text_embeddings = self.encode_text(prompt)
            
        timesteps = model.scheduler.timesteps.to(model.device)
        
        if etas is None or etas == 0:
            eta_is_zero = True
            zs = None
        else:
            eta_is_zero = False
            if isinstance(etas, (int, float)):
                etas = [etas] * model.scheduler.num_inference_steps
            xts = self._sample_xts_from_x0(model, x0, num_inference_steps)
            alpha_bar = model.scheduler.alphas_cumprod
            variance_noise_shape = (num_inference_steps, model.unet.in_channels, 
                                   model.unet.sample_size, model.unet.sample_size)
            zs = torch.zeros(size=variance_noise_shape, device=model.device)
        
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xt = x0
        
        for t in tqdm(reversed(timesteps), desc="Inverting"):
            idx = t_to_idx[int(t)]
            
            if not eta_is_zero:
                xt = xts[idx][None]
            
            with torch.no_grad():
                out = model.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)
                if prompt:
                    cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states=text_embeddings)
            
            if prompt:
                noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
            else:
                noise_pred = out.sample
            
            if eta_is_zero:
                xt = self._forward_step(model, noise_pred, t, xt)
            else:
                xtm1 = xts[idx + 1][None]
                pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5
                
                prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
                alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
                
                variance = self._get_variance(model, t)
                pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** 0.5 * noise_pred
                mu_xt = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
                
                z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
                zs[idx] = z
                
                xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
                xts[idx + 1] = xtm1
        
        if zs is not None:
            zs[-1] = torch.zeros_like(zs[-1])
        
        return xt, zs, xts
    
    def _sample_xts_from_x0(self, model, x0, num_inference_steps=50):
        """Sample intermediate noisy latents"""
        alpha_bar = model.scheduler.alphas_cumprod
        sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        variance_noise_shape = (num_inference_steps, model.unet.in_channels,
                               model.unet.sample_size, model.unet.sample_size)
        
        timesteps = model.scheduler.timesteps.to(model.device)
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(variance_noise_shape).to(x0.device)
        
        for t in reversed(timesteps):
            idx = t_to_idx[int(t)]
            xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
        
        xts = torch.cat([xts, x0], dim=0)
        return xts
    
    def _forward_step(self, model, model_output, timestep, sample):
        """Forward diffusion step"""
        next_timestep = min(model.scheduler.config.num_train_timesteps - 2,
                           timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)
        
        alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample = model.scheduler.add_noise(pred_original_sample, model_output, torch.LongTensor([next_timestep]))
        return next_sample
    
    def _get_variance(self, model, timestep):
        """Get variance for diffusion step"""
        prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
    
    def edit(self, wts, zs, target_prompt="", steps=100, skip=36, 
             target_cfg_scale=15, edit_concepts=None, guidance_scales=None,
             warmup_steps=None, reverse_editing=None, thresholds=None):
        """Edit image using SEGA"""
        edit_concepts = edit_concepts or []
        guidance_scales = guidance_scales or [7] * len(edit_concepts)
        warmup_steps = warmup_steps or [1] * len(edit_concepts)
        reverse_editing = reverse_editing or [False] * len(edit_concepts)
        thresholds = thresholds or [0.95] * len(edit_concepts)
        
        editing_args = dict(
            editing_prompt=edit_concepts,
            reverse_editing_direction=reverse_editing,
            edit_warmup_steps=warmup_steps,
            edit_guidance_scale=guidance_scales,
            edit_threshold=thresholds,
            edit_momentum_scale=0.5,
            edit_mom_beta=0.6,
            eta=1,
        )
        
        latents = wts[skip].expand(1, -1, -1, -1)
        sega_out = self.sega_pipe(
            prompt=target_prompt,
            latents=latents,
            guidance_scale=target_cfg_scale,
            num_images_per_prompt=1,
            num_inference_steps=steps,
            use_ddpm=True,
            wts=wts,
            zs=zs[skip:],
            **editing_args
        )
        return sega_out.images[0]

def edit_image(
    input_image: Union[str, Image.Image],
    target_prompt: str = "",
    edit_concepts: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    source_prompt: str = "",
    num_steps: int = 100,
    source_guidance: float = 3.5,
    target_guidance: float = 15,
    skip_steps: int = 36,
    guidance_scales: Optional[List[float]] = None,
    warmup_steps: Optional[List[int]] = None,
    reverse_editing: Optional[List[bool]] = None,
    thresholds: Optional[List[float]] = None,
    seed: Optional[int] = None,
    model_id: str = "runwayml/stable-diffusion-v1-5"
) -> tuple:
    """
    Edit an image using LEDITS (DDPM Inversion + SEGA)
    
    Args:
        input_image: Path to input image or PIL Image
        target_prompt: Target prompt describing desired output
        edit_concepts: List of concepts to add/remove (SEGA)
        output_path: Where to save edited image (optional)
        source_prompt: Source prompt for inversion (default: "")
        num_steps: Number of diffusion steps (default: 100)
        source_guidance: CFG scale for inversion (default: 3.5)
        target_guidance: CFG scale for editing (default: 15)
        skip_steps: Steps to skip in reverse process (default: 36)
        guidance_scales: SEGA guidance scales per concept
        warmup_steps: SEGA warmup steps per concept
        reverse_editing: Whether to remove (True) or add (False) each concept
        thresholds: SEGA thresholds per concept
        seed: Random seed for reproducibility
        model_id: Hugging Face model ID
    
    Returns:
        Tuple of (edited_image, saved_path)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    editor = LEDITSEditor(model_id=model_id)
    
    # Load and invert
    x0 = editor.load_image(input_image)
    print("Inverting image...")
    zs, wts = editor.invert(
        x0,
        prompt_src=source_prompt, 
        num_inference_steps=num_steps,
        cfg_scale_src=source_guidance
    )
    
    # Edit
    print("Editing image...")
    edited_img = editor.edit(
        wts, zs,
        target_prompt=target_prompt,
        steps=num_steps,
        skip=skip_steps,
        target_cfg_scale=target_guidance,
        edit_concepts=edit_concepts,
        guidance_scales=guidance_scales,
        warmup_steps=warmup_steps,
        reverse_editing=reverse_editing,
        thresholds=thresholds
    )
    
    # Generate unique filename if none provided
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Generate unique filename
    unique_id = uuid.uuid4().hex[:8]
    save_path = str(Path(output_dir) / f"edited_{unique_id}.png")
    
    edited_img.save(save_path)
    print(f"Saved edited image to: {save_path}")
    
    return edited_img, save_path