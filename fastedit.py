import time
import random
import os
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from diffusers import StableDiffusionImageVariationPipeline
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig
from accelerate import Accelerator


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def numpy_to_pil(images):
    """Convert numpy array to PIL image."""
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def load_and_preprocess_image(image_path, target_size=512):
    """Load and preprocess image for editing."""
    im = Image.open(image_path).convert("RGB")
    
    # Transform for diffusion model
    img_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    
    # Transform for CLIP conditioning
    cond_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
    
    img_base = img_transform(im)
    img = 2. * img_base - 1.
    cond = cond_transform(img_base)
    
    return img, cond


@torch.no_grad()
def encode_image(image, device, image_encoder, feature_extractor):
    """Encode image to CLIP embeddings."""
    dtype = next(image_encoder.parameters()).dtype
    
    if not isinstance(image, torch.Tensor):
        image = feature_extractor(images=image, return_tensors="pt").pixel_values
    
    image = image.to(device=device, dtype=dtype)
    image_embeddings = image_encoder(image).image_embeds
    image_embeddings = image_embeddings.unsqueeze(1)
    
    return image_embeddings


@torch.no_grad()
def sample_image(image_embeddings, device, unet, scheduler, guidance_scale=1.0, seed=0):
    """Generate image from embeddings using the diffusion model."""
    do_classifier_free_guidance = guidance_scale > 1.0
    
    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_embeddings)
        target_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])
    else:
        target_embeddings = image_embeddings
    
    # Generate initial latent noise
    latents_shape = (1, unet.config.in_channels, 512 // 8, 512 // 8)
    latents_dtype = target_embeddings.dtype
    
    if seed != -1:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None
    
    latents = torch.randn(latents_shape, generator=generator, device=device, dtype=latents_dtype)
    
    # Set timesteps
    scheduler.set_timesteps(50)
    timesteps_tensor = scheduler.timesteps.to(device)
    
    # Denoising loop
    for t in tqdm(timesteps_tensor, desc="Sampling", leave=False):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=target_embeddings).sample
        
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents


def decode_image(latents, vae):
    """Decode latents to image."""
    latents = 1 / vae.config.scaling_factor * latents
    samples = vae.decode(latents, return_dict=False)[0]
    ims = (samples / 2 + 0.5).clamp(0, 1)
    x_sample = ims.cpu().permute(0, 2, 3, 1).float().numpy()
    image = numpy_to_pil(x_sample)[0]
    return image


def configure_lora_model(unet, device, learning_rate=4e-4):
    """Configure UNet with LoRA for efficient fine-tuning."""
    for param in unet.parameters():
        param.requires_grad_(False)
    
    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    unet.add_adapter(unet_lora_config)
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    
    optimizer = torch.optim.AdamW(lora_layers, lr=learning_rate)
    return unet, optimizer


def calculate_timesteps(similarity_score):
    """Calculate timesteps based on semantic similarity."""
    timestep_values = {
        'high': [200, 300, 400, 600],
        'mid': [200, 400, 600, 800],
        'low': [300, 500, 600, 800]
    }
    
    if similarity_score <= 0.18:
        return timestep_values['low'], "High Timesteps (Low Similarity)"
    elif similarity_score <= 0.24:
        return timestep_values['mid'], "Mid Timesteps (Medium Similarity)"
    else:
        return timestep_values['high'], "Low Timesteps (High Similarity)"


def fine_tune_model(unet, optimizer, scheduler, image_latents, image_embeddings, 
                   timestep_values, num_steps=50, batch_size=4, accelerator=None):
    """Fine-tune the model on the input image."""
    image_latent = image_latents.repeat(batch_size, 1, 1, 1)
    image_emb = image_embeddings.repeat(batch_size, 1, 1)
    
    progress_bar = tqdm(range(num_steps), desc="Fine-tuning")
    history = []
    
    for step in range(num_steps):
        unet.train()
        
        with accelerator.accumulate(unet):
            # Sample noise
            batch_noise = torch.randn((batch_size,) + image_latents.shape[1:]).to(image_latents.device)
            batch_timesteps = torch.tensor(timestep_values, device=image_latents.device)
            
            # Add noise to latents
            noisy_latents = scheduler.add_noise(image_latent, batch_noise, batch_timesteps)
            
            # Predict noise
            noise_pred = unet(noisy_latents, batch_timesteps, image_emb).sample
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, batch_noise, reduction="none").mean([1, 2, 3]).mean()
            accelerator.backward(loss)
            
            optimizer.step()
            optimizer.zero_grad()
        
        history.append(loss.detach().item())
        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.detach().item())
    
    progress_bar.close()
    return history


def generate_interpolation(text_embeddings, image_embeddings, unet, vae, scheduler, 
                          device, num_interpolations=9, guidance_scale=2.5):
    """Generate interpolated images between source and target."""
    images = []
    
    for alpha in tqdm(np.linspace(0.0, 1.0, num_interpolations), desc="Generating variations"):
        # Interpolate embeddings
        interpolated_emb = alpha * text_embeddings + (1 - alpha) * image_embeddings
        
        # Generate image
        latents = sample_image(interpolated_emb, device, unet, scheduler, guidance_scale=guidance_scale)
        img = decode_image(latents, vae)
        img_tensor = transforms.ToTensor()(img)
        images.append(img_tensor)
    
    return images


class FastEdit:
    """FastEdit image editor class."""
    
    def __init__(self, device=None, seed=5679):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seed_everything(seed)
        
        print(f"Initializing FastEdit on {self.device}...")
        
        # Initialize accelerator
        self.accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="fp16")
        
        # Load models
        print("Loading Stable Diffusion pipeline...")
        self.sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
        ).to(self.device)
        
        self.vae = self.sd_pipe.vae
        self.scheduler = self.sd_pipe.scheduler
        self.image_encoder = self.sd_pipe.image_encoder
        self.feature_extractor = self.sd_pipe.feature_extractor
        self.unet = self.sd_pipe.unet
        
        # Freeze VAE and image encoder
        self.vae.requires_grad_(False).eval()
        self.image_encoder.requires_grad_(False).eval()
        
        # Load text encoder
        print("Loading CLIP text encoder...")
        self.text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        
        print("FastEdit initialized successfully!")
    
    def edit_image(self, image_path, prompt, output_dir=None, num_steps=50, 
                   guidance_scale=2.5, num_interpolations=9, learning_rate=4e-4):
        """
        Edit an image based on a text prompt.
        
        Args:
            image_path: Path to input image
            prompt: Text description of desired edit
            output_path: Path to save output (optional)
            num_steps: Number of fine-tuning steps
            guidance_scale: Guidance scale for generation
            num_interpolations: Number of interpolation steps
            learning_rate: Learning rate for fine-tuning
        
        Returns:
            PIL Image of the final edited result
        """
        start_time = time.time()
        
        # Load and encode image
        print(f"Loading image: {image_path}")
        img, cond = load_and_preprocess_image(image_path)
        img_tensor = img.to(self.device).unsqueeze(0)
        cond_tensor = cond.to(self.device).unsqueeze(0)
        
        # Encode to latents
        print("Encoding image to latents...")
        latent_dist = self.vae.encode(img_tensor).latent_dist
        image_latents = latent_dist.sample(generator=torch.Generator())
        image_latents *= self.vae.config.scaling_factor
        
        # Get image embeddings
        image_embeddings = encode_image(cond_tensor, self.device, self.image_encoder, self.feature_extractor)
        
        # Get text embeddings
        print(f"Processing prompt: '{prompt}'")
        text_embeddings = self.text_encoder.get_text_features(
            **self.text_processor(prompt, return_tensors="pt").to(self.device)
        ).unsqueeze(0)
        
        # Calculate semantic similarity
        cosine_sim = round(F.cosine_similarity(text_embeddings, image_embeddings, dim=-1).item(), 2)
        print(f"Semantic similarity score: {cosine_sim}")
        
        # Determine timesteps
        timestep_values, timestep_desc = calculate_timesteps(cosine_sim)
        print(f"Using {timestep_desc}")
        
        # Configure LoRA
        print("Configuring LoRA fine-tuning...")
        unet, optimizer = configure_lora_model(self.unet, self.device, learning_rate)
        unet, optimizer = self.accelerator.prepare(unet, optimizer)
        
        # Fine-tune model
        print(f"Fine-tuning for {num_steps} steps...")
        ft_start = time.time()
        fine_tune_model(
            unet, optimizer, self.scheduler, image_latents, image_embeddings,
            timestep_values, num_steps=num_steps, accelerator=self.accelerator
        )
        ft_time = time.time() - ft_start
        print(f"Fine-tuning completed in {ft_time:.2f}s")
        
        # Generate interpolated results
        print(f"Generating {num_interpolations} interpolated images...")
        unet.eval()
        images = generate_interpolation(
            text_embeddings, image_embeddings, unet, self.vae, 
            self.scheduler, self.device, num_interpolations, guidance_scale
        )
        
        # Create output grid
        images_tensor = torch.stack(images)
        result_grid = make_grid(images_tensor, nrow=min(5, num_interpolations), padding=2)
        
        # Save output
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Generate unique filename
        unique_id = uuid.uuid4().hex[:8]
        save_path = str(Path(output_dir) / f"edited_{unique_id}.png")
        
        save_image(result_grid, save_path)
        print(f"Results saved to: {save_path}")
        
        total_time = time.time() - start_time
        print(f"Total editing time: {total_time:.2f}s")
        
        # Return the final (most edited) image
        return images[-1]

def main(
    image: str,
    prompt: str,
    output_dir: str = None,
    steps: int = 50,
    guidance: float = 2.5,
    interpolations: int = 10,
    lr: float = 4e-4,
    seed: int = 5679,
    device: str = None,
):
    # Initialize editor
    dev = torch.device(device) if device else None
    editor = FastEdit(device=dev, seed=seed)

    # Edit image
    result = editor.edit_image(
        image_path=image,
        prompt=prompt,
        output_dir=output_dir,
        num_steps=steps,
        guidance_scale=guidance,
        num_interpolations=interpolations,
        learning_rate=lr
    )

    print("Editing complete!")
    return result