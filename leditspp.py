import torch
from PIL import Image
from diffusers import LEditsPPPipelineStableDiffusion
from typing import Union, List, Optional
import uuid
from pathlib import Path


class LEDitsImageEditor:
    """
    A simple interface for editing images using the LEDits++ pipeline.
    """
    
    def __init__(self, model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
        """
        Initialize the LEDits++ pipeline.
        
        Args:
            model_id: HuggingFace model ID to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Loading model on {self.device.upper()}...")
        self.pipeline = LEditsPPPipelineStableDiffusion.from_pretrained(
            model_id,
            torch_dtype=self.dtype
        )
        self.pipeline = self.pipeline.to(self.device)
        #self.pipeline.enable_vae_tiling() # works in latest version of diffusers
        print("Model loaded successfully!")
        
        self.inverted = False
    
    def invert_image(
        self,
        image: Union[str, Image.Image],
        source_prompt: str = "",
        source_guidance_scale: float = 3.5,
        num_inversion_steps: int = 50,
        skip: float = 0.15,
        target_size: tuple = (512, 512)
    ):
        """
        Invert the input image to prepare for editing.
        
        Args:
            image: Path to image file or PIL Image object
            source_prompt: Optional prompt describing the input image
            source_guidance_scale: Strength of guidance during inversion
            num_inversion_steps: Number of inversion steps
            skip: Portion of initial steps to skip (lower = stronger changes)
            target_size: Size to resize image to (width, height)
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Resize image
        image_resized = image.resize(target_size)
        
        print("Inverting image...")
        _ = self.pipeline.invert(
            image=image_resized,
            source_prompt=source_prompt,
            source_guidance_scale=source_guidance_scale,
            num_inversion_steps=num_inversion_steps,
            skip=skip
        )
        
        self.inverted = True
        print("Image inverted successfully!")
    
    def edit_image(
        self,
        editing_prompt: Union[str, List[str]],
        edit_guidance_scale: float = 10.0,
        edit_threshold: float = 0.75,
        edit_warmup_steps: int = 0,
        reverse_editing_direction: bool = False,
        use_cross_attn_mask: bool = False,
        use_intersect_mask: bool = True,
        guidance_rescale: float = 0.0
    ) -> Image.Image:
        """
        Edit the inverted image using text prompts.
        
        Args:
            editing_prompt: Single prompt or list of prompts for editing
            edit_guidance_scale: Strength of editing guidance
            edit_threshold: Masking threshold (lower = more editing)
            edit_warmup_steps: Number of initial steps without editing
            reverse_editing_direction: If True, decrease instead of increase
            use_cross_attn_mask: Use cross-attention masking
            use_intersect_mask: Use intersection masking
            guidance_rescale: Rescale factor for guidance
            
        Returns:
            PIL Image object of the edited image
        """
        if not self.inverted:
            raise ValueError("Image must be inverted first! Call invert_image() before editing.")
        
        # Convert single prompt to list
        if isinstance(editing_prompt, str):
            editing_prompt = [editing_prompt]
        
        print("Generating edited image...")
        output = self.pipeline(
            editing_prompt=editing_prompt,
            edit_guidance_scale=edit_guidance_scale,
            edit_threshold=edit_threshold,
            edit_warmup_steps=edit_warmup_steps,
            reverse_editing_direction=reverse_editing_direction,
            use_cross_attn_mask=use_cross_attn_mask,
            use_intersect_mask=use_intersect_mask,
            guidance_rescale=guidance_rescale,
        )
        
        print("Image generated successfully!")
        return output.images[0]


def edit_image_simple(
    image_path: str,
    editing_prompt: Union[str, List[str]],
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    source_prompt: str = "",
    edit_guidance_scale: float = 10.0,
    edit_threshold: float = 0.75,
    output_path: Optional[str] = None,
    auto_save: bool = True,
    output_dir: str = "outputs/leditsppted"
) -> tuple[Image.Image, str]:
    """
    Simple one-function interface to edit an image.
    
    Args:
        image_path: Path to input image
        editing_prompt: Text prompt(s) describing desired edits
        model_id: HuggingFace model ID
        source_prompt: Optional description of input image
        edit_guidance_scale: Strength of editing (higher = stronger)
        edit_threshold: Masking threshold (lower = more editing)
        output_path: Optional specific path to save output image
        auto_save: If True, automatically save with UUID filename
        output_dir: Directory to save outputs when auto_save is True
        
    Returns:
        Tuple of (PIL Image object, saved file path)
        
    Example:
        >>> edited_img, path = edit_image_simple(
        ...     "input.jpg",
        ...     "cherry blossom",
        ...     edit_guidance_scale=12.0
        ... )
        >>> print(f"Saved to: {path}")
    """
    editor = LEDitsImageEditor(model_id=model_id)
    editor.invert_image(image_path, source_prompt=source_prompt)
    edited_image = editor.edit_image(
        editing_prompt=editing_prompt,
        edit_guidance_scale=edit_guidance_scale,
        edit_threshold=edit_threshold
    )
    
    # Determine save path
    if output_path:
        save_path = output_path
    elif auto_save:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Generate unique filename
        unique_id = uuid.uuid4().hex[:8]
        save_path = str(Path(output_dir) / f"edited_{unique_id}.png")
    else:
        save_path = None
    
    # Save if path is set
    if save_path:
        edited_image.save(save_path)
        print(f"Saved to {save_path}")
    
    return edited_image, save_path