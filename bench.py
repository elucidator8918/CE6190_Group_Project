"""
Image Editing Benchmark Suite

This module provides a unified benchmarking framework for evaluating image editing methods
using LPIPS (perceptual similarity) and CLIP score metrics.

Supported Methods:
    - leditspp: LEDITS++ image editing
    - fastedit: FastEdit image editing
    - ledits: LEDITS image editing

Supported Datasets:
    - ourbench: Custom benchmark dataset
    - tedbench: TED benchmark dataset

Usage:
    python benchmark.py --method leditspp --dataset ourbench
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
from pathlib import Path

import torch
import lpips
import numpy as np
import pandas as pd
from PIL import Image
from torchmetrics.multimodal import CLIPScore
import warnings
warnings.filterwarnings("ignore")

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "benchmark.log"

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = sys.stderr = Tee(sys.stdout, open(log_file, "w"))

class ImageEditingBenchmark:
    """
    A benchmark suite for evaluating image editing methods.

    Attributes:
        method (str): The editing method to use.
        dataset (str): The dataset to benchmark on.
        device (torch.device): Device for computation (CPU/CUDA).
        lpips_fn: LPIPS metric function.
        clip_metric: CLIP score metric.
    """

    SUPPORTED_METHODS = ["leditspp", "fastedit", "ledits"]
    SUPPORTED_DATASETS = ["ourbench", "tedbench"]
    CLIP_INPUT_SIZE = (224, 224)

    def __init__(self, method: str, dataset: str, device: str = None):
        """
        Initialize the benchmark suite.

        Args:
            method: Editing method to use ('leditspp', 'fastedit', or 'ledits').
            dataset: Dataset to benchmark on ('ourbench' or 'tedbench').
            device: Device for computation. If None, automatically selects CUDA if available.

        Raises:
            ValueError: If method or dataset is not supported.
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Method '{method}' not supported. Choose from {self.SUPPORTED_METHODS}"
            )
        if dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset '{dataset}' not supported. Choose from {self.SUPPORTED_DATASETS}"
            )

        self.method = method
        self.dataset = dataset
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load dataset configuration
        self.data_dir = Path("assets") / dataset
        self.json_path = self.data_dir / (
            "dataset.json" if dataset == "ourbench" else "FastEdit.json"
        )

        if not self.json_path.exists():
            raise FileNotFoundError(
                f"Dataset configuration not found at {self.json_path}"
            )

        with open(self.json_path, "r") as f:
            self.dataset_items = json.load(f)

        # Initialize metrics
        self._initialize_metrics()

        # Import editing function
        self._import_editing_method()

    def _initialize_metrics(self) -> None:
        """Initialize LPIPS and CLIP score metrics."""
        print(f"Initializing metrics on {self.device}...")
        self.lpips_fn = lpips.LPIPS(net="vgg").to(self.device)
        self.clip_metric = CLIPScore(
            model_name_or_path="openai/clip-vit-large-patch14"
        ).to(self.device)

    def _import_editing_method(self) -> None:
        """Dynamically import the appropriate editing method."""
        if self.method == "leditspp":
            from leditspp import edit_image_simple

            self.edit_fn = edit_image_simple
        elif self.method == "fastedit":
            from fastedit import main as fastedit_main

            self.edit_fn = fastedit_main
        elif self.method == "ledits":
            from ledits import edit_image

            self.edit_fn = edit_image

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """
        Convert PIL image to normalized tensor.

        Args:
            img: PIL Image to convert.

        Returns:
            Normalized tensor of shape [1, C, H, W] in range [0, 1].
        """
        img = img.resize(self.CLIP_INPUT_SIZE, Image.BICUBIC)
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _edit_image(self, image_path: str, prompt: str) -> Image.Image:
        """
        Apply the selected editing method to an image.

        Args:
            image_path: Path to the input image.
            prompt: Text prompt for editing.

        Returns:
            Edited PIL Image.
        """
        if self.method == "leditspp":
            edited, _ = self.edit_fn(
                image_path=image_path, editing_prompt=prompt, edit_guidance_scale=10.0, output_dir=f"outputs/leditspp_{self.dataset}"
            )
            edited = Image.fromarray(np.array(edited).clip(0, 255).astype(np.uint8))

        elif self.method == "fastedit":
            edited = self.edit_fn(image=image_path, prompt=prompt, output_dir=f"outputs/fastedit_{self.dataset}")
            edited = Image.fromarray(
                (np.transpose(np.array(edited), (1, 2, 0)) * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )

        elif self.method == "ledits":
            edited, _ = self.edit_fn(
                seed=42,
                output_dir=f"outputs/ledits_{self.dataset}",
                input_image=image_path,
                target_prompt=prompt,
                target_guidance=10.0,
            )
            edited = Image.fromarray(np.array(edited).clip(0, 255).astype(np.uint8))

        return edited

    def _compute_metrics(
        self, original: torch.Tensor, edited: torch.Tensor, prompt: str
    ) -> Tuple[float, float]:
        """
        Compute LPIPS and CLIP score metrics.

        Args:
            original: Original image tensor.
            edited: Edited image tensor.
            prompt: Text prompt used for editing.

        Returns:
            Tuple of (lpips_score, clip_score).
        """
        # LPIPS: perceptual similarity (lower is more similar)
        lpips_score = self.lpips_fn(original, edited).item()

        # CLIP Score: text-image alignment (higher is better)
        self.clip_metric.reset()
        self.clip_metric.update(edited, prompt)
        clip_score = self.clip_metric.compute().item()

        return lpips_score, clip_score

    def run(self) -> pd.DataFrame:
        """
        Run the benchmark on all dataset items.

        Returns:
            DataFrame containing results with columns: image, prompt, lpips, clip_score.
        """
        print(f"\n{'='*60}")
        print(f"Running Benchmark")
        print(f"Method: {self.method}")
        print(f"Dataset: {self.dataset}")
        print(f"Device: {self.device}")
        print(f"Number of images: {len(self.dataset_items)}")
        print(f"{'='*60}\n")

        results = []

        for idx, item in enumerate(self.dataset_items, 1):
            img_name = item["img_name"]
            prompt = item["target_text"]
            image_path = str(self.data_dir / "images" / img_name)

            if not Path(image_path).exists():
                print(f"Warning: Image not found at {image_path}, skipping...")
                continue

            print(f"[{idx}/{len(self.dataset_items)}] Editing {img_name}...")

            # Apply editing
            edited_pil = self._edit_image(image_path, prompt)

            # Load original image
            original_pil = Image.open(image_path).convert("RGB")

            # Convert to tensors
            original_tensor = self._pil_to_tensor(original_pil)
            edited_tensor = self._pil_to_tensor(edited_pil)

            # Compute metrics
            lpips_score, clip_score = self._compute_metrics(
                original_tensor, edited_tensor, prompt
            )

            # Store results
            results.append(
                {
                    "image": img_name,
                    "prompt": prompt,
                    "lpips": lpips_score,
                    "clip_score": clip_score,
                }
            )

            print(f"  LPIPS: {lpips_score:.4f} | CLIP Score: {clip_score:.4f}")

        return pd.DataFrame(results)

    def save_results(self, df: pd.DataFrame, output_dir: str = ".") -> str:
        """
        Save benchmark results to CSV.

        Args:
            df: DataFrame containing results.
            output_dir: Directory to save results.

        Returns:
            Path to saved CSV file.
        """
        output_dir = Path(output_dir) / "benchmark_results"
        output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

        output_path = output_dir / f"{self.method}_clip_lpips_{self.dataset}.csv"
        df.to_csv(output_path, index=False)
        return str(output_path)

    def print_summary(self, df: pd.DataFrame) -> None:
        """
        Print summary statistics of benchmark results.

        Args:
            df: DataFrame containing results.
        """
        print(f"\n{'='*60}")
        print("Summary Statistics")
        print(f"{'='*60}")
        print(f"Number of images processed: {len(df)}")
        print(f"\nLPIPS (perceptual similarity, lower is better):")
        print(f"  Mean:   {df['lpips'].mean():.4f}")
        print(f"  Median: {df['lpips'].median():.4f}")
        print(f"  Std:    {df['lpips'].std():.4f}")
        print(f"\nCLIP Score (text-image alignment, higher is better):")
        print(f"  Mean:   {df['clip_score'].mean():.4f}")
        print(f"  Median: {df['clip_score'].median():.4f}")
        print(f"  Std:    {df['clip_score'].std():.4f}")
        print(f"{'='*60}\n")


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Image Editing Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                python benchmark.py --method leditspp --dataset ourbench
                python benchmark.py --method fastedit --dataset tedbench --device cuda:0
                python benchmark.py        # Runs all methods on all datasets
                    """,
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=ImageEditingBenchmark.SUPPORTED_METHODS,
        help="Editing method to benchmark",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=ImageEditingBenchmark.SUPPORTED_DATASETS,
        help="Dataset to use for benchmarking",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for computation (e.g., 'cuda', 'cuda:0', 'cpu'). Auto-detects if not specified.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save results (default: current directory)",
    )

    args = parser.parse_args()

    # Determine combinations to run
    methods_to_run = (
        [args.method] if args.method else ImageEditingBenchmark.SUPPORTED_METHODS
    )
    datasets_to_run = (
        [args.dataset] if args.dataset else ImageEditingBenchmark.SUPPORTED_DATASETS
    )

    for method in methods_to_run:
        for dataset in datasets_to_run:
            print(f"\n{'#'*60}")
            print(f"Running benchmark: Method={method}, Dataset={dataset}")
            print(f"{'#'*60}\n")

            benchmark = ImageEditingBenchmark(
                method=method, dataset=dataset, device=args.device
            )

            results_df = benchmark.run()

            # Save results
            output_path = benchmark.save_results(results_df, args.output_dir)
            print(f"Results saved to: {output_path}")

            # Print summary
            benchmark.print_summary(results_df)


if __name__ == "__main__":
    main()
