# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "opencv-python-headless",
#     "pillow",
#     "tqdm",
#     "rich",
#     "pycocotools",
# ]
# ///
import os
import json
import shutil
import cv2
import argparse
import random
import numpy as np

from dataclasses import dataclass, field
from typing import List, Dict, Any
from PIL import Image
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from pycocotools import mask as coco_mask


@dataclass
class DatasetStats:
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    augmentations_generated: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return self.train_count + self.val_count + self.test_count


class TomatoImageAugmenter:
    def __init__(
        self,
        flip_prob=0.5,
        brightness_range=0.2,
        contrast_range=0.2,
        noise_prob=0.3,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_prob = noise_prob

    def _apply_flip(self, image, groundtruth):
        flip_type = ""
        if np.random.random() < self.flip_prob:
            image = cv2.flip(image, 1)
            groundtruth = cv2.flip(groundtruth, 1)
            flip_type += "h"
        if np.random.random() < self.flip_prob:
            image = cv2.flip(image, 0)
            groundtruth = cv2.flip(groundtruth, 0)
            flip_type += "v"
        suffix = f"flip{flip_type}" if flip_type else ""
        return image, groundtruth, suffix

    def _apply_brightness_contrast(self, image):
        modified_image = image.copy().astype(np.float32)
        
        # Apply brightness
        brightness_factor = np.random.uniform(
            1 - self.brightness_range, 1 + self.brightness_range
        )
        modified_image = modified_image * brightness_factor
        
        # Apply contrast
        contrast_factor = np.random.uniform(
            1 - self.contrast_range, 1 + self.contrast_range
        )
        mean = np.mean(modified_image)
        modified_image = mean + contrast_factor * (modified_image - mean)
        
        return np.clip(modified_image, 0, 255).astype(np.uint8), f"bright{brightness_factor:.2f}_cont{contrast_factor:.2f}"

    def _apply_gaussian_noise(self, image):
        if np.random.random() > self.noise_prob:
            return image, ""
        row, col, ch = image.shape
        var = np.random.uniform(3, 10)
        gauss = np.random.normal(0, var**0.5, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8), f"noise{var:.1f}"

    def augment(self, image, groundtruth):
        suffixes = []

        image, bright_suffix = self._apply_brightness_contrast(image)
        if bright_suffix:
            suffixes.append(bright_suffix)

        image, noise_suffix = self._apply_gaussian_noise(image)
        if noise_suffix:
            suffixes.append(noise_suffix)

        image, groundtruth, flip_suffix = self._apply_flip(image, groundtruth)
        if flip_suffix:
            suffixes.append(flip_suffix)

        full_suffix = "_aug_" + "_".join(suffixes) if suffixes else ""
        return image, groundtruth, full_suffix


def load_coco_annotations(json_path: str) -> Dict[str, Any]:
    """Load COCO format annotations"""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_mask_from_annotations(annotations: List[Dict], image_shape: tuple, category_mapping: Dict[int, int] = None) -> np.ndarray:
    """Create segmentation mask from COCO annotations"""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for ann in annotations:
        if 'segmentation' in ann and ann['segmentation']:
            # Handle polygon segmentation
            if isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                for seg in ann['segmentation']:
                    if len(seg) >= 6:  # At least 3 points (x,y pairs)
                        # Convert to numpy array and reshape
                        poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                        # Fill polygon with category value or 255 for binary mask
                        value = category_mapping.get(ann['category_id'], 255) if category_mapping else 255
                        cv2.fillPoly(mask, [poly], value)
    
    return mask


def setup_progress_bars(console):
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )

    tasks = {
        "test": progress.add_task("[cyan]Processing test data...", total=0),
        "train": progress.add_task("[green]Processing training data...", total=0),
        "val": progress.add_task("[yellow]Processing validation data...", total=0),
        "augment": progress.add_task("[magenta]Generating augmentations...", total=0),
    }

    return progress, tasks


def organize_dataset(
    src_path,
    dst_path,
    val_split=0.2,
    image_size=640,
    augmentations=0,
    seed=42,
    console=None,
    grayscale=False,
):
    stats = DatasetStats()
    random.seed(seed)
    np.random.seed(seed)

    # Create destination directories
    for split in ["train", "val", "test"]:
        for folder in ["images", "groundtruth"]:
            os.makedirs(os.path.join(dst_path, split, folder), exist_ok=True)

    progress, tasks = setup_progress_bars(console)

    with progress:
        # Load annotations
        train_annotations = load_coco_annotations(os.path.join(src_path, "annotations", "train.json"))
        test_annotations = load_coco_annotations(os.path.join(src_path, "annotations", "test.json"))
        
        # Build category mapping if multi-class
        category_mapping = None
        categories = train_annotations.get('categories', [])
        category_mapping = {cat['id']: idx + 1 for idx, cat in enumerate(categories)}
        stats.categories = {cat['name']: cat['id'] for cat in categories}

        # Create image ID to annotations mapping
        def create_ann_mapping(annotations):
            ann_map = {}
            for ann in annotations['annotations']:
                img_id = ann['image_id']
                if img_id not in ann_map:
                    ann_map[img_id] = []
                ann_map[img_id].append(ann)
            return ann_map

        train_ann_map = create_ann_mapping(train_annotations)
        test_ann_map = create_ann_mapping(test_annotations)

        # Process test data
        test_images = test_annotations['images']
        progress.update(tasks["test"], total=len(test_images))

        for img_info in test_images:
            try:
                img_name = img_info['file_name']
                img_id = img_info['id']
                
                # Load and resize image
                src_img_path = os.path.join(src_path, "test", img_name)
                if not os.path.exists(src_img_path):
                    stats.errors.append(f"Test image not found: {img_name}")
                    continue
                
                image = cv2.imread(src_img_path)
                if image is None:
                    stats.errors.append(f"Failed to load test image: {img_name}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                original_shape = image.shape
                image = cv2.resize(image, (image_size, image_size))
                
                # Create mask from annotations
                annotations = test_ann_map.get(img_id, [])
                mask = create_mask_from_annotations(annotations, original_shape, category_mapping)
                mask = cv2.resize(mask, (image_size, image_size))
                
                # Save files
                base_name = os.path.splitext(img_name)[0] + ".png"
                
                color_mode = cv2.COLOR_RGB2GRAY if grayscale else cv2.COLOR_RGB2BGR
                cv2.imwrite(
                    os.path.join(dst_path, "test", "images", base_name),
                    cv2.cvtColor(image, color_mode)
                )
                cv2.imwrite(
                    os.path.join(dst_path, "test", "groundtruth", base_name),
                    mask
                )
                
                stats.test_count += 1
                
            except Exception as e:
                stats.errors.append(f"Error processing test image {img_name}: {str(e)}")
            
            progress.update(tasks["test"], advance=1)

        # Split training data into train/val
        train_images = train_annotations['images']
        random.shuffle(train_images)
        split_idx = int(len(train_images) * (1 - val_split))
        train_imgs = train_images[:split_idx]
        val_imgs = train_images[split_idx:]

        # Process training and validation sets
        total_augmentations = (len(train_imgs) + len(val_imgs)) * augmentations
        progress.update(tasks["train"], total=len(train_imgs))
        progress.update(tasks["val"], total=len(val_imgs))
        progress.update(
            tasks["augment"],
            total=total_augmentations if total_augmentations > 0 else 1,
        )

        for split, img_list, task_id in [
            ("train", train_imgs, tasks["train"]),
            ("val", val_imgs, tasks["val"]),
        ]:
            for img_info in img_list:
                try:
                    img_name = img_info['file_name']
                    img_id = img_info['id']
                    
                    # Load and resize image
                    src_img_path = os.path.join(src_path, "train", img_name)
                    if not os.path.exists(src_img_path):
                        stats.errors.append(f"Train image not found: {img_name}")
                        continue
                    
                    image = cv2.imread(src_img_path)
                    if image is None:
                        stats.errors.append(f"Failed to load train image: {img_name}")
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    original_shape = image.shape
                    image = cv2.resize(image, (image_size, image_size))
                    
                    # Create mask from annotations
                    annotations = train_ann_map.get(img_id, [])
                    mask = create_mask_from_annotations(annotations, original_shape, category_mapping)
                    mask = cv2.resize(mask, (image_size, image_size))
                    
                    # Save original files
                    base_name = os.path.splitext(img_name)[0] + ".png"
                    
                    color_mode = cv2.COLOR_RGB2GRAY if grayscale else cv2.COLOR_RGB2BGR
                    cv2.imwrite(
                        os.path.join(dst_path, split, "images", base_name),
                        cv2.cvtColor(image, color_mode)
                    )
                    cv2.imwrite(
                        os.path.join(dst_path, split, "groundtruth", base_name),
                        mask
                    )
                    
                    if split == "train":
                        stats.train_count += 1
                    else:
                        stats.val_count += 1

                    # Generate augmentations
                    if augmentations > 0:
                        augmenter = TomatoImageAugmenter(seed=seed + hash(img_name) % 10000)
                        
                        for aug_idx in range(1, augmentations + 1):
                            try:
                                aug_image, aug_mask, suffix = augmenter.augment(
                                    image.copy(), mask.copy()
                                )
                                
                                aug_image = cv2.resize(aug_image, (image_size, image_size))
                                aug_mask = cv2.resize(aug_mask, (image_size, image_size))
                                
                                aug_name = base_name.replace(".png", f"{suffix}_{aug_idx}.png")
                                
                                cv2.imwrite(
                                    os.path.join(dst_path, split, "images", aug_name),
                                    cv2.cvtColor(aug_image, color_mode)
                                )
                                cv2.imwrite(
                                    os.path.join(dst_path, split, "groundtruth", aug_name),
                                    aug_mask
                                )
                                
                                stats.augmentations_generated += 1
                                progress.update(tasks["augment"], advance=1)
                                
                            except Exception as e:
                                stats.errors.append(
                                    f"Error during augmentation {aug_idx} for {img_name}: {str(e)}"
                                )

                except Exception as e:
                    stats.errors.append(f"Error processing {split} image {img_name}: {str(e)}")

                progress.update(task_id, advance=1)

            # Skip augmentation progress if no augmentations
            if augmentations == 0 and task_id == tasks["val"]:
                progress.update(tasks["augment"], advance=1)

    return stats


def display_summary(console, stats, start_dir, dest_dir):
    console.print("\n")
    console.rule("[bold green]Dataset Reorganization Complete!")

    summary = Table(title="Dataset Summary")
    summary.add_column("Split", style="cyan")
    summary.add_column("Image Count", style="green")
    summary.add_row("Train", str(stats.train_count))
    summary.add_row("Validation", str(stats.val_count))
    summary.add_row("Test", str(stats.test_count))
    summary.add_row("Total Original", str(stats.total_count))
    summary.add_row("Augmentations", str(stats.augmentations_generated))
    summary.add_row("Total Images", str(stats.total_count + stats.augmentations_generated))
    console.print(summary)

    if stats.categories:
        cat_table = Table(title="Categories Found")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("ID", style="green")
        for name, cat_id in stats.categories.items():
            cat_table.add_row(name, str(cat_id))
        console.print(cat_table)

    if stats.errors:
        error_panel = Panel(
            "\n".join(
                stats.errors[:5]
                + (["...and more errors"] if len(stats.errors) > 5 else [])
            ),
            title=f"[bold red]Errors ({len(stats.errors)} total)",
            border_style="red",
        )
        console.print(error_panel)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare tomato segmentation dataset from COCO format"
    )
    parser.add_argument(
        "--src", type=str, default="data/laboro_big", help="Source directory"
    )
    parser.add_argument(
        "--dst", type=str, default="data/tomato_prepared", help="Destination directory"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split (0.0-1.0)"
    )
    parser.add_argument(
        "--augmentations", type=int, default=5, help="Augmentations per image"
    )
    parser.add_argument("--image-size", type=int, default=640, help="Image resize")
    parser.add_argument(
        "--grayscale",
        action='store_true',
        help="Convert images to grayscale"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    console = Console()
    console.rule("[bold blue]Tomato Dataset Preparation")

    config = Table(title="Configuration")
    config.add_column("Parameter", style="cyan")
    config.add_column("Value", style="green")
    config.add_row("Source", args.src)
    config.add_row("Destination", args.dst)
    config.add_row("Validation Split", f"{args.val_split:.2f}")
    config.add_row("Augmentations", str(args.augmentations))
    config.add_row("Random Seed", str(args.seed))
    config.add_row("Image Size", f"[{args.image_size} x {args.image_size}]")
    config.add_row("Grayscale", str(args.grayscale))
    console.print(config)

    if not os.path.exists(args.src):
        console.print(
            Panel(
                "[bold red]Source directory does not exist!",
                title="Error",
                border_style="red",
            )
        )
        return 1

    # Check for required files
    required_files = [
        os.path.join(args.src, "annotations", "train.json"),
        os.path.join(args.src, "annotations", "test.json"),
        os.path.join(args.src, "train"),
        os.path.join(args.src, "test"),
    ]
    
    for req_file in required_files:
        if not os.path.exists(req_file):
            console.print(
                Panel(
                    f"[bold red]Required file/directory not found: {req_file}",
                    title="Error",
                    border_style="red",
                )
            )
            return 1

    if os.path.exists(args.dst):
        console.print(
            Panel(
                "[bold orange1]Target directory already exists!",
                title="Warning",
                border_style="orange1",
            )
        )
        overrides_dir = Confirm.ask(
            "Would you like to continue (this may override some existing files)?"
        )
        if not overrides_dir:
            return 1

    try:
        console.print("\n[bold yellow]Starting dataset preparation...")
        stats = organize_dataset(
            args.src,
            args.dst,
            val_split=args.val_split,
            image_size=args.image_size,
            augmentations=args.augmentations,
            seed=args.seed,
            console=console,
            grayscale=args.grayscale,
        )
        display_summary(console, stats, args.src, args.dst)
        return 0
        
    except Exception as e:
        console.print_exception()
        console.print(
            Panel(
                f"[bold red]Unhandled error: {str(e)}",
                title="Error",
                border_style="red",
            )
        )
        return 1


if __name__ == "__main__":
    exit(main())
