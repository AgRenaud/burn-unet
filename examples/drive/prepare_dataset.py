# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "opencv-python-headless",
#     "pillow",
#     "tqdm",
#     "rich",
# ]
# ///
import os
import shutil
import cv2
import argparse
import random
import numpy as np

from dataclasses import dataclass, field
from typing import List
from PIL import Image
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table


@dataclass
class DatasetStats:
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    augmentations_generated: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return self.train_count + self.val_count + self.test_count


class RetinalImageAugmenter:
    def __init__(
        self,
        rotation_range=30,
        flip_prob=0.5,
        shift_range=0.1,
        scale_range=0.15,
        brightness_range=0.2,
        contrast_range=0.2,
        perspective_prob=0.3,
        noise_prob=0.3,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.perspective_prob = perspective_prob
        self.noise_prob = noise_prob

    def _apply_rotation(self, image, groundtruth, fov_mask):
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        groundtruth = cv2.warpAffine(
            groundtruth, M, (w, h), borderMode=cv2.BORDER_CONSTANT
        )
        fov_mask = cv2.warpAffine(fov_mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        return image, groundtruth, fov_mask, f"rot{angle:.1f}"

    def _apply_flip(self, image, groundtruth, fov_mask):
        flip_type = ""
        if np.random.random() < self.flip_prob:
            image = cv2.flip(image, 1)
            groundtruth = cv2.flip(groundtruth, 1)
            fov_mask = cv2.flip(fov_mask, 1)
            flip_type += "h"
        if np.random.random() < self.flip_prob:
            image = cv2.flip(image, 0)
            groundtruth = cv2.flip(groundtruth, 0)
            fov_mask = cv2.flip(fov_mask, 0)
            flip_type += "v"
        suffix = f"flip{flip_type}" if flip_type else ""
        return image, groundtruth, fov_mask, suffix

    def _apply_shift_scale(self, image, groundtruth, fov_mask):
        h, w = image.shape[:2]
        tx = np.random.uniform(-self.shift_range, self.shift_range) * w
        ty = np.random.uniform(-self.shift_range, self.shift_range) * h
        scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
        M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        groundtruth = cv2.warpAffine(
            groundtruth, M, (w, h), borderMode=cv2.BORDER_CONSTANT
        )
        fov_mask = cv2.warpAffine(fov_mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        return image, groundtruth, fov_mask, f"shift{tx:.1f}_{ty:.1f}_scale{scale:.2f}"

    def _apply_perspective_transform(self, image, groundtruth, fov_mask):
        if np.random.random() > self.perspective_prob:
            return image, groundtruth, fov_mask, ""
        h, w = image.shape[:2]
        src_points = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        max_perturbation = min(h, w) * 0.05
        dst_points = np.float32(
            [
                [
                    np.random.uniform(0, max_perturbation),
                    np.random.uniform(0, max_perturbation),
                ],
                [
                    w - 1 - np.random.uniform(0, max_perturbation),
                    np.random.uniform(0, max_perturbation),
                ],
                [
                    np.random.uniform(0, max_perturbation),
                    h - 1 - np.random.uniform(0, max_perturbation),
                ],
                [
                    w - 1 - np.random.uniform(0, max_perturbation),
                    h - 1 - np.random.uniform(0, max_perturbation),
                ],
            ]
        )
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        image = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        groundtruth = cv2.warpPerspective(
            groundtruth, M, (w, h), borderMode=cv2.BORDER_CONSTANT
        )
        fov_mask = cv2.warpPerspective(
            fov_mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT
        )
        return image, groundtruth, fov_mask, "perspective"

    def _apply_brightness_contrast(self, image, fov_mask):
        modified_image = image.copy().astype(np.uint8)
        brightness_factor = np.random.uniform(
            1 - self.brightness_range * 0.5, 1 + self.brightness_range * 0.5
        )
        binary_mask = (fov_mask > 127).astype(np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 3:
            binary_mask = np.stack([binary_mask] * 3, axis=2)

        hsv_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HSV)
        hsv_img[:, :, 2] = np.where(
            binary_mask[:, :, 0] > 0,
            np.clip(hsv_img[:, :, 2] * brightness_factor, 0, 255),
            hsv_img[:, :, 2],
        )
        modified_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

        contrast_factor = np.random.uniform(
            1 - self.contrast_range * 0.7, 1 + self.contrast_range * 0.7
        )
        if contrast_factor != 1.0:
            for c in range(3):
                channel = modified_image[:, :, c]
                mean = np.sum(channel * binary_mask[:, :, 0]) / max(
                    np.sum(binary_mask[:, :, 0]), 1
                )
                channel_factor = contrast_factor * 0.7 if c == 0 else contrast_factor
                modified_image[:, :, c] = np.where(
                    binary_mask[:, :, 0] > 0,
                    np.clip(mean + channel_factor * (channel - mean), 0, 255),
                    channel,
                )
        return np.clip(modified_image, 0, 255).astype(
            np.uint8
        ), f"bright{brightness_factor:.2f}_cont{contrast_factor:.2f}"

    def _apply_gaussian_noise(self, image):
        if np.random.random() > self.noise_prob:
            return image, ""
        row, col, ch = image.shape
        var = np.random.uniform(3, 10)
        gauss = np.random.normal(0, var**0.5, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8), f"noise{var:.1f}"

    def augment(self, image, groundtruth, fov_mask):
        suffixes = []

        image, bright_suffix = self._apply_brightness_contrast(image, fov_mask)
        if bright_suffix:
            suffixes.append(bright_suffix)

        image, noise_suffix = self._apply_gaussian_noise(image)
        if noise_suffix:
            suffixes.append(noise_suffix)

        image, groundtruth, fov_mask, flip_suffix = self._apply_flip(
            image, groundtruth, fov_mask
        )
        if flip_suffix:
            suffixes.append(flip_suffix)

        image, groundtruth, fov_mask, rot_suffix = self._apply_rotation(
            image, groundtruth, fov_mask
        )
        if rot_suffix:
            suffixes.append(rot_suffix)

        image, groundtruth, fov_mask, shift_suffix = self._apply_shift_scale(
            image, groundtruth, fov_mask
        )
        if shift_suffix:
            suffixes.append(shift_suffix)

        image, groundtruth, fov_mask, perspective_suffix = (
            self._apply_perspective_transform(image, groundtruth, fov_mask)
        )
        if perspective_suffix:
            suffixes.append(perspective_suffix)

        _, groundtruth = cv2.threshold(groundtruth, 127, 255, cv2.THRESH_BINARY)
        _, fov_mask = cv2.threshold(fov_mask, 127, 255, cv2.THRESH_BINARY)

        full_suffix = "_aug_" + "_".join(suffixes) if suffixes else ""
        return image, groundtruth, fov_mask, full_suffix


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
        for folder in ["images", "groundtruth", "masks"]:
            os.makedirs(os.path.join(dst_path, split, folder), exist_ok=True)

    progress, tasks = setup_progress_bars(console)

    with progress:
        # Process test data
        test_images = sorted(os.listdir(os.path.join(src_path, "test/images")))
        progress.update(tasks["test"], total=len(test_images))

        for img_name in test_images:
            try:
                # Copy image
                src_img = os.path.join(src_path, "test/images", img_name)
                dst_img = os.path.join(dst_path, "test/images", img_name)
                shutil.copy(src_img, dst_img)

                # Copy mask if exists
                mask_name = img_name.replace("_test.tif", "_test_mask.gif")
                src_mask = os.path.join(src_path, "test/mask", mask_name)
                if os.path.exists(src_mask):
                    shutil.copy(
                        src_mask, os.path.join(dst_path, "test/masks", mask_name)
                    )

                stats.test_count += 1
            except Exception as e:
                stats.errors.append(f"Error processing test image {img_name}: {str(e)}")

            progress.update(tasks["test"], advance=1)

        # Get training images and split into train/val
        train_images = sorted(os.listdir(os.path.join(src_path, "training/images")))
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
            for img_name in img_list:
                try:
                    # Get paths
                    img_path = os.path.join(src_path, "training/images", img_name)
                    gt_name = img_name.replace("_training.tif", "_manual1.gif")
                    gt_path = os.path.join(src_path, "training/1st_manual", gt_name)
                    mask_name = img_name.replace("_training.tif", "_training_mask.gif")
                    mask_path = os.path.join(src_path, "training/mask", mask_name)

                    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (image_size, image_size))
                    groundtruth = np.array(Image.open(gt_path).convert("L"))
                    groundtruth = cv2.resize(groundtruth, (image_size, image_size))
                    fov_mask = np.array(Image.open(mask_path).convert("L"))
                    fov_mask = cv2.resize(fov_mask, (image_size, image_size))

                    std_img_name = img_name.replace(
                        "_training.tif", ".png"
                    )  # 21_training.tif => 21.png

                    color = cv2.COLOR_RGB2BGR
                    if grayscale:
                        color = cv2.COLOR_RGB2GRAY

                    cv2.imwrite(
                        os.path.join(dst_path, split, "images", std_img_name),
                        cv2.cvtColor(image, color),
                    )
                    cv2.imwrite(
                        os.path.join(dst_path, split, "groundtruth", std_img_name),
                        groundtruth,
                    )
                    cv2.imwrite(
                        os.path.join(dst_path, split, "masks", std_img_name),
                        fov_mask,
                    )
                    if split == "train":
                        stats.train_count += 1
                    else:
                        stats.val_count += 1

                    # Generate augmentations
                    if augmentations > 0:
                        augmenter = RetinalImageAugmenter(
                            seed=seed + hash(img_name) % 10000
                        )

                        for aug_idx in range(1, augmentations + 1):
                            try:
                                aug_image, aug_gt, aug_mask, suffix = augmenter.augment(
                                    image.copy(), groundtruth.copy(), fov_mask.copy()
                                )

                                aug_image = cv2.resize(
                                    aug_image, (image_size, image_size)
                                )
                                aug_gt = cv2.resize(aug_gt, (image_size, image_size))
                                aug_mask = cv2.resize(
                                    aug_mask, (image_size, image_size)
                                )

                                std_aug_img_name = std_img_name.replace(
                                    ".png", f"{suffix}_{aug_idx}.png"
                                )

                                cv2.imwrite(
                                    os.path.join(
                                        dst_path, split, "images", std_aug_img_name
                                    ),
                                    cv2.cvtColor(aug_image, color),
                                )
                                cv2.imwrite(
                                    os.path.join(
                                        dst_path, split, "groundtruth", std_aug_img_name
                                    ),
                                    aug_gt,
                                )
                                cv2.imwrite(
                                    os.path.join(
                                        dst_path, split, "masks", std_aug_img_name
                                    ),
                                    aug_mask,
                                )

                                stats.augmentations_generated += 1
                                progress.update(tasks["augment"], advance=1)
                            except Exception as e:
                                stats.errors.append(
                                    f"Error during augmentation {aug_idx} for {img_name}: {str(e)}"
                                )

                except Exception as e:
                    stats.errors.append(
                        f"Error processing {split} image {img_name}: {str(e)}"
                    )

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
    summary.add_row(
        "Total Images", str(stats.total_count + stats.augmentations_generated)
    )
    console.print(summary)

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
        description="Prepare DRIVE dataset for segmentation"
    )
    parser.add_argument(
        "--src", type=str, default="data/DRIVE", help="Source directory"
    )
    parser.add_argument(
        "--dst", type=str, default="data/DRIVE_prepared", help="Destination directory"
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
        action=argparse.BooleanOptionalAction,
        help="Image are converted to Grayscale",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    console = Console()
    console.rule("[bold blue]DRIVE Dataset Preparation")

    config = Table(title="Configuration")
    config.add_column("Parameter", style="cyan")
    config.add_column("Value", style="green")
    config.add_row("Source", args.src)
    config.add_row("Destination", args.dst)
    config.add_row("Validation Split", f"{args.val_split:.2f}")
    config.add_row("Augmentations", str(args.augmentations))
    config.add_row("Random Seed", str(args.seed))
    config.add_row("Image Size", f"[{args.image_size} ; {args.image_size}]")
    config.add_row("Is grayscale", str(args.grayscale))
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

    if os.path.exists(args.dst):
        console.print(
            Panel(
                "[bold orange1]Target directory already exists!",
                title="Warning",
                border_style="orange1",
            )
        )
        overrides_dir = Confirm.ask(
            "Would you like to continue (this may overrides some existing files) ?"
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
