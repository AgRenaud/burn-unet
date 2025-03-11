import os
import argparse
import random
import shutil
import glob

from pathlib import Path


def create_train_val_split(data_dir, output_dir, val_ratio=0.2, seed=42):
    """
    Create train/validation split from a dataset directory containing images and masks.
    
    Args:
        data_dir (str): Path to the dataset directory containing 'images' and 'masks' folders
        output_dir (str): Output directory where train/val splits will be created
        val_ratio (float): Ratio of validation set size (default: 0.2)
        seed (int): Random seed for reproducibility (default: 42)
    """
    random.seed(seed)
    
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    masks_dir = data_path / "masks"
    
    if not images_dir.exists() or not images_dir.is_dir():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists() or not masks_dir.is_dir():
        raise ValueError(f"Masks directory not found: {masks_dir}")
    
    output_path = Path(output_dir)
    
    train_dir = output_path / "train"
    train_images_dir = train_dir / "images"
    train_masks_dir = train_dir / "masks"
    
    val_dir = output_path / "val"
    val_images_dir = val_dir / "images"
    val_masks_dir = val_dir / "masks"
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(str(images_dir / ext)))
        image_files.extend(glob.glob(str(images_dir / ext.upper())))
    
    image_files.sort()
    
    random.shuffle(image_files)
    
    num_val = int(len(image_files) * val_ratio)
    num_train = len(image_files) - num_val
    
    train_images = image_files[:num_train]
    val_images = image_files[num_train:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    for idx, img_path in enumerate(train_images):
        img_filename = os.path.basename(img_path)
        stem = Path(img_filename).stem
        
        shutil.copy2(img_path, train_images_dir / img_filename)
        
        mask_pattern = str(masks_dir / f"{stem}.*")
        mask_files = glob.glob(mask_pattern)
        
        if mask_files:
            mask_path = mask_files[0]
            mask_filename = os.path.basename(mask_path)
            shutil.copy2(mask_path, train_masks_dir / mask_filename)
        else:
            print(f"Warning: No mask found for training image {img_filename}")
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(train_images)} training images")
    
    for idx, img_path in enumerate(val_images):
        img_filename = os.path.basename(img_path)
        stem = Path(img_filename).stem
        
        shutil.copy2(img_path, val_images_dir / img_filename)
        
        mask_pattern = str(masks_dir / f"{stem}.*")
        mask_files = glob.glob(mask_pattern)
        
        if mask_files:
            mask_path = mask_files[0]
            mask_filename = os.path.basename(mask_path)
            shutil.copy2(mask_path, val_masks_dir / mask_filename)
        else:
            print(f"Warning: No mask found for validation image {img_filename}")
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(val_images)} validation images")
    
    print("\nDataset split completed successfully!")
    print(f"Train set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    print(f"Train directory: {train_dir}")
    print(f"Validation directory: {val_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/validation split for image segmentation dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory containing 'images' and 'masks' folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory where train/val splits will be created")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio of validation set size (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    create_train_val_split(args.data_dir, args.output_dir, args.val_ratio, args.seed)
