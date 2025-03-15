# DRIVE: Digital Retinal Images for Vessel Extraction

This guide explains how to use the U-Net implementation for retinal vessel segmentation using the DRIVE (Digital Retinal Images for Vessel Extraction) dataset.

## Getting Started

### 1. Download and Prepare the Dataset

First, download the DRIVE dataset:

```bash
# Download dataset
./download_dataset.sh
```

Then prepare the dataset into the format expected by our U-Net implementation:

```bash
# Prepare dataset with default options (80% training, 20% validation)
# You'll need https://github.com/astral-sh/uv to run the script.

uv run prepare_dataset.py --src data/DRIVE --dst data/DRIVE_AUGMENTED --val-split 0.2 --image-size 64 --augmentations 100 --seed 42
```

### 2. Training the U-Net Model

Run the training with the prepared dataset:

> Make sure params `--image-size` and `--grayscale` match the prepared dataset

```bash
# Run with wgpu backend
cargo run -F wgpu --release -- --data-dir data/DRIVE_AUGMENTED --base-channels 32 --epochs 100 --save-checkpoints --image-size 64 --batch-size 32

# Run with cuda backend
cargo run -F cuda --release -- --data-dir data/DRIVE_AUGMENTED --base-channels 32 --epochs 100 --save-checkpoints --image-size 64 --batch-size 32

# Without specified backend, will use NdArray (CPU)
cargo run --release -- --data-dir data/DRIVE_AUGMENTED --base-channels 32 --epochs 100 --save-checkpoints --image-size 64 --batch-size 32
```

### Configuration Options

- `--data-dir`: Path to the prepared dataset
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate
- `--num-workers`: Number of data loading worker threads
- `--base-channels`: Number of base channels in the U-Net (default: 64)
- `--image-size`: Images size (default: 640)
- `--seed`: Random seed for reproducibility
- `--artifact-dir`: Directory to save model artifacts
- `--save-checkpoints`: Whether to save model checkpoints (default: true)

## Dataset Structure

The prepared dataset has the following structure:

```
data/DRIVE_prepared/
├── train/
│   ├── images/       # Training images
│   ├── groundtruth/  # Vessel segmentation ground truth
│   └── masks/        # Field of view masks (optional)
├── val/
│   ├── images/       # Validation images
│   ├── groundtruth/  # Vessel segmentation ground truth
│   └── masks/        # Field of view masks (optional)
└── test/
    ├── images/       # Test images
    ├── groundtruth/  # Vessel segmentation ground truth (if available)
    └── masks/        # Field of view masks (optional)
```
