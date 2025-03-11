import os
import argparse
import glob

from PIL import Image


def resize_images(input_dir, output_dir, size=(256, 256)):
    """
    Resize all images in input_dir and save them to output_dir with the same filename
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    print(f"Found {len(image_files)} images in {input_dir}")

    for i, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path)

            img_resized = img.resize(size, Image.LANCZOS)

            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, filename)

            img_resized.save(output_path)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Resized {len(image_files)} images to {size}")


def main():
    parser = argparse.ArgumentParser(description="Resize images for U-Net segmentation training")
    parser.add_argument("--images_dir", type=str, required=True, 
                        help="Directory containing original images")
    parser.add_argument("--masks_dir", type=str, required=True, 
                        help="Directory containing original masks")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for resized dataset")
    parser.add_argument("--width", type=int, default=256, 
                        help="Target width for resized images (default: 256)")
    parser.add_argument("--height", type=int, default=256, 
                        help="Target height for resized images (default: 256)")
    
    args = parser.parse_args()
    
    output_images_dir = os.path.join(args.output_dir, "images")
    output_masks_dir = os.path.join(args.output_dir, "masks")
    
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory {args.images_dir} does not exist")
        return

    if not os.path.exists(args.masks_dir):
        print(f"Error: Masks directory {args.masks_dir} does not exist")
        return

    target_size = (args.width, args.height)
    
    print(f"Resizing images to {target_size}...")
    resize_images(args.images_dir, output_images_dir, target_size)

    print("Resizing masks...")
    resize_images(args.masks_dir, output_masks_dir, target_size)

    print("\nResizing completed!")
    print("Dataset structure created at:")
    print(f"  {args.output_dir}/")
    print(f"  ├── images/ ({len(os.listdir(output_images_dir))} files)")
    print(f"  └── masks/ ({len(os.listdir(output_masks_dir))} files)")
    

if __name__ == "__main__":
    main()
