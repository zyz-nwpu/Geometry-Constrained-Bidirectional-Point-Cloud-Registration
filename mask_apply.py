import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
MASK_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def clean_path(value):
    return Path(value.strip().strip('"').strip("'")).resolve()


def output_directory(image_dir):
    return image_dir.parent / f"{image_dir.name}_apply_mask"


def image_files(image_dir):
    return sorted(path for path in image_dir.iterdir() if path.suffix in IMAGE_EXTENSIONS and path.is_file())


def find_mask(mask_dir, image_path):
    for extension in MASK_EXTENSIONS:
        candidate = mask_dir / f"{image_path.stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def apply_mask(image, mask):
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    normalized_mask = (binary_mask.astype(np.float32) / 255.0)[:, :, None]
    return (image.astype(np.float32) * normalized_mask).astype(np.uint8)


def process_folders(image_dir, mask_dir):
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    output_dir = output_directory(image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = image_files(image_dir)
    if not files:
        raise RuntimeError(f"No images found in {image_dir}")

    saved = 0
    for image_path in tqdm(files, desc="apply masks"):
        mask_path = find_mask(mask_dir, image_path)
        if mask_path is None:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue

        masked_image = apply_mask(image, mask)
        output_path = output_dir / f"{image_path.stem}.jpg"
        cv2.imwrite(str(output_path), masked_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        saved += 1

    print(f"Saved images: {saved}")
    print(f"Output: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Apply foreground masks to image sequences.")
    parser.add_argument("--images", type=Path)
    parser.add_argument("--masks", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = args.images.resolve() if args.images else clean_path(input("Input image directory: "))
    mask_dir = args.masks.resolve() if args.masks else clean_path(input("Mask directory: "))
    process_folders(image_dir, mask_dir)


if __name__ == "__main__":
    main()
