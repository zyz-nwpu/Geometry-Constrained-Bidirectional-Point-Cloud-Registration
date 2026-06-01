import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


CHECKPOINT = "third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
MASK_THRESHOLD = 0.0
DISPLAY_HEIGHT = 1000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}


Image.MAX_IMAGE_PIXELS = None


try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    build_sam2_video_predictor = None


selected_points = []
selected_labels = []


def clean_path(value):
    return Path(value.strip().strip('"').strip("'")).resolve()


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        selected_labels.append(1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        selected_points.append([x, y])
        selected_labels.append(0)


def scaled_image(image):
    height, width = image.shape[:2]
    if height <= DISPLAY_HEIGHT:
        return image.copy(), 1.0
    scale = DISPLAY_HEIGHT / height
    resized = cv2.resize(image, (int(width * scale), DISPLAY_HEIGHT))
    return resized, scale


def select_object_interactive(image_path):
    selected_points.clear()
    selected_labels.clear()

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    display_image, scale = scaled_image(image)
    window_name = "Select foreground points"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Left click: foreground; right click: background; space: confirm; c: clear; esc: exit")

    while True:
        canvas = display_image.copy()
        for point, label in zip(selected_points, selected_labels):
            color = (0, 0, 255) if label == 1 else (0, 255, 0)
            cv2.circle(canvas, (int(point[0]), int(point[1])), 5, color, -1)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1)
        if key == 32 and selected_points:
            break
        if key == 27:
            cv2.destroyAllWindows()
            sys.exit(0)
        if key == ord("c"):
            selected_points.clear()
            selected_labels.clear()

    cv2.destroyAllWindows()
    points = np.asarray(selected_points, dtype=np.float32) / scale
    labels = np.asarray(selected_labels, dtype=np.int32)
    return points, labels


def frame_files(input_dir):
    return sorted(path for path in input_dir.iterdir() if path.suffix in IMAGE_EXTENSIONS and path.is_file())


def save_mask(mask_logits, frame_name, output_dir):
    mask = (mask_logits[0] > MASK_THRESHOLD).cpu().numpy().squeeze()
    mask_uint8 = (mask * 255).astype(np.uint8)
    output_path = output_dir / f"{frame_name.stem}.jpg"
    Image.fromarray(mask_uint8, mode="L").save(output_path, quality=100, subsampling=0)


def process_directory(predictor, input_dir):
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {input_dir}")

    frames = frame_files(input_dir)
    if not frames:
        raise RuntimeError(f"No images found in {input_dir}")

    output_dir = input_dir.with_name(f"{input_dir.name}_mask")
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_state = predictor.init_state(video_path=str(input_dir))
    points, labels = select_object_interactive(frames[0])
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )

    for frame_index, object_ids, mask_logits in tqdm(
        predictor.propagate_in_video(inference_state),
        total=len(frames),
        desc="generate masks",
    ):
        save_mask(mask_logits, frames[frame_index], output_dir)

    predictor.reset_state(inference_state)
    print(f"Output: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate foreground masks with Segment Anything Model 2.")
    parser.add_argument("--images", type=Path)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--config", default=CONFIG)
    return parser.parse_args()


def main():
    args = parse_args()

    if build_sam2_video_predictor is None:
        raise ImportError("Segment Anything Model 2 is not available. Install third_party/sam2 first.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this mask generation script.")

    predictor = build_sam2_video_predictor(args.config, args.checkpoint, device="cuda")

    if args.images:
        process_directory(predictor, args.images.resolve())
        torch.cuda.empty_cache()
        return

    while True:
        value = input("Input image directory, or q to quit: ").strip()
        if value.lower() == "q":
            break
        if value:
            process_directory(predictor, clean_path(value))
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
