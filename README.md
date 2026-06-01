# Geometry-Constrained Bidirectional Point Cloud Registration

<p align="center">
  <strong>Geometry-Constrained Bidirectional Point Cloud Registration for Thin Cultural Heritage Artifacts</strong>
</p>

<p align="center">
  <a href="https://zyz-nwpu.github.io/Geometry-Constrained-Bidirectional-Registration">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-1a7f64?style=for-the-badge">
  </a>
  <a href="https://github.com/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration">
    <img alt="Source Code" src="https://img.shields.io/badge/Source-Code-24292f?style=for-the-badge&logo=github">
  </a>
</p>


## Environment Setup

This project uses a Conda environment named `gcbreg`.

### 1. Create the Conda environment

```bash
conda create -n gcbreg python=3.10 -y
conda activate gcbreg
```

### 2. Install PyTorch with GPU support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Initialize third-party submodules

```bash
git submodule update --init --recursive
```

### 4. Install SAM 2

```bash
cd third_party/sam2
pip install -e .
cd ../..
```

### 5. Prepare the SAM 2 checkpoint

Create the checkpoint folder:

```text
third_party/sam2/checkpoints/
```

Download the checkpoint:

https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

Place it at:

```text
third_party/sam2/checkpoints/sam2.1_hiera_large.pt
```

### 6. Configure COLMAP

COLMAP is used as the Structure-from-Motion and Multi-View Stereo reconstruction tool.

For Windows users, the official CUDA-enabled prebuilt package is available here:

https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-cuda.zip

After extracting COLMAP, add the directory containing `colmap.exe` to your system `PATH`.


## Processing Pipeline

### 7. Prepare the artifact dataset

For each artifact, organize the front-side and back-side data as follows:

```text
Object_01/
├── front/
│   └── input/
├── back/
│   └── input/
├── sparse_front/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── sparse_back/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── point_front/
└── point_back/
```

The `front/input/` and `back/input/` folders contain the original images of the front and back sides, respectively. The `sparse_front/0/` and `sparse_back/0/` folders contain the corresponding COLMAP sparse reconstruction results.

### 8. Generate foreground masks

Run `mask_get.py` separately for the front and back input folders:

```bash
python mask_get.py
```

Input examples:

```text
Object_01/front/input/
Object_01/back/input/
```

The script automatically generates:

```text
Object_01/front/input_mask/
Object_01/back/input_mask/
```

The generated mask files keep the same filename stems as the original input images.

### 9. Apply masks to input images

Run `mask_apply.py` to remove image backgrounds using the generated masks:

```bash
python mask_apply.py
```

For the front side, use:

```text
Input folder: Object_01/front/input/
Mask folder:  Object_01/front/input_mask/
```

For the back side, use:

```text
Input folder: Object_01/back/input/
Mask folder:  Object_01/back/input_mask/
```

The script automatically generates:

```text
Object_01/front/input_apply_mask/
Object_01/back/input_apply_mask/
```

The output images keep the original filenames, with background regions set to black.

### 10. Clean sparse point clouds

Run `clear_point.py` separately for the front-side and back-side point clouds. This step projects 3D points into the image views and removes points that are inconsistent with the foreground masks.

Front side:

```bash
python clear_point.py \
  --input_ply Object_01/raw_front.ply \
  --colmap_dir Object_01/sparse_front/0 \
  --mask_dir Object_01/front/input_mask \
  --output_ply Object_01/point_front/points3D_clean_front.ply \
  --threshold 0.5
```

Back side:

```bash
python clear_point.py \
  --input_ply Object_01/raw_back.ply \
  --colmap_dir Object_01/sparse_back/0 \
  --mask_dir Object_01/back/input_mask \
  --output_ply Object_01/point_back/points3D_clean_back.ply \
  --threshold 0.5
```

The `threshold` parameter controls the minimum proportion of valid views in which a 3D point must fall inside the foreground mask to be retained.

### 11. Merge the front and back sides

After both cleaned point clouds are generated, run:

```bash
python together_pointcloud.py
```

Then enter the artifact dataset folder:

```text
Object_01/
```

The script reads:

```text
Object_01/point_front/
Object_01/point_back/
Object_01/sparse_front/0/
Object_01/sparse_back/0/
```

and generates:

```text
Object_01/point/Merged_*.ply
Object_01/sparse/0/
```

`Merged_*.ply` is the merged front-back point cloud. The `sparse/0/` folder contains the merged COLMAP sparse model.

The overall processing order is:

```text
mask_get → mask_apply → clear_point → together_pointcloud
```

The front and back sides should each complete mask generation, mask application, and point cloud cleaning before running the final front-back merging step.


## Acknowledgements

This project builds on several open-source tools and libraries:

- [SAM 2](https://github.com/facebookresearch/sam2) for semantic mask generation.
- [COLMAP](https://github.com/colmap/colmap) for SfM-MVS reconstruction.
