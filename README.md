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

```text
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Place it at:

```text
third_party/sam2/checkpoints/sam2.1_hiera_large.pt
```

### 6. Configure COLMAP

COLMAP is used as the Structure-from-Motion and Multi-View Stereo reconstruction tool.

For Windows users, the official CUDA-enabled prebuilt package is available here:

```text
https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-cuda.zip
```

After extracting COLMAP, add the directory containing `colmap.exe` to your system `PATH`.

Before running `convert_dense.py`, check the `COLMAP_EXE` path in the script and modify it according to your local COLMAP installation:

```python
COLMAP_EXE = r"E:\3DGS\gaussian-splatting-main\3dgs_tools\colmap\bin\colmap.exe"
```


## Processing Pipeline

### 7. Prepare the artifact dataset

For each artifact, the front-side and back-side images should be placed in two separate `input` folders:

```text
Object_01/
├── front/
│   └── input/
│       ├── 0001.jpg
│       ├── 0002.jpg
│       └── ...
└── back/
    └── input/
        ├── 0001.jpg
        ├── 0002.jpg
        └── ...
```

Here, `front/input/` contains the original images of the front side, and `back/input/` contains the original images of the back side.

### 8. Reconstruct each side with COLMAP

Run `convert_dense.py` separately for the front side and the back side.

Front side:

```bash
python convert_dense.py --images Object_01/front/input --overwrite
```

Back side:

```bash
python convert_dense.py --images Object_01/back/input --overwrite
```

For a single side, the script uses the parent folder of `input/` as the workspace. Therefore, the outputs are generated inside `front/` or `back/`.

For the front side, the outputs are:

```text
Object_01/
└── front/
    ├── input/
    ├── database.db
    ├── sparse/
    │   └── 0/
    │       ├── cameras.bin
    │       ├── images.bin
    │       └── points3D.bin
    └── dense/
        └── fused.ply
```

For the back side, the outputs are:

```text
Object_01/
└── back/
    ├── input/
    ├── database.db
    ├── sparse/
    │   └── 0/
    │       ├── cameras.bin
    │       ├── images.bin
    │       └── points3D.bin
    └── dense/
        └── fused.ply
```

The file `dense/fused.ply` is the dense point cloud of the corresponding side. It will be used as the input point cloud for the point-cleaning step.

### 9. Generate foreground masks

Run `mask_get.py` separately for the front and back input folders:

```bash
python mask_get.py
```

Input examples:

```text
Object_01/front/input/
Object_01/back/input/
```

The script automatically generates mask folders next to the input folders:

```text
Object_01/front/input_mask/
Object_01/back/input_mask/
```

The mask filenames should keep the same filename stems as the original input images. For example:

```text
Object_01/front/input/0001.jpg
Object_01/front/input_mask/0001.jpg
```

### 10. Apply masks to input images

Run `mask_apply.py` separately for the front side and the back side:

```bash
python mask_apply.py
```

For the front side, enter:

```text
Input folder: Object_01/front/input/
Mask folder:  Object_01/front/input_mask/
```

For the back side, enter:

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

### 11. Clean front-side and back-side point clouds

Run `clear_point.py` separately for the front-side and back-side point clouds. This step projects 3D points into the image views and removes points that are inconsistent with the foreground masks.

Create the output folders first:

```bash
mkdir Object_01/point_front
mkdir Object_01/point_back
```

Front side:

```bash
python clear_point.py \
  --input_ply Object_01/front/dense/fused.ply \
  --colmap_dir Object_01/front/sparse/0 \
  --mask_dir Object_01/front/input_mask \
  --output_ply Object_01/point_front/points3D_clean_front.ply \
  --threshold 0.5
```

Back side:

```bash
python clear_point.py \
  --input_ply Object_01/back/dense/fused.ply \
  --colmap_dir Object_01/back/sparse/0 \
  --mask_dir Object_01/back/input_mask \
  --output_ply Object_01/point_back/points3D_clean_back.ply \
  --threshold 0.5
```

The cleaned point clouds are saved as:

```text
Object_01/point_front/points3D_clean_front.ply
Object_01/point_back/points3D_clean_back.ply
```

The `threshold` parameter controls the minimum proportion of valid views in which a 3D point must fall inside the foreground mask to be retained.

### 12. Prepare sparse folders for front-back merging

The merging script reads the sparse models from the following fixed folders:

```text
Object_01/sparse_front/0/
Object_01/sparse_back/0/
```

Therefore, copy the single-side sparse models to these folders before merging:

```bash
mkdir Object_01/sparse_front
mkdir Object_01/sparse_back
```

Copy the front-side sparse model:

```bash
cp -r Object_01/front/sparse/0 Object_01/sparse_front/0
```

Copy the back-side sparse model:

```bash
cp -r Object_01/back/sparse/0 Object_01/sparse_back/0
```

After this step, the dataset should contain:

```text
Object_01/
├── point_front/
│   └── points3D_clean_front.ply
├── point_back/
│   └── points3D_clean_back.ply
├── sparse_front/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── sparse_back/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

Each of `point_front/` and `point_back/` should contain only the final cleaned `.ply` file used for merging.

### 13. Merge the front and back sides

After the cleaned point clouds and sparse folders are prepared, run:

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

The file `point/Merged_*.ply` is the merged front-back point cloud of the artifact. The folder `sparse/0/` contains the merged COLMAP sparse model.

### 14. Final dataset structure

After completing the full pipeline, the dataset should be organized as follows:

```text
Object_01/
├── front/
│   ├── input/
│   ├── input_mask/
│   ├── input_apply_mask/
│   ├── database.db
│   ├── sparse/
│   │   └── 0/
│   │       ├── cameras.bin
│   │       ├── images.bin
│   │       └── points3D.bin
│   └── dense/
│       └── fused.ply
├── back/
│   ├── input/
│   ├── input_mask/
│   ├── input_apply_mask/
│   ├── database.db
│   ├── sparse/
│   │   └── 0/
│   │       ├── cameras.bin
│   │       ├── images.bin
│       └── points3D.bin
│   └── dense/
│       └── fused.ply
├── point_front/
│   └── points3D_clean_front.ply
├── point_back/
│   └── points3D_clean_back.ply
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
├── point/
│   └── Merged_*.ply
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

The overall workflow is:

```text
front/input → front/sparse + front/dense/fused.ply
back/input  → back/sparse  + back/dense/fused.ply

front/input → front/input_mask → front/input_apply_mask
back/input  → back/input_mask  → back/input_apply_mask

front/dense/fused.ply + front/sparse/0 + front/input_mask → point_front
back/dense/fused.ply  + back/sparse/0  + back/input_mask  → point_back

point_front + point_back + sparse_front + sparse_back → point + sparse
```


## Acknowledgements

This project builds on several open-source tools and libraries:

- [SAM 2](https://github.com/facebookresearch/sam2) for semantic mask generation.
- [COLMAP](https://github.com/colmap/colmap) for SfM-MVS reconstruction.
