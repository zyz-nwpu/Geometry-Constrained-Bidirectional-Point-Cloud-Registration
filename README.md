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

### 7. Prepare the input data

For each artifact, place the front-side and back-side input images in two separate folders:

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

The folder `input/` stores the original image sequence of one side of the artifact.

The following steps use `Object_01/front/input/` as an example. The same operations should also be applied to `Object_01/back/input/` before front-back merging.

### 8. Reconstruct one side with COLMAP

Run `convert_dense.py` on one input folder:

```bash
python convert_dense.py --images Object_01/front/input --overwrite
```

The script uses the parent folder of `input/` as the workspace. Therefore, for `Object_01/front/input/`, the output will be generated inside `Object_01/front/`:

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

Here:

```text
front/sparse/0/      COLMAP sparse reconstruction result
front/dense/fused.ply  dense point cloud of this side
```

Repeat this step for the other side:

```bash
python convert_dense.py --images Object_01/back/input --overwrite
```

### 9. Generate foreground masks

Run `mask_get.py` for one input folder:

```bash
python mask_get.py
```

When prompted, enter:

```text
Object_01/front/input/
```

The script generates the mask folder next to `input/`:

```text
Object_01/front/input_mask/
```

Example:

```text
Object_01/front/input/0001.jpg
Object_01/front/input_mask/0001.jpg
```

The generated mask files keep the same filename stems as the original input images.

Repeat this step for the other side:

```text
Object_01/back/input/
```

which generates:

```text
Object_01/back/input_mask/
```

### 10. Apply masks to input images

This step is optional. It is used only when masked input images with black backgrounds are needed.

Run `mask_apply.py`:

```bash
python mask_apply.py
```

For one side, enter:

```text
Input folder: Object_01/front/input/
Mask folder:  Object_01/front/input_mask/
```

The script generates:

```text
Object_01/front/input_apply_mask/
```

The output images keep the original filenames, with background regions set to black.

If masked images are also needed for the other side, repeat this step using:

```text
Input folder: Object_01/back/input/
Mask folder:  Object_01/back/input_mask/
```

which generates:

```text
Object_01/back/input_apply_mask/
```

### 11. Clean the point cloud of one side

Run `clear_point.py` to remove background points and reconstruction noise from the dense point cloud.

Create the output folder:

```bash
mkdir Object_01/point_front
```

For the front side, run:

```bash
python clear_point.py \
  --input_ply Object_01/front/dense/fused.ply \
  --colmap_dir Object_01/front/sparse/0 \
  --mask_dir Object_01/front/input_mask \
  --output_ply Object_01/point_front/point_clean_front.ply \
  --threshold 0.5
```

This step projects 3D points into the image views and keeps only the points that are sufficiently consistent with the foreground masks.

The cleaned point cloud is saved as:

```text
Object_01/point_front/point_clean_front.ply
```

Then repeat the same operation for the other side:

```bash
mkdir Object_01/point_back
```

```bash
python clear_point.py \
  --input_ply Object_01/back/dense/fused.ply \
  --colmap_dir Object_01/back/sparse/0 \
  --mask_dir Object_01/back/input_mask \
  --output_ply Object_01/point_back/point_clean_back.ply \
  --threshold 0.5
```

The cleaned back-side point cloud is saved as:

```text
Object_01/point_back/point_clean_back.ply
```

The `threshold` parameter controls the minimum proportion of valid views in which a 3D point must fall inside the foreground mask to be retained.

### 12. Prepare sparse folders for front-back merging

The merging script reads the sparse models from the following fixed folders:

```text
Object_01/sparse_front/0/
Object_01/sparse_back/0/
```

Therefore, copy the reconstructed sparse models to these folders before merging.

Create the target folders:

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

After this step, the required files for merging should be:

```text
Object_01/
├── point_front/
│   └── point_clean_front.ply
├── point_back/
│   └── point_clean_back.ply
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

Run `together_pointcloud.py`:

```bash
python together_pointcloud.py
```

When prompted, enter the artifact dataset folder:

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

Here:

```text
Object_01/point/Merged_*.ply  merged front-back point cloud
Object_01/sparse/0/           merged COLMAP sparse model
```

### 14. Final dataset structure

After completing the full pipeline, the dataset should be organized as follows:

```text
Object_01/
├── front/
│   ├── input/
│   ├── input_mask/
│   ├── input_apply_mask/        # optional
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
│   ├── input_apply_mask/        # optional
│   ├── database.db
│   ├── sparse/
│   │   └── 0/
│   │       ├── cameras.bin
│   │       ├── images.bin
│   │       └── points3D.bin
│   └── dense/
│       └── fused.ply
├── point_front/
│   └── point_clean_front.ply
├── point_back/
│   └── point_clean_back.ply
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
input → sparse/0 + dense/fused.ply
input → input_mask
input + input_mask → input_apply_mask    # optional
dense/fused.ply + sparse/0 + input_mask → point_clean_*.ply
point_clean_front.ply + point_clean_back.ply → Merged_*.ply
```


## Acknowledgements

This project builds on several open-source tools and libraries:

- [SAM 2](https://github.com/facebookresearch/sam2) for semantic mask generation.
- [COLMAP](https://github.com/colmap/colmap) for SfM-MVS reconstruction.
