# Geometry-Constrained Bidirectional Point Cloud Registration for Thin Cultural Heritage Artifacts

<p align="center">
  <a href="https://zyz-nwpu.github.io/Geometry-Constrained-Bidirectional-Registration">
    <img alt="Project" src="https://img.shields.io/badge/Project-Page-1a7f64?style=for-the-badge">
  </a>
  <span>
    <img alt="Paper" src="https://img.shields.io/badge/Paper-Coming%20Soon-24292f?style=for-the-badge">
  </span>
</p>

This repository provides the implementation for geometry-constrained bidirectional point cloud registration of thin cultural heritage artifacts. The workflow reconstructs the front and back sides independently, extracts artifact-only geometry with semantic masks, and merges both sides under geometry-aware registration constraints.

## Environment

Create and activate the Conda environment:

```bash
conda create -n gcbreg python=3.10 -y
conda activate gcbreg
```

Install the PyTorch build that matches the local CUDA environment from [here](https://pytorch.org/get-started/locally/).

Initialize the third-party dependencies:

```bash
git submodule update --init --recursive
```

Install Segment Anything Model 2 from `third_party/sam2`:

```bash
cd third_party/sam2
pip install -e .
cd ../..
```

Download the Segment Anything Model 2 checkpoint from [here](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) and place it at:

```text
third_party/sam2/checkpoints/sam2.1_hiera_large.pt
```

Install COLMAP from [here](https://github.com/colmap/colmap) or download the Windows CUDA package from [here](https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-cuda.zip), then set `COLMAP_EXE` in `convert_dense.py` to the path of `colmap.exe`.

## Data Organization

For each artifact, place the front-side and back-side images in separate `input` folders:

```text
artifact/
  front/input/
  back/input/
```

Intermediate reconstruction results are generated beside each input folder. Before registration, arrange the purified point clouds and sparse models as:

```text
artifact/
  point_front/point_clean_front.ply
  point_back/point_clean_back.ply
  sparse_front/0/
  sparse_back/0/
```

## Scripts

| Script | Purpose |
| --- | --- |
| `convert_dense.py` | Runs COLMAP sparse reconstruction and dense fusion for one image sequence. |
| `mask_get.py` | Generates foreground masks with Segment Anything Model 2. |
| `mask_apply.py` | Applies foreground masks to images when masked image sequences are required. |
| `clear_point.py` | Filters dense point clouds by projecting 3D points into mask-validated image views. |
| `together_pointcloud.py` | Registers and merges the front-side and back-side point clouds and sparse models. |

## Processing

Run dense reconstruction for each side:

```bash
python convert_dense.py --images artifact/front/input --overwrite
python convert_dense.py --images artifact/back/input --overwrite
```

Generate foreground masks for each side:

```bash
python mask_get.py --images artifact/front/input
python mask_get.py --images artifact/back/input
```

Optionally apply masks to the input images:

```bash
python mask_apply.py --images artifact/front/input --masks artifact/front/input_mask
python mask_apply.py --images artifact/back/input --masks artifact/back/input_mask
```

Purify the reconstructed dense point cloud:

```bash
python clear_point.py \
  --input_ply artifact/front/dense/fused.ply \
  --colmap_dir artifact/front/sparse/0 \
  --mask_dir artifact/front/input_mask \
  --output_ply artifact/point_front/point_clean_front.ply \
  --threshold 0.9
```

Run the same filtering step for the back side by replacing the front-side paths with the corresponding back-side reconstruction, masks, and output point cloud.

Merge the front-side and back-side reconstructions:

```bash
python together_pointcloud.py --dataset artifact
```

The final merged point cloud and sparse model are written to:

```text
artifact/point/
artifact/sparse/0/
```

The complete workflow is therefore:

```text
front/back images -> dense reconstruction -> masks -> purified point clouds -> bidirectional registration
```

## Acknowledgements

We gratefully acknowledge the open-source communities behind Segment Anything Model 2, available [here](https://github.com/facebookresearch/sam2), and COLMAP, available [here](https://github.com/colmap/colmap), whose tools support semantic mask generation and multi-view reconstruction in this work.
