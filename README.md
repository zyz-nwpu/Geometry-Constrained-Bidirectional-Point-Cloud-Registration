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

This repository contains the implementation and project materials for geometry-constrained bidirectional point cloud registration of thin cultural heritage artifacts.

Project page:

**https://zyz-nwpu.github.io/Geometry-Constrained-Bidirectional-Registration**

## Highlights

- Geometry-constrained bidirectional registration for thin, sheet-like heritage artifacts.
- Semantic-guided 2D-3D purification for artifact-only point cloud reconstruction.
- PCA-based canonical normalization and thickness-aware registration constraints.
- Visual results for multiple thin cultural heritage artifacts.

## Method Overview

The pipeline contains three main stages:

1. **Semantic-guided purification**: reconstruct front- and back-side point clouds independently and remove background structures using image masks and 3D filtering.
2. **Geometric normalization**: use PCA to define canonical axes and estimate the artifact thickness direction.
3. **Bidirectional registration**: evaluate rotation hypotheses, refine candidates with ICP, and select the geometry-consistent alignment.

---

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

Check whether PyTorch can access the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
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

Test the installation:

```bash
colmap -h
```

## Acknowledgements

This project builds on several open-source tools and libraries:

- [SAM 2](https://github.com/facebookresearch/sam2) for semantic mask generation.
- [COLMAP](https://github.com/colmap/colmap) for SfM-MVS reconstruction.
