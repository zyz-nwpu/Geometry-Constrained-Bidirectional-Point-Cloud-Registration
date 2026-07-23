# Geometry-Constrained Bidirectional Point Cloud Registration for Thin, Sheet-Like Heritage Artifacts

<p align="center">
  <strong>GCBPCR</strong>
</p>

<p align="center">
  <a href="https://zyz-nwpu.github.io/Geometry-Constrained-Bidirectional-Point-Cloud-Registration/">
    <img alt="Project page" src="https://img.shields.io/badge/Project-Page-1a7f64?style=for-the-badge">
  </a>
  <img alt="Paper status" src="https://img.shields.io/badge/Paper-Coming_Soon-6e7781?style=for-the-badge">
  <a href="LICENSE">
    <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-24292f?style=for-the-badge">
  </a>
</p>

This repository provides the official implementation of **Geometry-Constrained Bidirectional Point Cloud Registration (GCBPCR)** for thin, sheet-like heritage artifacts. The pipeline supports independent front- and back-side reconstruction, semantic-guided 2D-3D point-cloud purification, PCA-based canonical normalization, thickness-aware registration, global rotation-hypothesis evaluation, and point-to-plane ICP refinement.

The repository name and acronym follow the core method name in the paper. The application domain, *thin, sheet-like heritage artifacts*, remains explicit in the full paper title and throughout the documentation.

## Method Overview

```text
front/back images
        |
        v
independent SfM-MVS reconstruction
        |
        v
semantic-guided 2D-3D purification
        |
        v
PCA normalization and thickness estimation
        |
        v
rotation-hypothesis evaluation + point-to-plane ICP
        |
        v
geometry-aware selection and merged reconstruction
```

## Environment

The reference setup uses Python 3.10, a CUDA-capable PyTorch installation, COLMAP, and Segment Anything Model 2 (SAM 2).

```bash
conda create -n gcbpcr python=3.10 -y
conda activate gcbpcr
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA runtime. For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Initialize and install the third-party dependencies:

```bash
git submodule update --init --recursive
pip install -e third_party/sam2
```

Download the [SAM 2.1 Hiera Large checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) and place it at:

```text
third_party/sam2/checkpoints/sam2.1_hiera_large.pt
```

Install [COLMAP](https://colmap.github.io/install.html) and either make `colmap` available on `PATH`, set the `COLMAP_EXE` environment variable, or pass `--colmap /path/to/colmap` to `convert_dense.py`.

## Data Organization

For each artifact, place the front- and back-side images in separate input directories:

```text
artifact/
  front/input/
  back/input/
```

Before registration, arrange the purified point clouds and sparse models as follows:

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
| `mask_get.py` | Generates foreground masks with SAM 2. |
| `mask_apply.py` | Applies foreground masks to image sequences when masked inputs are needed. |
| `clear_point.py` | Filters a dense point cloud using COLMAP camera geometry and foreground masks. |
| `together_pointcloud.py` | Registers and merges front/back point clouds and their sparse COLMAP models. |

## Processing

Run dense reconstruction for both sides:

```bash
python convert_dense.py --images artifact/front/input --overwrite
python convert_dense.py --images artifact/back/input --overwrite
```

Generate foreground masks:

```bash
python mask_get.py --images artifact/front/input
python mask_get.py --images artifact/back/input
```

Optionally apply the masks to the input images:

```bash
python mask_apply.py --images artifact/front/input --masks artifact/front/input_mask
python mask_apply.py --images artifact/back/input --masks artifact/back/input_mask
```

Purify the front-side dense point cloud:

```bash
python clear_point.py \
  --input_ply artifact/front/dense/fused.ply \
  --colmap_dir artifact/front/sparse/0 \
  --mask_dir artifact/front/input_mask \
  --output_ply artifact/point_front/point_clean_front.ply \
  --threshold 0.9
```

Repeat this command for the back-side reconstruction with the corresponding paths, then run registration and merging:

```bash
python together_pointcloud.py --dataset artifact
```

The merged point cloud and sparse model are written to:

```text
artifact/point/
artifact/sparse/0/
```

## Citation

The manuscript is currently submitted for publication. If this repository supports your research, please cite the paper using the metadata in [`CITATION.cff`](CITATION.cff). The final bibliographic record and paper URL will be added after publication.

```bibtex
@article{zhang2026gcbpcr,
  title   = {Geometry-Constrained Bidirectional Point Cloud Registration for Thin, Sheet-Like Heritage Artifacts},
  author  = {Zhang, Yuezhe and Wei, Lei and Du, Jingnan and Wan, Shuai},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

## Acknowledgements

We thank the open-source communities behind [Segment Anything Model 2](https://github.com/facebookresearch/sam2) and [COLMAP](https://github.com/colmap/colmap), which support semantic mask generation and multi-view reconstruction in this work.

## License

This project is released under the [MIT License](LICENSE).
