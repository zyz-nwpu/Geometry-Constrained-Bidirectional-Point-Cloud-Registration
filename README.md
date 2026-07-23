# Geometry-Constrained Bidirectional Point Cloud Registration for Thin, Sheet-Like Heritage Artifacts

<p align="center">
  <a href="https://zyz-nwpu.github.io/Geometry-Constrained-Bidirectional-Point-Cloud-Registration/">
    <img alt="Project" src="https://img.shields.io/badge/Project-Page-1a7f64?style=for-the-badge">
  </a>
</p>

This repository provides the official implementation of Geometry-Constrained Bidirectional Point Cloud Registration (GCBPCR) for Thin, Sheet-Like Heritage Artifacts. The code supports double-sided reconstruction, semantic mask-guided point cloud purification, and geometry-constrained front-back registration for thin, sheet-like heritage objects.

## Environment

Create and activate the Conda environment:

```bash
conda create -n gcbpcr python=3.10 -y
conda activate gcbpcr
```

Install PyTorch with GPU support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

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

Download the Segment Anything Model 2 checkpoint from [here](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt).

Place it at:

```text
third_party/sam2/checkpoints/sam2.1_hiera_large.pt
```

Install COLMAP and make `colmap` available on `PATH`, set `COLMAP_EXE`, or pass its path with `--colmap`.

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

## Acknowledgements

We gratefully acknowledge the open-source communities behind Segment Anything Model 2, available [here](https://github.com/facebookresearch/sam2), and COLMAP, available [here](https://github.com/colmap/colmap), whose tools support semantic mask generation and multi-view reconstruction in this work.

## License

This project is released under the MIT License.
