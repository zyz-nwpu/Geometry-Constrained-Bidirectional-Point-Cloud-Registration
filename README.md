# Geometry-Constrained-Bidirectional-Registration
[![Hits](https://hits.sh/github.com/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration.svg?label=views)](https://hits.sh/github.com/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration/)
Official implementation of geometry-constrained bidirectional point cloud registration for thin, sheet-like heritage artifacts.


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

### 4. Install the required third-party segmentation package

```bash
cd third_party/sam2
pip install -e .
cd ../..
```

If the CUDA extension fails to build during installation, the package can usually still be used for mask prediction, although some optional post-processing functions may be limited.

### 5. Prepare the checkpoint file

Create a checkpoint folder inside `third_party/sam2`:

Download the checkpoint from [here](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt).

Place the downloaded file under:

```text
third_party/sam2/checkpoints/
```

The final checkpoint path should be:

```text
third_party/sam2/checkpoints/sam2.1_hiera_large.pt
```





## Acknowledgements

This project uses several third-party tools and libraries. We thank the authors and contributors of the following projects:

- [SAM 2](https://github.com/facebookresearch/sam2), which is used for semantic mask generation during the preprocessing stage.
- [COLMAP](https://github.com/colmap/colmap), which is used for Structure-from-Motion (SfM) and Multi-View Stereo (MVS) reconstruction.

