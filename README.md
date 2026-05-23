# Geometry-Constrained-Bidirectional-Registration

<p align="center">
  <a href="https://github.com/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration">
    <img src="https://img.shields.io/github/stars/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration/fork">
    <img src="https://img.shields.io/github/forks/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration?style=social" alt="GitHub forks">
  </a>
  <a href="https://github.com/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration/watchers">
    <img src="https://img.shields.io/github/watchers/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration?style=social" alt="GitHub watchers">
  </a>
</p>

<p align="center">
  <a href="https://hits.sh/github.com/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration/">
    <img src="https://hits.sh/github.com/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration.svg?label=views" alt="Repository views">
  </a>
  <img src="https://img.shields.io/github/issues/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration" alt="GitHub issues">
  <img src="https://img.shields.io/github/license/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration" alt="License">
  <img src="https://img.shields.io/github/last-commit/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration" alt="Last commit">
  <img src="https://img.shields.io/github/repo-size/zyz-nwpu/Geometry-Constrained-Bidirectional-Registration" alt="Repo size">
</p>

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

### 4. Install the required SAM2 package

```bash
cd third_party/sam2
pip install -e .
cd ../..
```

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
### 6. Configure COLMAP

COLMAP is used as the external Structure-from-Motion (SfM) and Multi-View Stereo (MVS) reconstruction tool. You can either download the official pre-built package or build it from source.

#### Option A: Download the pre-built package

For Windows users, the official CUDA-enabled pre-built package can be downloaded from [here](https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-cuda.zip).

After downloading and extracting the package, add the directory containing `colmap.exe` to the system `PATH` environment variable.

Test COLMAP with:

```bash
colmap -h
```

#### Option B: Build from source

The COLMAP source code is included as a third-party submodule under:

```text
third_party/colmap
```

To build COLMAP from source with CUDA support, make sure the following system-level tools are installed:

```text
Visual Studio 2019 or newer with C++ build tools
NVIDIA CUDA Toolkit
CMake
Git
VCPKG
```

Build COLMAP with VCPKG:

```bash
git submodule update --init --recursive
cd third_party
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
vcpkg install colmap[cuda,tests]:x64-windows
```

### 7. Dataset Structure


## Acknowledgements

This project uses several third-party tools and libraries. We thank the authors and contributors of the following projects:

- [SAM 2](https://github.com/facebookresearch/sam2), which is used for semantic mask generation during the preprocessing stage.
- [COLMAP](https://github.com/colmap/colmap), which is used for Structure-from-Motion (SfM) and Multi-View Stereo (MVS) reconstruction.

