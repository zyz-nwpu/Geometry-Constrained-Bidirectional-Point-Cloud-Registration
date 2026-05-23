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
### 6. Build COLMAP from Source

COLMAP is included as a third-party submodule under:

```text
third_party/colmap
```

The COLMAP source code is used for attribution and reproducibility. If you want to build COLMAP from source with CUDA support, please install the following system-level tools first:

```text
Visual Studio 2019 or newer with C++ build tools
NVIDIA CUDA Toolkit
CMake
Git
VCPKG
```

COLMAP compilation is not installed inside the `gcbreg` Conda environment. The Conda environment is used for the Python-based parts of this project, while COLMAP is compiled and used as an external executable.

First, enter the project root directory and initialize the submodules:

```bash
git submodule update --init --recursive
```

Then enter the COLMAP source directory:

```bash
cd third_party/colmap
```

Install VCPKG under the `third_party` directory:

```bash
cd ..
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
```

Build COLMAP with CUDA support through VCPKG:

```bash
vcpkg install colmap[cuda,tests]:x64-windows
```

After compilation, COLMAP can be tested with:

```bash
colmap -h
```

If `colmap` is not recognized, add the generated COLMAP executable directory to the system `PATH`, or call the executable using its full path.


## Acknowledgements

This project uses several third-party tools and libraries. We thank the authors and contributors of the following projects:

- [SAM 2](https://github.com/facebookresearch/sam2), which is used for semantic mask generation during the preprocessing stage.
- [COLMAP](https://github.com/colmap/colmap), which is used for Structure-from-Motion (SfM) and Multi-View Stereo (MVS) reconstruction.

