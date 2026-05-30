# LagrangianSplats: Divergence-Free Transport of Gaussian Primitives for Fluid Reconstruction

Official implementation of the SIGGRAPH 2026 paper:
**LagrangianSplats: Divergence-Free Transport of Gaussian Primitives for Fluid Reconstruction**.

[Project Page](https://taoningxiao.github.io/lagrangiansplats/) | [Paper (arXiv)](https://arxiv.org/abs/2605.09299) | [Dataset (Hugging Face)](https://huggingface.co/datasets/tnx123/LagrangianSplats-Dataset)

![](imgs/representative.jpg)

## Overview

LagrangianSplats reconstructs physically valid 3D fluid motion from sparse 2D observations by transporting Gaussian primitives with a divergence-free velocity representation.

Our framework couples:

- A Lagrangian 3D Gaussian Splatting representation for transport-consistent motion modeling.
- A continuous divergence-free kernel velocity field that enforces incompressibility by construction.
- A sliding-window optimization scheme that enables tractable long-horizon gradient propagation.

## Abstract

Reconstructing 3D fluid velocity fields from sparse 2D video observations is a highly ill-posed inverse problem, demanding both transport consistency with observed motion and physical validity under fluid laws. Existing methods typically impose these constraints through soft penalties, often leading to compromised accuracy and convergence issues. We introduce a reconstruction framework that structurally enforces both constraints.

Specifically, we parameterize the reconstructed velocity using a continuous divergence-free kernel representation, driving the advection of a Lagrangian 3D Gaussian Splatting representation. This formulation intrinsically guarantees both flow incompressibility and long-range transport coherence by construction. To enable efficient optimization of such a constrained system, we introduce a sliding-window scheme that propagates gradients over meaningful temporal horizons while maintaining tractable training costs.

## Installation

The code has been tested with Python 3.7, PyTorch 1.13.1, and CUDA 11.7 on NVIDIA RTX 4090.

```bash
git submodule update --init --recursive
conda env create -f environment.yml
conda activate lagrangian-splats
bash scripts/install.sh
```

Make sure the CUDA toolkit visible through `nvcc` matches the CUDA version used by PyTorch. For example, CUDA extensions can fail to build if PyTorch is compiled with CUDA 11.7 but `nvcc` is CUDA 12.x.

Check versions before building:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version
```

The CUDA extensions are built from:

```text
submodules/depth-diff-gaussian-rasterization
submodules/simple-knn
```

Optional sanity check after installation:

```bash
python -c "import diff_gaussian_rasterization; import simple_knn._C; print('CUDA extensions OK')"
```

## Dataset

Scene data and ground-truth volumes are hosted on [Hugging Face](https://huggingface.co/datasets/tnx123/LagrangianSplats-Dataset).

Download one scene:

```bash
export LAGRANGIAN_SPLATS_DATA_URL=https://huggingface.co/datasets/tnx123/LagrangianSplats-Dataset/resolve/main
bash scripts/download_data.sh scalarsyn
```

Replace `scalarsyn` with any scene name listed below, or use `all` to download all released scenes.

Expected dataset assets:

```text
data/<scene>.tar.gz
gt/<scene>.tar.gz
SHA256SUMS
```

Expected local structure after extraction:

```text
data/smoke/scalarsyn
data/smoke/scalarreal
data/smoke/suzanne
data/smoke/sphere
data/smoke/biplume

gt/scalarsyn
gt/suzanne
gt/sphere
gt/biplume
```

`scalarreal` is real-captured and does not include default GT volumes. GT files are only required for evaluation against reference density/velocity fields.

## Quick Start

Run reconstruction and evaluation for one scene:

```bash
python main.py --configs configs/scenes/<scene>.py
```

Available scenes:

```text
scalarsyn
scalarreal
suzanne
sphere
biplume
```

Outputs are saved under `log/<expname>_<timestamp>/` (including checkpoints, visualizations, and videos).

## Citation

If you find this project useful, please cite:

```bibtex
@article{tao2026lagrangiansplats,
  title={LagrangianSplats: Divergence-Free Transport of Gaussian Primitives for Fluid Reconstruction},
  author={Tao, Ningxiao and Chen, Baoquan and Chu, Mengyu},
  journal={arXiv preprint arXiv:2605.09299},
  year={2026}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
