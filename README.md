# Is the Straight-Through Estimator Enough for Quantized Test-Time Adaptation?

This repository contains the implementation and experimental notebooks for a study on quantized test-time adaptation (QTTA). The project investigates whether the standard straight-through estimator (STE) is enough for adapting quantized neural networks at test time, or whether a Gaussian-smoothed surrogate gradient is needed to stabilize optimization under quantization.

## Overview

The project focuses on:

- ResNet-20 trained on CIFAR-10
- Fake-quantization for activations and weights
- Calibration of quantization scales
- Test-time adaptation under quantized settings
- Comparison between STE and Gaussian-smoothed surrogate gradients
- Reproducible experiments through Jupyter notebooks

## Repository structure

- `Notebook/` — experiment notebooks for training, quantization checks, adaptation, and figure generation
- `qtta-code/qtta_lib.py` — core implementation for the quantized model, fake quantizer, and surrogate gradient logic
- `qtta-code/resnet20_fp32.pt` — pretrained FP32 ResNet-20 checkpoint
- `qtta-code/train_log.json` — recorded training metrics
- `resullts/` — result artifacts and experimental outputs

## Notebooks

The notebook sequence covers the main workflow:

1. `Notebook/01_train_resnet20_fp32.ipynb` — train the full-precision ResNet-20 baseline
2. `Notebook/02_quantizer_tests_ptq.ipynb` — evaluate post-training quantization behavior
3. `Notebook/03_tta_cifar10c_fixed_sigma.ipynb` — test-time adaptation experiments on CIFAR-10-C
4. `Notebook/04_scan_where_ste_fails.ipynb` — diagnose failure modes of the STE formulation
5. `Notebook/05_weight_adaptation.ipynb` — experiments with weight adaptation
6. `Notebook/06_oracle_adaptive_sigma.ipynb` — adaptive-sigma ablation studies
7. `Notebook/07_figures.ipynb` — generate figures and summaries

## Requirements

The project is built around PyTorch and torchvision. Typical dependencies include:

- Python 3.9+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Jupyter

Example install:

```bash
pip install torch torchvision numpy matplotlib jupyter
```

## Usage

Open the notebooks in the `Notebook/` directory in order. The implementation in `qtta-code/qtta_lib.py` provides the core quantization and adaptation logic used by the experiments.

If you want to run the code outside the notebooks, start from the helper functions and model definitions in `qtta-code/qtta_lib.py` and adapt the paths for your local environment.

## Notes

This repository is mainly research-oriented and uses notebook-driven experimentation. It is intended to reproduce the study and inspect how quantized test-time adaptation behaves under different surrogate-gradient settings.

## Citation

If you use this project in your work, please cite the associated paper or project findings if available in the original publication. The repository title reflects the core question under investigation: whether the straight-through estimator alone is sufficient for quantized test-time adaptation.
