# FedNeRF-Privacy3D

## Federated Neural Radiance Fields for Privacy-Preserving 3D Scene Reconstruction

---

## Overview

FedNeRF-Privacy3D is a research-oriented implementation of Federated Neural Radiance Fields (FedNeRF) for privacy-preserving 3D scene reconstruction.

Instead of sharing raw images captured by multiple cameras, each client trains a local Tiny-NeRF model and shares only privacy-preserving model parameters.

The server aggregates these parameters using Federated Averaging (FedAvg) to build a global NeRF capable of reconstructing the scene while preserving data privacy.

---

## Features

- Hybrid Tiny-NeRF
- Federated Learning (FedAvg)
- Gradient Clipping
- Gaussian Noise Privacy
- Novel View Synthesis
- PSNR, SSIM, LPIPS Evaluation
- Google Colab Training
- VS Code Development
- Research-quality Modular Code

---

## Project Structure

```
FedNeRF_Privacy3D/

├── notebooks/
├── src/
├── datasets/
├── checkpoints/
├── outputs/
├── logs/
├── docs/
└── tests/
```

---

## Current Progress

- [x] Module 1 – Environment Setup
- [ ] Module 2 – Dataset Management
- [ ] Module 3 – Dataset Loader
- [ ] Module 4 – Camera Mathematics
- [ ] Module 5 – Ray Generation
- [ ] Module 6 – Positional Encoding
- [ ] Module 7 – Hybrid Tiny-NeRF
- [ ] Module 8 – Local Training
- [ ] Module 9 – Federated Learning
- [ ] Module 10 – Privacy Module
- [ ] Module 11 – Global Reconstruction
- [ ] Module 12 – Evaluation
- [ ] Module 13 – Experiments
- [ ] Module 14 – Final Demonstration

---

## Author

Engineering Final Year Major Project

Open Source Educational Implementation