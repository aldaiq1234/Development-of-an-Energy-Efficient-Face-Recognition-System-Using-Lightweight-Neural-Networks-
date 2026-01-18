# Development of an Energy-Efficient Face Recognition System Using Lightweight Neural Networks

Diploma project (2025–2026) by Алла-Әділ  
Astana, Kazakhstan

## Overview
The project develops a lightweight face recognition model for edge devices (e.g., RK3588, Jetson Nano, Coral Dev Board) with:
- Target: ≥98.5% on LFW / ≥96% on MegaFace after INT8 quantization
- Power: ≤2.5 W average
- Real-time inference: ≤50 ms/frame

Architectures explored: MobileNetV3-Small, EfficientNet-Lite, PP-LCNet, ShuffleNetV2, GhostNet

## Folder Structure
- /Diploma/          → Main diploma files (thesis text, Risk Management Plan, presentations)
- /notebooks/        → Jupyter notebooks for training, quantization, benchmarking (add if not yet)
- /src/              → Inference scripts, TFLite conversion
- /experiments/      → Power/time measurements, logs
- /dataset/          → Scripts for data preparation (LFW, custom Kazakh faces – no raw data!)

## Setup
```bash
pip install -r requirements.txt
