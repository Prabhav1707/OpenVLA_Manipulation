# OpenVLA Manipulation Demo

A self-contained demo of a Vision-Language-Action (VLA) model controlling a simulated robot arm in PyBullet.

## What this does

- Loads **OpenVLA-7B** (Stanford, 2024) in 4-bit quantization on a consumer GPU
- Sets up a **PyBullet** tabletop simulation with a Franka Panda arm and a red cube
- Runs a closed-loop inference pipeline:
  - Camera image captured from simulation
  - Image + language instruction fed into OpenVLA
  - Predicted 7-DOF end-effector action applied to the arm via IK
  - Repeat

## Stack

| Component | Tool |
|---|---|
| VLA model | OpenVLA-7B (Llama 2 7B + SigLIP) |
| Quantization | bitsandbytes 4-bit NF4 |
| Simulation | PyBullet |
| ML framework | PyTorch + HuggingFace Transformers |
| IK solver | PyBullet built-in |
| Hardware | NVIDIA RTX 3060 6GB VRAM |

## Setup

```bash
conda create -n openvla python=3.10 -y
conda activate openvla
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 accelerate==0.30.0 bitsandbytes huggingface_hub timm==0.9.16 einops
pip install pybullet numpy matplotlib pillow
```

Login to Hugging Face (required to download model weights):
```bash
hf auth login
```

## Run

```bash
# Test inference only
python3 inference.py

# Full simulation loop
python3 main.py
```

## How it works

OpenVLA takes a 224x224 camera image and a language instruction like *"pick up the red cube"* and outputs 7 floats representing end-effector deltas [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]. These are applied to the Franka Panda arm via PyBullet's built-in IK solver at each control step.

## Background

Built as a self-directed project to understand VLA model architecture and inference pipelines, in preparation for applied robotics work in industrial automation.

Key concepts covered:
- Vision Transformer (SigLIP) image encoding
- Action tokenization — continuous joint values discretized into 256 bins mapped to LLM vocabulary tokens
- 4-bit quantization (NF4) for running 7B models on consumer GPUs
- Closed-loop robot control via sim camera feedback
