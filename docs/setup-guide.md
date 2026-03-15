# vLLM + Gemma 3 27B on Dual RTX 3090 Setup Guide (March 2026)

## Executive Summary

Running **Gemma 3 27B IT** on dual RTX 3090 GPUs (2×24GB = 48GB total VRAM) requires **quantization** since the full BF16 model needs ~54GB. The recommended approach uses **W4A16 quantized models** with **tensor parallelism**.

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| **GPUs** | 2× RTX 3090 (24GB each) |
| **Total VRAM** | 48GB |
| **PSU** | 750W+ (each card is 350W TDP) |
| **NVLink** | ✅ Present (~112 GB/s bidirectional) |
| **CUDA Compute** | 8.6 (Ampere) - fully supported |

### NVLink Advantage
With NVLink connected, you get **~3.5x faster GPU-to-GPU communication** compared to PCIe 4.0:
- **NVLink**: ~112 GB/s bidirectional
- **PCIe 4.0**: ~32 GB/s per direction

This significantly improves tensor parallel performance, especially for:
- Larger batch sizes
- Longer context lengths
- Higher throughput inference

NCCL automatically detects and uses NVLink when available.

---

## Software Requirements

| Component | Version |
|-----------|---------|
| **vLLM** | v0.17.0 (latest, March 2026) |
| **Python** | 3.10 - 3.13 |
| **PyTorch** | 2.10.0 (required for vLLM 0.17.0) |
| **CUDA** | 12.4 recommended |
| **NVIDIA Driver** | 565+ (server variant for multi-GPU) |

---

## Memory Analysis

| Configuration | VRAM Needed | Fits 48GB? |
|---------------|-------------|------------|
| **BF16 (native)** | ~54GB | ❌ No |
| **FP8** | ~27GB | ✅ Yes |
| **W8A8 (INT8)** | ~27GB | ✅ Yes |
| **W4A16 (INT4)** | ~13.5GB | ✅ Yes (plenty headroom) |

**Recommendation:** Use **W4A16 quantized model** - achieves ~99.7% accuracy vs full precision.

---

## Model Options

| Model ID | Type | Notes |
|----------|------|-------|
| `google/gemma-3-27b-it` | Full precision | BF16, needs quantization flag |
| `RedHatAI/gemma-3-27b-it-quantized.w4a16` | W4A16 quantized | **Recommended** |
| `RedHatAI/gemma-3-27b-it-quantized.w8a8` | W8A8 quantized | Higher quality, more VRAM |

---

## Installation Plan

### Step 1: System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y build-essential dkms

# Blacklist nouveau driver (if not done)
sudo bash -c 'echo "blacklist nouveau" >> /etc/modprobe.d/blacklist.conf'
sudo update-initramfs -u
```

### Step 2: Install NVIDIA Driver (Server Variant)

```bash
# List available drivers
sudo ubuntu-drivers list --gpgpu

# Install server driver (includes multi-GPU support)
sudo ubuntu-drivers install --gpgpu nvidia:565-server

# Or specific version
sudo apt install nvidia-driver-565-server nvidia-utils-565-server

# Reboot
sudo reboot

# Verify
nvidia-smi
```

### Step 3: Install CUDA 12.4

```bash
# Download CUDA 12.4
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# Install toolkit only (driver already installed)
sudo sh cuda_12.4.0_550.54.14_linux.run --toolkit --silent

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

### Step 4: Create Python Environment & Install vLLM

```bash
# Create virtual environment
python -m venv ~/vllm-env
source ~/vllm-env/bin/activate

# Install vLLM with CUDA 12.4
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124

# Alternative: using uv (faster)
# uv pip install vllm --torch-backend=auto

# Verify
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'GPUs detected: {torch.cuda.device_count()}')"
```

### Step 5: Verify Multi-GPU Setup & NVLink

```bash
# Check GPU topology
nvidia-smi topo -m

# Expected output WITH NVLink:
#         GPU0    GPU1
# GPU0     X      NV2
# GPU1    NV2      X
# (NV2 = NVLink 2-way connection - optimal!)

# Verify NVLink status
nvidia-smi nvlink --status

# Check NVLink bandwidth
nvidia-smi nvlink --capabilities
```

---

## Running Gemma 3 27B

### Option A: W4A16 Quantized (Recommended)

```bash
source ~/vllm-env/bin/activate

vllm serve RedHatAI/gemma-3-27b-it-quantized.w4a16 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

### Option B: FP8 Dynamic Quantization

```bash
source ~/vllm-env/bin/activate

vllm serve google/gemma-3-27b-it \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --quantization fp8 \
    --enforce-eager \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

### Option C: Docker (Simplest)

```bash
docker run --gpus all \
    --shm-size=16g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model RedHatAI/gemma-3-27b-it-quantized.w4a16 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --max-model-len 8192
```

---

## Critical vLLM Flags for Gemma 3

| Flag | Value | Why |
|------|-------|-----|
| `--trust-remote-code` | Required | Gemma 3 uses custom code |
| `--enforce-eager` | Required | Workaround for torch.compile bug with multimodal |
| `--tensor-parallel-size` | `2` | Split across both GPUs |
| `--gpu-memory-utilization` | `0.90` | Use 90% of VRAM |
| `--max-model-len` | `8192` | Context length (Gemma 3 supports 128K, reduce for VRAM) |
| `--dtype` | `bfloat16` | Native Gemma 3 precision |

---

## Environment Variables

```bash
# GPU Selection (if needed)
export CUDA_VISIBLE_DEVICES=0,1

# NCCL debugging (optional)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not present

# Fix for CUDA 12.9+ cublas issue
unset LD_LIBRARY_PATH
```

---

## Testing the API

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "RedHatAI/gemma-3-27b-it-quantized.w4a16",
        "messages": [{"role": "user", "content": "Explain quantum computing in simple terms"}],
        "max_tokens": 512
    }'
```

---

## Complete Setup Script

```bash
#!/bin/bash
set -euo pipefail

echo "=== Setting up vLLM + Gemma 3 27B on Dual RTX 3090 ==="

# Step 1: Create environment
echo "[1/4] Creating Python environment..."
python -m venv ~/vllm-env
source ~/vllm-env/bin/activate

# Step 2: Install vLLM
echo "[2/4] Installing vLLM..."
pip install --upgrade pip
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124

# Step 3: Verify installation
echo "[3/4] Verifying installation..."
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Step 4: Download model (optional, vLLM will auto-download)
echo "[4/4] Pre-downloading model..."
huggingface-cli download RedHatAI/gemma-3-27b-it-quantized.w4a16

echo "=== Setup complete! ==="
echo ""
echo "To start the server:"
echo "  source ~/vllm-env/bin/activate"
echo "  vllm serve RedHatAI/gemma-3-27b-it-quantized.w4a16 \\"
echo "      --trust-remote-code \\"
echo "      --tensor-parallel-size 2 \\"
echo "      --enforce-eager \\"
echo "      --max-model-len 8192 \\"
echo "      --port 8000"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce `--max-model-len` to 4096 or use W4A16 model |
| torch.compile assertion error | Add `--enforce-eager` flag |
| GPUs not detected | Check `nvidia-smi`, verify driver |
| CUBLAS_STATUS_INVALID_VALUE | `unset LD_LIBRARY_PATH` or use CUDA 12.4 |
| NCCL timeout | Set `NCCL_DEBUG=INFO` to diagnose |
| Slow first inference | Model loading is slow; subsequent requests are fast |
| NVLink not detected | Check `nvidia-smi nvlink --status`, reseat bridge |
| Topology shows PHB not NV2 | NVLink bridge not connected or faulty |

---

## Performance Tips

1. **NVLink enables larger context**: With NVLink's 3.5x faster interconnect, you can push `--max-model-len` higher (try 16384 or 32768) without communication bottlenecks.

2. **Use vLLM v0.17.0's performance mode:**
   ```bash
   --performance-mode throughput  # Options: balanced, interactivity, throughput
   ```

3. **Pre-download the model** to avoid timeout during first serve:
   ```bash
   huggingface-cli download RedHatAI/gemma-3-27b-it-quantized.w4a16
   ```

4. **Increase shared memory** for Docker:
   ```bash
   --shm-size=16g
   ```

5. **Verify NVLink is being used** by checking NCCL debug output:
   ```bash
   NCCL_DEBUG=INFO vllm serve ... 2>&1 | grep -i nvlink
   ```

---

## Sources

- [vLLM Documentation](https://docs.vllm.ai/en/stable/)
- [vLLM GitHub Releases v0.17.0](https://github.com/vllm-project/vllm/releases)
- [google/gemma-3-27b-it on HuggingFace](https://huggingface.co/google/gemma-3-27b-it)
- [RedHatAI/gemma-3-27b-it-quantized.w4a16](https://huggingface.co/RedHatAI/gemma-3-27b-it-quantized.w4a16)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [Ubuntu NVIDIA Drivers](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/)
