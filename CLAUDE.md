# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wan2.1 is an open-source large-scale video generative model that supports text-to-video, image-to-video, first-last-frame-to-video, video editing (VACE), and text-to-image generation. The system is built on a Diffusion Transformer (DiT) architecture with Flow Matching scheduling.

## Development Commands

### Installation
```bash
pip install -r requirements.txt
# For development features:
pip install -e ".[dev]"
```

### Code Formatting
```bash
make format  # Runs isort and yapf on generate.py, gradio/, and wan/
```

### Testing
```bash
# Run comprehensive test suite with models
./tests/test.sh <model_directory> <num_gpus>

# Example test commands from test.sh:
python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./model_dir
torchrun --nproc_per_node=8 generate.py --task t2v-14B --dit_fsdp --t5_fsdp --ulysses_size 8 --ckpt_dir ./model_dir
```

### Main Generation Script
```bash
# Single GPU
python generate.py --task [t2v-14B|t2v-1.3B|i2v-14B|flf2v-14B|vace-14B|vace-1.3B|t2i-14B] --size WxH --ckpt_dir ./model_path --prompt "your prompt"

# Multi-GPU with FSDP
torchrun --nproc_per_node=8 generate.py --dit_fsdp --t5_fsdp --ulysses_size 8 [other args]
```

### Gradio Demos
Located in `gradio/` directory with task-specific Python files for interactive demos.

## Architecture Overview

The codebase follows a modular pipeline architecture:

### Core Pipeline Flow
```
Input (Text/Image/Video) → Encoding → DiT Transformer → VAE Decoder → Output Video
```

### Key Components (`wan/` module)

**Main Generation Classes:**
- `WanT2V`: Text-to-video generation
- `WanI2V`: Image-to-video with CLIP conditioning
- `WanFLF2V`: First-last frame interpolation
- `WanVace/WanVaceMP`: Video editing with masks and references

**Core Model Architecture (`wan/modules/`):**
- `model.py`: WanModel - Main Diffusion Transformer with 3D patch embedding, dual attention (self + cross), and Flow Matching
- `vae.py`: WanVAE - 3D Video Autoencoder with causal convolutions and temporal compression
- `attention.py`: Optimized attention mechanisms with Flash Attention support
- `t5.py`, `clip.py`: Text and image encoders for conditioning

**Configuration System (`wan/configs/`):**
- Shared configurations across models with task-specific parameter files
- Support for 1.3B and 14B model variants

### Multi-Modal Conditioning
- **Text**: T5-UMT5-XXL encoder for multilingual text
- **Images**: CLIP ViT-H/14 for visual conditioning
- **Videos**: VAE latent encoding with frame-level conditioning

### Memory Optimizations
- Model offloading (`--offload_model True`)
- CPU text encoding (`--t5_cpu`)
- Flash Attention for efficient computation
- FSDP + xDiT USP for distributed inference

## Configuration Patterns

**Resolution Support:**
- 480P: 832x480 (supported by all models)
- 720P: 1280x720 (14B models only)
- Custom resolutions via `--size WxH` parameter

**Multi-GPU Strategies:**
- `--ulysses_size N`: Attention parallelism (requires num_heads divisible by N)
- `--ring_size N`: Sequence parallelism (requires sequence_length divisible by N)
- Use Ring strategy for 1.3B model (12 heads not divisible by 8 GPUs)

**Prompt Extension:**
- `--use_prompt_extend`: Enable prompt enhancement
- `--prompt_extend_method [local_qwen|dashscope]`: Local vs API-based extension
- `--prompt_extend_model`: Specify Qwen model for local extension

## File Structure Notes

- `generate.py`: Main inference script with comprehensive CLI arguments
- `wan/text2video.py`, `wan/image2video.py`, etc.: Task-specific pipeline implementations  
- `wan/utils/`: Utilities for prompt extension, format conversion, and Flow Matching solvers
- `ComfyUI-HunyuanLoom/` and `FlowEdit/`: Integrated third-party tools
- Example files in `examples/` for testing different generation modes

## Important Implementation Details

- Uses Flow Matching instead of traditional DDPM noise scheduling
- 3D RoPE (Rotary Position Encoding) for spatial-temporal sequences
- Causal convolutions in VAE for temporal consistency
- Adaptive layer normalization with time-step conditioning
- Support for unlimited-length video processing via streaming VAE

When working with model loading, inference, or architectural changes, refer to the specific task implementation files and the modular component structure in `wan/modules/`.