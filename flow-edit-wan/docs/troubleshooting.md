# FlowEdit-Wan Troubleshooting Guide

This guide helps you resolve common issues when using FlowEdit-Wan for video editing.

## Common Issues and Solutions

### 1. Memory Issues

#### Error: "CUDA out of memory"

**Symptoms**: 
- Process crashes with CUDA OOM error
- GPU memory usage spikes during loading

**Solutions**:

```bash
# Enable all memory optimizations
python flow_edit_wan.py \
    --input_video input.mp4 \
    --source_prompt "..." \
    --target_prompt "..." \
    --model_path ./model \
    --force_offload \
    --use_cpu_text_encoder \
    --low_vram_mode
```

**Advanced memory settings**:
- Reduce batch size (if applicable)
- Use smaller model (1.3B instead of 14B)
- Process shorter video segments
- Lower video resolution

#### Error: "RuntimeError: Unable to allocate memory"

**Symptoms**:
- System runs out of RAM
- Slow performance with high memory usage

**Solutions**:
- Close other applications
- Use `--max_frames` to limit video length
- Process video in chunks
- Use swap file if necessary

### 2. Model Loading Issues

#### Error: "Model path not found"

**Symptoms**:
- FileNotFoundError when loading model
- Invalid model directory structure

**Solutions**:

```bash
# Check if model directory exists and contains required files
ls -la ./Wan2.1-I2V-14B-720P/

# Required files should include:
# - transformer/
# - vae/
# - text_encoder/
# - scheduler/
# - model_index.json
```

Download models using:
```bash
# HuggingFace
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./Wan2.1-I2V-14B-720P

# Or ModelScope
modelscope download Wan-AI/Wan2.1-I2V-14B-720P --local_dir ./Wan2.1-I2V-14B-720P
```

#### Error: "Unsupported model type"

**Symptoms**:
- ValueError about unknown model type
- Auto-detection picks wrong model

**Solutions**:
```bash
# Explicitly specify model type
python flow_edit_wan.py \
    --model_type i2v-14B \
    --model_path ./your_model_path \
    # ... other args
```

Available model types:
- `i2v-14B`, `i2v-1.3B`
- `t2v-14B`, `t2v-1.3B`  
- `flf2v-14B`
- `vace-14B`, `vace-1.3B`

### 3. Video Input/Output Issues

#### Error: "Video file not found" or "Failed to load video"

**Symptoms**:
- FileNotFoundError or video loading fails
- Corrupt or unsupported video format

**Solutions**:

```bash
# Check file exists and is readable
ls -la input_video.mp4
file input_video.mp4

# Convert to supported format using ffmpeg
ffmpeg -i input_video.mov -c:v libx264 -c:a aac input_video.mp4

# Supported formats: MP4, AVI, MOV, MKV
```

#### Error: "Failed to save video"

**Symptoms**:
- Video processing completes but saving fails
- Output directory permission issues

**Solutions**:
```bash
# Create output directory with proper permissions
mkdir -p output/
chmod 755 output/

# Check disk space
df -h

# Try different output format
python flow_edit_wan.py \
    --output output/result.mp4 \
    # ... other args
```

### 4. Poor Editing Quality

#### Problem: Minimal or no changes visible

**Possible Causes**:
- `skip_steps` too high
- `target_guidance_scale` too low
- Poor prompt descriptions

**Solutions**:
```bash
# Reduce skip_steps for more aggressive editing
--skip_steps 4

# Increase target guidance scale
--target_guidance_scale 15.0

# Improve prompt specificity
--source_prompt "A fluffy orange tabby cat sitting on a wooden chair"
--target_prompt "A fluffy orange golden retriever sitting on a wooden chair"
```

#### Problem: Too much change, losing source structure

**Possible Causes**:
- `skip_steps` too low  
- `target_guidance_scale` too high
- Insufficient drift steps

**Solutions**:
```bash
# Increase skip_steps for more conservative editing
--skip_steps 12

# Reduce target guidance scale
--target_guidance_scale 8.0

# Add more drift steps for refinement
--drift_steps 6
```

#### Problem: Blurry or low-quality output

**Possible Causes**:
- Too few drift steps
- Inappropriate flow shift values
- Resolution mismatch

**Solutions**:
```bash
# Increase drift steps
--drift_steps 6

# Adjust flow shift parameters
--flow_shift 6.0
--drift_flow_shift 3.0

# Check input video resolution and model compatibility
```

### 5. Performance Issues

#### Problem: Very slow processing

**Possible Causes**:
- CPU-only execution
- Inefficient memory usage
- Large video files

**Solutions**:
```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU utilization
nvidia-smi

# Optimize settings
--force_offload false  # Keep model on GPU if possible
--n_avg 1             # Don't oversample velocity predictions

# Process smaller chunks
--max_frames 81       # Limit to 81 frames
```

#### Problem: Process hangs or freezes

**Possible Causes**:
- Memory issues
- Model loading problems
- CUDA context issues

**Solutions**:
```bash
# Reset CUDA context
python -c "import torch; torch.cuda.empty_cache()"

# Try CPU-only mode
--device cpu

# Check system resources
top
nvidia-smi
```

### 6. Configuration Issues

#### Problem: YAML configuration not loading

**Symptoms**:
- YAML parsing errors
- Configuration validation failures

**Solutions**:
```yaml
# Check YAML syntax
# Use proper indentation (spaces, not tabs)
# Quote string values if they contain special characters

# Example valid config:
input_video: "path/to/video.mp4"
source_prompt: "A cat on a chair"
target_prompt: "A dog on a chair"
steps: 30
```

#### Problem: Parameter validation errors

**Symptoms**:
- ValueError about parameter ranges
- Inconsistent step relationships

**Solutions**:
```yaml
# Ensure skip_steps < steps
steps: 30
skip_steps: 8    # Must be < 30

# Ensure drift_steps < steps  
drift_steps: 4   # Must be < 30

# Valid guidance scale ranges
source_guidance_scale: 6.0   # 0.0 to 50.0
target_guidance_scale: 12.0  # 0.0 to 50.0
```

## Debugging Tips

### 1. Enable Verbose Logging

```python
# Add to your script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Test with Minimal Configuration

```bash
# Start with basic settings
python flow_edit_wan.py \
    --input_video short_test.mp4 \
    --source_prompt "simple description" \
    --target_prompt "simple change" \
    --model_path ./model \
    --steps 10 \
    --max_frames 16
```

### 3. Monitor System Resources

```bash
# Terminal 1: Run your command
python flow_edit_wan.py ...

# Terminal 2: Monitor resources
watch -n 1 'nvidia-smi; echo "---"; free -h'
```

### 4. Test Model Loading Separately

```python
# Test script: test_model.py
from utils.model_utils import load_wan_model
import torch

try:
    pipeline = load_wan_model("./model_path", device="cuda")
    print("✅ Model loaded successfully")
    print(f"Device: {next(pipeline.transformer.parameters()).device}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
```

## Getting Help

### 1. Collect System Information

```bash
# System info
python -c "
import torch
import sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

### 2. Create Minimal Reproduction Case

```bash
# Create a short test video
ffmpeg -f lavfi -i testsrc=duration=2:size=512x512:rate=16 -pix_fmt yuv420p test_input.mp4

# Test with minimal settings
python flow_edit_wan.py \
    --input_video test_input.mp4 \
    --source_prompt "test pattern" \
    --target_prompt "colorful pattern" \
    --model_path ./model \
    --steps 10 \
    --output test_output.mp4
```

### 3. Check Dependencies

```bash
# Verify all requirements are installed
pip list | grep -E "(torch|diffusers|transformers|opencv|imageio)"

# Update if needed
pip install -r requirements.txt --upgrade
```

### 4. Report Issues

When reporting issues, include:

1. **Full error message and stack trace**
2. **System information** (GPU, CUDA version, Python version)
3. **Command used** (with sensitive paths redacted)
4. **Input video characteristics** (resolution, length, format)
5. **Model type and path**
6. **Configuration file** (if used)

## Performance Optimization

### For Different GPU Memory Sizes

**8GB VRAM (RTX 3070, RTX 4060 Ti)**:
```bash
--model_type i2v-1.3B \
--force_offload \
--use_cpu_text_encoder \
--max_frames 41
```

**12GB VRAM (RTX 3060, RTX 4070)**:
```bash
--model_type i2v-14B \
--force_offload \
--max_frames 81
```

**16GB+ VRAM (RTX 3080, RTX 4080, RTX 4090)**:
```bash
--model_type i2v-14B \
--max_frames 161  # Full length videos
```

**24GB+ VRAM (RTX 3090, RTX 4090, A100)**:
```bash
--model_type i2v-14B \
# No memory restrictions needed
```