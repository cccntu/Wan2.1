# FlowEdit-Wan: Video-to-Video Editing with Wan 2.1

This directory contains a self-contained implementation of FlowEdit for video-to-video editing using Wan 2.1 models. FlowEdit is an inversion-free text-based editing method that works with Flow Matching models.

## Features

- **Inversion-free editing**: No need for complex inversion procedures
- **Text-based control**: Edit videos using natural language prompts
- **Multiple model support**: Works with Wan 2.1 14B and 1.3B models
- **Memory efficient**: Optimized for consumer GPUs
- **Flexible parameters**: Fine-tune editing strength and quality

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure you have Wan 2.1 models downloaded
# See main repository README for model download instructions
```

## Quick Start

### Basic Video Editing

```bash
python flow_edit_wan.py \
    --input_video "input.mp4" \
    --source_prompt "A cat sitting on a chair" \
    --target_prompt "A dog sitting on a chair" \
    --model_path "./Wan2.1-I2V-14B-720P" \
    --output "edited_video.mp4"
```

### Batch Editing

```bash
python batch_edit.py --config_file examples/batch_config.yaml
```

### Using Configuration Files

```bash
python flow_edit_wan.py --config examples/example_config.yaml
```

## Parameters Guide

### Core Parameters

- `--input_video`: Path to input video file
- `--source_prompt`: Description of the input video content
- `--target_prompt`: Description of desired output
- `--model_path`: Path to Wan 2.1 model directory

### FlowEdit Parameters

- `--steps`: Total denoising steps (default: 30)
- `--skip_steps`: Initial steps to skip - controls edit strength (default: 8)
- `--drift_steps`: Final refinement steps (default: 4)
- `--n_avg`: Number of velocity predictions to average (default: 1)

### Guidance Parameters

- `--source_guidance_scale`: Source prompt adherence (default: 6.0)
- `--target_guidance_scale`: Target prompt adherence (default: 12.0)
- `--drift_guidance_scale`: Guidance for drift steps (default: 6.0)

### Flow Parameters

- `--flow_shift`: Main editing flow shift (default: 6.0)
- `--drift_flow_shift`: Drift steps flow shift (default: 3.0)

## Parameter Tuning Guide

### For Subtle Edits (colors, clothing, expressions)
```bash
--skip_steps 4 \
--drift_steps 2 \
--target_guidance_scale 8.0
```

### For Moderate Edits (style changes, object properties)
```bash
--skip_steps 8 \
--drift_steps 4 \
--target_guidance_scale 12.0
```

### For Major Edits (object replacement, scene changes)
```bash
--skip_steps 15 \
--drift_steps 6 \
--target_guidance_scale 18.0
```

## Memory Optimization

For GPUs with limited VRAM:

```bash
python flow_edit_wan.py \
    --input_video "input.mp4" \
    --source_prompt "prompt" \
    --target_prompt "prompt" \
    --model_path "./model" \
    --force_offload \
    --use_cpu_text_encoder \
    --low_vram_mode
```

## Examples

### Example 1: Object Replacement
```bash
python flow_edit_wan.py \
    --input_video "examples/cat_video.mp4" \
    --source_prompt "A fluffy orange cat playing with a ball of yarn" \
    --target_prompt "A fluffy orange dog playing with a ball of yarn" \
    --model_path "./Wan2.1-I2V-14B-720P" \
    --skip_steps 12 \
    --target_guidance_scale 15.0
```

### Example 2: Style Transfer
```bash
python flow_edit_wan.py \
    --input_video "examples/person_walking.mp4" \
    --source_prompt "A person walking in a modern city street" \
    --target_prompt "A person walking in a medieval fantasy town" \
    --model_path "./Wan2.1-I2V-14B-720P" \
    --skip_steps 10 \
    --target_guidance_scale 14.0
```

### Example 3: Weather/Environment Change
```bash
python flow_edit_wan.py \
    --input_video "examples/sunny_scene.mp4" \
    --source_prompt "A garden scene on a sunny day" \
    --target_prompt "A garden scene during a snowstorm" \
    --model_path "./Wan2.1-I2V-14B-720P" \
    --skip_steps 6 \
    --target_guidance_scale 10.0
```

## File Structure

```
flow-edit-wan/
├── README.md
├── requirements.txt
├── flow_edit_wan.py          # Main editing script
├── batch_edit.py             # Batch processing script
├── wan_flow_edit.py          # Core FlowEdit implementation
├── utils/
│   ├── video_io.py           # Video loading/saving utilities
│   ├── model_utils.py        # Model loading and management
│   └── config_utils.py       # Configuration handling
├── examples/
│   ├── example_config.yaml   # Example configuration
│   ├── batch_config.yaml     # Batch processing config
│   └── sample_videos/        # Sample input videos
└── docs/
    ├── algorithm.md          # FlowEdit algorithm explanation
    └── troubleshooting.md    # Common issues and solutions
```

## Technical Details

FlowEdit works by:

1. **Encoding**: Convert input video to latent space using Wan VAE
2. **Trajectory Initialization**: Start with source video latents
3. **Differential Sampling**: Compute velocity differences between source and target prompts
4. **ODE Integration**: Evolve the latent trajectory toward the target
5. **Refinement**: Apply final drift steps for quality improvement
6. **Decoding**: Convert edited latents back to video

## Troubleshooting

### Common Issues

**Out of Memory Error**:
```bash
# Use memory optimization flags
--force_offload --use_cpu_text_encoder --low_vram_mode
```

**Poor Edit Quality**:
```bash
# Increase guidance scale and adjust skip steps
--target_guidance_scale 15.0 --skip_steps 10
```

**Too Much Change**:
```bash
# Reduce skip steps and guidance
--skip_steps 4 --target_guidance_scale 8.0
```

### GPU Memory Requirements

- **Wan 2.1 14B**: 16GB+ VRAM (24GB recommended)
- **Wan 2.1 1.3B**: 8GB+ VRAM (12GB recommended)
- **With optimizations**: Can run 14B on 12GB VRAM

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project follows the same license as the main Wan 2.1 repository (Apache 2.0).

## Citation

If you use this implementation in your research, please cite both FlowEdit and Wan 2.1:

```bibtex
@article{kulikov2024flowedit,
    title = {FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models},
    author = {Kulikov, Vladimir and Kleiner, Matan and Huberman-Spiegelglas, Inbar and Michaeli, Tomer},
    journal = {arXiv preprint arXiv:2412.08629},
    year = {2024}
}

@article{wan2025,
    title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
    author={Team Wan and [full author list]},
    journal = {arXiv preprint arXiv:2503.20314},
    year={2025}
}
```

## Acknowledgments

- Original FlowEdit implementation by Kulikov et al.
- Wan 2.1 model by Alibaba Wan Team
- ComfyUI-HunyuanLoom integration by logtd