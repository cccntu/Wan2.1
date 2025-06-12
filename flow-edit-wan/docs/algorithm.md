# FlowEdit Algorithm Explanation

This document explains how the FlowEdit algorithm works and how it's adapted for Wan 2.1 models.

## Overview

FlowEdit is an **inversion-free** method for editing videos using pre-trained Flow Matching models. Unlike traditional methods that require inverting the input video through the diffusion process, FlowEdit directly manipulates the generation trajectory.

## Key Concepts

### 1. Flow Matching vs. Diffusion

**Traditional Diffusion Models**:
- Use noise addition: `x_t = √(α_t) * x_0 + √(1-α_t) * ε`
- Complex noise schedules with βt parameters

**Flow Matching (used by Wan 2.1)**:
- Simple linear interpolation: `x_t = (1-t) * x_0 + t * ε`
- Time parameter t goes from 1 (pure noise) to 0 (clean data)
- Predicts velocity field instead of noise

### 2. FlowEdit Core Idea

Instead of inverting the input video, FlowEdit:

1. **Starts with source video latents** as the initial state
2. **Computes velocity differences** between source and target prompts
3. **Evolves the trajectory** using differential guidance
4. **Applies refinement steps** at the end for quality

## Algorithm Steps

### Phase 1: Setup (Steps 0 to skip_steps)

```
Skip initial steps entirely
- Reduces the "budget" for changes
- Higher skip_steps = more conservative editing
```

### Phase 2: Differential Editing (Steps skip_steps to steps-drift_steps)

For each timestep t:

1. **Sample noise**: `ε ~ N(0,I)`

2. **Create noisy source**: `z_t^src = (1-t) * x_src + t * ε`

3. **Create noisy target**: `z_t^tgt = x_edit + z_t^src - x_src`
   - This maintains the "edit displacement" while adding appropriate noise

4. **Predict velocities**:
   - `v_t^src = Model(z_t^src, t, prompt_src)`
   - `v_t^tgt = Model(z_t^tgt, t, prompt_tgt)`

5. **Compute velocity difference**: `Δv = v_t^tgt - v_t^src`

6. **Update trajectory**: `x_edit = x_edit + (t_prev - t) * Δv`

### Phase 3: Drift/Refinement (Final drift_steps)

Switch to regular sampling with target prompt only:

```
x_edit = x_edit + (t_prev - t) * Model(x_edit, t, prompt_tgt)
```

This removes artifacts and improves quality.

## Mathematical Foundation

### Flow Matching ODE

The continuous-time flow is governed by:
```
dx/dt = v_θ(x_t, t)
```

Where `v_θ` is the learned velocity field.

### FlowEdit Modification

Instead of solving the standard ODE, FlowEdit uses:
```
dx_edit/dt = v_θ(x_edit + x_t^src - x_src, t, prompt_tgt) - v_θ(x_t^src, t, prompt_src)
```

This creates a "differential flow" that moves from source to target semantics.

## Wan 2.1 Specific Adaptations

### 1. Timestep Scaling
```python
# Wan uses millisecond timesteps
timestep_scaled = timestep * 1000.0
```

### 2. Guidance Scaling
```python
# Guidance values are also scaled
guidance = torch.tensor([guidance_scale]) * 1000.0
```

### 3. Text Conditioning
```python
# Wan uses dual text encoders
transformer(
    latents,
    timestep=timestep_scaled,
    text_states=prompt_embeds,      # Primary text features
    text_states_2=prompt_embeds_2,  # Secondary text features
    guidance=guidance
)
```

### 4. Rotary Position Encoding
```python
# 3D RoPE for spatial-temporal modeling
freqs_cos, freqs_sin = get_rotary_pos_embed(num_frames, height, width)
```

## Parameter Effects

### skip_steps
- **Low (0-4)**: Maximum editing power, may lose source structure
- **Medium (5-10)**: Balanced editing, preserves most structure  
- **High (15+)**: Conservative editing, minimal changes

### drift_steps
- **0**: Pure differential editing, may be blurry
- **2-4**: Good balance of editing and quality
- **8+**: Heavy refinement, may drift from target

### target_guidance_scale
- **Low (6-8)**: Subtle changes
- **Medium (10-14)**: Standard editing strength
- **High (15+)**: Strong changes, may introduce artifacts

### flow_shift
- **Low (1-3)**: Gentle trajectory curves
- **Medium (4-8)**: Standard flow dynamics
- **High (8+)**: Aggressive early changes

## Implementation Details

### Memory Optimization

1. **Mixed Precision**: Use bfloat16 for computation, float32 for integration
```python
x_edit = x_edit.to(torch.float32)
x_edit = x_edit + (t_prev - t) * v_delta
x_edit = x_edit.to(torch.bfloat16)
```

2. **Gradient Checkpointing**: Reduce memory during forward pass

3. **Model Offloading**: Move model components between GPU/CPU

### Numerical Stability

- Use Euler integration for ODE solving
- Clamp extreme values during trajectory evolution
- Careful handling of timestep boundaries

## Comparison with Other Methods

| Method | Inversion Required | Edit Quality | Speed | Memory |
|--------|-------------------|--------------|--------|--------|
| DDIM Inversion | Yes | High | Slow | High |
| Null-text Inversion | Yes | High | Very Slow | High |
| FlowEdit | No | High | Fast | Medium |

## Limitations

1. **Flow Matching Specific**: Only works with flow-based models
2. **Prompt Dependency**: Requires good source/target prompt pairs
3. **Semantic Constraints**: Limited by model's semantic understanding
4. **Balance Tuning**: Requires parameter tuning for optimal results

## Future Directions

- **Automatic Parameter Selection**: Learn optimal parameters per edit type
- **Multi-scale Editing**: Apply different strengths at different resolutions
- **Temporal Consistency**: Better preservation of temporal coherence
- **Interactive Editing**: Real-time parameter adjustment