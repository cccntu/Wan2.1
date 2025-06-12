"""
Model loading and management utilities for FlowEdit-Wan.
"""

import os
import sys
import torch
import gc
from pathlib import Path
from typing import Optional, Union

# Add parent directory to path for wan imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from wan.image2video import WanI2V
    from wan.text2video import WanT2V
    from wan.first_last_frame2video import WanFLF2V
    from wan.vace import WanVace
except ImportError as e:
    print(f"Warning: Could not import Wan modules: {e}")
    print("Make sure you're running from the correct directory and Wan is installed")


def get_available_models():
    """Get list of available model types."""
    return ["i2v-14B", "i2v-1.3B", "t2v-14B", "t2v-1.3B", "flf2v-14B", "vace-14B", "vace-1.3B"]


def determine_model_type(model_path: str) -> str:
    """
    Automatically determine model type from path.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Inferred model type
    """
    path_lower = model_path.lower()
    
    if 'i2v' in path_lower:
        if '1.3b' in path_lower or '1_3b' in path_lower:
            return 'i2v-1.3B'
        else:
            return 'i2v-14B'
    elif 'flf2v' in path_lower:
        return 'flf2v-14B'
    elif 'vace' in path_lower:
        if '1.3b' in path_lower or '1_3b' in path_lower:
            return 'vace-1.3B'
        else:
            return 'vace-14B'
    elif 't2v' in path_lower:
        if '1.3b' in path_lower or '1_3b' in path_lower:
            return 't2v-1.3B'
        else:
            return 't2v-14B'
    else:
        # Default fallback
        return 'i2v-14B'


def load_wan_model(model_path: str,
                   model_type: Optional[str] = None,
                   device: Union[str, torch.device] = "cuda",
                   force_offload: bool = False,
                   use_cpu_text_encoder: bool = False,
                   low_vram_mode: bool = False,
                   **kwargs):
    """
    Load Wan 2.1 model pipeline.
    
    Args:
        model_path: Path to model directory
        model_type: Model type (auto-detected if None)
        device: Device to load model on
        force_offload: Enable model offloading
        use_cpu_text_encoder: Keep text encoder on CPU
        low_vram_mode: Enable low VRAM optimizations
        **kwargs: Additional arguments for model loading
        
    Returns:
        Loaded pipeline
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Auto-determine model type if not provided
    if model_type is None:
        model_type = determine_model_type(model_path)
        print(f"Auto-detected model type: {model_type}")
    
    # Validate model type
    if model_type not in get_available_models():
        raise ValueError(f"Unsupported model type: {model_type}. Available: {get_available_models()}")
    
    # Convert device
    if isinstance(device, str):
        device = torch.device(device)
    
    # Set up loading arguments
    load_args = {
        'ckpt_dir': model_path,
        'device': device,
        **kwargs
    }
    
    # Memory optimization settings
    if force_offload:
        load_args['offload_model'] = True
    
    if use_cpu_text_encoder:
        load_args['t5_cpu'] = True
    
    if low_vram_mode:
        load_args['offload_model'] = True
        load_args['t5_cpu'] = True
        # Additional low VRAM settings could go here
    
    try:
        # Load appropriate pipeline based on model type
        if model_type.startswith('i2v'):
            print(f"Loading Image-to-Video model from {model_path}")
            pipeline = WanI2V.from_pretrained(**load_args)
            
        elif model_type.startswith('t2v'):
            print(f"Loading Text-to-Video model from {model_path}")
            pipeline = WanT2V.from_pretrained(**load_args)
            
        elif model_type.startswith('flf2v'):
            print(f"Loading First-Last-Frame-to-Video model from {model_path}")
            pipeline = WanFLF2V.from_pretrained(**load_args)
            
        elif model_type.startswith('vace'):
            print(f"Loading VACE model from {model_path}")
            pipeline = WanVace.from_pretrained(**load_args)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Apply memory optimizations
        if low_vram_mode:
            # Enable memory efficient attention if available
            if hasattr(pipeline.transformer, 'enable_memory_efficient_attention'):
                pipeline.transformer.enable_memory_efficient_attention()
            
            # Enable gradient checkpointing if available
            if hasattr(pipeline.transformer, 'enable_gradient_checkpointing'):
                pipeline.transformer.enable_gradient_checkpointing()
        
        # Clean up memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f"Successfully loaded {model_type} model")
        return pipeline
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_type} from {model_path}: {str(e)}")


def get_model_info(pipeline) -> dict:
    """
    Get information about loaded model.
    
    Args:
        pipeline: Loaded Wan pipeline
        
    Returns:
        Dictionary with model information
    """
    info = {
        'type': type(pipeline).__name__,
        'device': str(next(pipeline.transformer.parameters()).device),
        'dtype': str(next(pipeline.transformer.parameters()).dtype),
    }
    
    # Get transformer config if available
    if hasattr(pipeline.transformer, 'config'):
        config = pipeline.transformer.config
        info.update({
            'in_channels': getattr(config, 'in_channels', None),
            'sample_size': getattr(config, 'sample_size', None),
            'attention_head_dim': getattr(config, 'attention_head_dim', None),
            'num_attention_heads': getattr(config, 'num_attention_heads', None),
            'num_layers': getattr(config, 'num_layers', None),
        })
    
    # Get VAE info if available
    if hasattr(pipeline, 'vae') and hasattr(pipeline.vae, 'config'):
        vae_config = pipeline.vae.config
        info.update({
            'vae_in_channels': getattr(vae_config, 'in_channels', None),
            'vae_out_channels': getattr(vae_config, 'out_channels', None),
            'vae_scaling_factor': getattr(vae_config, 'scaling_factor', None),
            'vae_shift_factor': getattr(vae_config, 'shift_factor', None),
        })
    
    return info


def optimize_model_memory(pipeline, 
                         enable_attention_slicing: bool = True,
                         enable_memory_efficient_attention: bool = True,
                         enable_gradient_checkpointing: bool = True):
    """
    Apply memory optimizations to model.
    
    Args:
        pipeline: Loaded pipeline
        enable_attention_slicing: Enable attention slicing
        enable_memory_efficient_attention: Enable memory efficient attention
        enable_gradient_checkpointing: Enable gradient checkpointing
    """
    print("Applying memory optimizations...")
    
    try:
        if enable_attention_slicing and hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
            print("✓ Enabled attention slicing")
        
        if enable_memory_efficient_attention:
            if hasattr(pipeline, 'enable_memory_efficient_attention'):
                pipeline.enable_memory_efficient_attention()
                print("✓ Enabled memory efficient attention")
            elif hasattr(pipeline.transformer, 'enable_memory_efficient_attention'):
                pipeline.transformer.enable_memory_efficient_attention()
                print("✓ Enabled memory efficient attention on transformer")
        
        if enable_gradient_checkpointing:
            if hasattr(pipeline, 'enable_gradient_checkpointing'):
                pipeline.enable_gradient_checkpointing()
                print("✓ Enabled gradient checkpointing")
            elif hasattr(pipeline.transformer, 'enable_gradient_checkpointing'):
                pipeline.transformer.enable_gradient_checkpointing()
                print("✓ Enabled gradient checkpointing on transformer")
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("✓ Cleaned up GPU memory")
    
    except Exception as e:
        print(f"Warning: Some memory optimizations failed: {e}")


def free_model_memory(pipeline):
    """
    Free model memory and move to CPU.
    
    Args:
        pipeline: Pipeline to free
    """
    try:
        # Move model to CPU
        if hasattr(pipeline, 'to'):
            pipeline.to('cpu')
        elif hasattr(pipeline, 'transformer'):
            pipeline.transformer.to('cpu')
            if hasattr(pipeline, 'vae'):
                pipeline.vae.to('cpu')
            if hasattr(pipeline, 'text_encoder'):
                pipeline.text_encoder.to('cpu')
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("✓ Model memory freed")
        
    except Exception as e:
        print(f"Warning: Failed to free model memory: {e}")