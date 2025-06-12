"""
Configuration utilities for FlowEdit-Wan.
"""

import yaml
import argparse
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file {config_path}: {e}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Output path for YAML file
    """
    # Create directory if needed
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save config to {config_path}: {e}")


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge configuration file with command line arguments.
    Command line arguments take precedence.
    
    Args:
        config: Configuration dictionary from file
        args: Command line arguments
        
    Returns:
        Updated arguments namespace
    """
    # Convert args to dict
    args_dict = vars(args).copy()
    
    # Only update args with config values if arg wasn't explicitly set
    for key, value in config.items():
        if key in args_dict:
            # Check if argument was set to default (indicating not explicitly provided)
            # This is a simplified check - in practice, you might want more sophisticated logic
            if args_dict[key] is None or (hasattr(args, '_defaults') and 
                                        key in args._defaults and 
                                        args_dict[key] == args._defaults[key]):
                args_dict[key] = value
        else:
            # Add new config values not in args
            args_dict[key] = value
    
    # Convert back to namespace
    return argparse.Namespace(**args_dict)


def create_example_config() -> Dict[str, Any]:
    """
    Create an example configuration dictionary.
    
    Returns:
        Example configuration
    """
    return {
        'input_video': 'examples/input.mp4',
        'output': 'output/edited_video.mp4',
        'model_path': './Wan2.1-I2V-14B-720P',
        'model_type': 'i2v-14B',
        'source_prompt': 'A cat sitting on a chair',
        'target_prompt': 'A dog sitting on a chair',
        'negative_prompt': 'blurry, low quality, distorted',
        'steps': 30,
        'skip_steps': 8,
        'drift_steps': 4,
        'n_avg': 1,
        'source_guidance_scale': 6.0,
        'target_guidance_scale': 12.0,
        'drift_guidance_scale': 6.0,
        'flow_shift': 6.0,
        'drift_flow_shift': 3.0,
        'force_offload': True,
        'use_cpu_text_encoder': False,
        'low_vram_mode': False,
        'seed': 42,
        'device': 'cuda',
        'max_frames': None,
        'frame_rate': 16
    }


def create_batch_config() -> Dict[str, Any]:
    """
    Create an example batch processing configuration.
    
    Returns:
        Batch configuration
    """
    return {
        'model_path': './Wan2.1-I2V-14B-720P',
        'model_type': 'i2v-14B',
        'output_dir': 'output/batch_results',
        'force_offload': True,
        'use_cpu_text_encoder': False,
        'low_vram_mode': False,
        'device': 'cuda',
        'frame_rate': 16,
        
        # Default parameters for all edits
        'defaults': {
            'steps': 30,
            'skip_steps': 8,
            'drift_steps': 4,
            'n_avg': 1,
            'source_guidance_scale': 6.0,
            'target_guidance_scale': 12.0,
            'drift_guidance_scale': 6.0,
            'flow_shift': 6.0,
            'drift_flow_shift': 3.0,
            'seed': 42
        },
        
        # List of editing tasks
        'edits': [
            {
                'input_video': 'examples/cat_video.mp4',
                'output': 'cat_to_dog.mp4',
                'source_prompt': 'A fluffy orange cat playing with a ball',
                'target_prompt': 'A fluffy orange dog playing with a ball',
                'skip_steps': 12,
                'target_guidance_scale': 15.0
            },
            {
                'input_video': 'examples/sunny_garden.mp4',
                'output': 'garden_snow.mp4',
                'source_prompt': 'A garden scene on a sunny day',
                'target_prompt': 'A garden scene during a snowstorm',
                'skip_steps': 6,
                'target_guidance_scale': 10.0
            },
            {
                'input_video': 'examples/person_walking.mp4',
                'output': 'robot_walking.mp4',
                'source_prompt': 'A person walking down a street',
                'target_prompt': 'A robot walking down a street',
                'skip_steps': 15,
                'target_guidance_scale': 18.0
            }
        ]
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = ['input_video', 'source_prompt', 'target_prompt', 'model_path']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' missing from config")
    
    # Validate numeric ranges
    numeric_validations = {
        'steps': (1, 1000),
        'skip_steps': (0, None),
        'drift_steps': (0, None),
        'n_avg': (1, 10),
        'source_guidance_scale': (0.0, 50.0),
        'target_guidance_scale': (0.0, 50.0),
        'drift_guidance_scale': (0.0, 50.0),
        'flow_shift': (0.1, 30.0),
        'drift_flow_shift': (0.1, 30.0),
        'seed': (0, 2**32 - 1),
        'frame_rate': (1, 120)
    }
    
    for field, (min_val, max_val) in numeric_validations.items():
        if field in config:
            value = config[field]
            if not isinstance(value, (int, float)):
                raise ValueError(f"Field '{field}' must be numeric, got {type(value)}")
            if min_val is not None and value < min_val:
                raise ValueError(f"Field '{field}' must be >= {min_val}, got {value}")
            if max_val is not None and value > max_val:
                raise ValueError(f"Field '{field}' must be <= {max_val}, got {value}")
    
    # Validate step relationships
    if 'steps' in config and 'skip_steps' in config:
        if config['skip_steps'] >= config['steps']:
            raise ValueError(f"skip_steps ({config['skip_steps']}) must be less than steps ({config['steps']})")
    
    if 'steps' in config and 'drift_steps' in config:
        if config['drift_steps'] >= config['steps']:
            raise ValueError(f"drift_steps ({config['drift_steps']}) must be less than steps ({config['steps']})")
    
    return True


def print_config(config: Dict[str, Any], title: str = "Configuration") -> None:
    """
    Pretty print configuration.
    
    Args:
        config: Configuration dictionary
        title: Title for the printout
    """
    print(f"\n{title}:")
    print("=" * len(title))
    
    def print_dict(d: Dict, indent: int = 0):
        for key, value in d.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_dict(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key}:")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        print(f"{prefix}  [{i}]:")
                        print_dict(item, indent + 2)
                    else:
                        print(f"{prefix}  [{i}]: {item}")
            else:
                print(f"{prefix}{key}: {value}")
    
    print_dict(config)
    print()