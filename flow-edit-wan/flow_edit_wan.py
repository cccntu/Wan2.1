#!/usr/bin/env python3
"""
FlowEdit-Wan: Video-to-Video Editing with Wan 2.1
Main script for editing videos using FlowEdit algorithm with Wan 2.1 models.
"""

import argparse
import os
import sys
import warnings
import yaml
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Add parent directory to path for wan imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wan_flow_edit import FlowEditWan
from utils.video_io import load_video, save_video
from utils.model_utils import load_wan_model
from utils.config_utils import load_config, merge_config_with_args

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="FlowEdit-Wan Video Editing")
    
    # Input/Output
    parser.add_argument("--input_video", type=str, required=True, 
                       help="Path to input video file")
    parser.add_argument("--output", type=str, default="edited_video.mp4",
                       help="Output video path")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file")
    
    # Model settings
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to Wan 2.1 model directory")
    parser.add_argument("--model_type", type=str, default="i2v-14B",
                       choices=["i2v-14B", "i2v-1.3B", "t2v-14B", "t2v-1.3B"],
                       help="Model type to use")
    
    # Prompts
    parser.add_argument("--source_prompt", type=str, required=True,
                       help="Description of the input video content")
    parser.add_argument("--target_prompt", type=str, required=True,
                       help="Description of desired output")
    parser.add_argument("--negative_prompt", type=str, default="",
                       help="Negative prompt to avoid")
    
    # FlowEdit parameters
    parser.add_argument("--steps", type=int, default=30,
                       help="Total denoising steps")
    parser.add_argument("--skip_steps", type=int, default=8,
                       help="Initial steps to skip (controls edit strength)")
    parser.add_argument("--drift_steps", type=int, default=4,
                       help="Final refinement steps")
    parser.add_argument("--n_avg", type=int, default=1,
                       help="Number of velocity predictions to average")
    
    # Guidance parameters
    parser.add_argument("--source_guidance_scale", type=float, default=6.0,
                       help="Source prompt adherence")
    parser.add_argument("--target_guidance_scale", type=float, default=12.0,
                       help="Target prompt adherence")
    parser.add_argument("--drift_guidance_scale", type=float, default=6.0,
                       help="Guidance for drift steps")
    
    # Flow parameters
    parser.add_argument("--flow_shift", type=float, default=6.0,
                       help="Main editing flow shift")
    parser.add_argument("--drift_flow_shift", type=float, default=3.0,
                       help="Drift steps flow shift")
    
    # Memory optimization
    parser.add_argument("--force_offload", action="store_true",
                       help="Force model offloading for memory efficiency")
    parser.add_argument("--use_cpu_text_encoder", action="store_true",
                       help="Use CPU for text encoding")
    parser.add_argument("--low_vram_mode", action="store_true",
                       help="Enable low VRAM optimizations")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum number of frames to process")
    parser.add_argument("--frame_rate", type=int, default=16,
                       help="Output video frame rate")
    
    return parser.parse_args()


def setup_device_and_seed(args):
    """Setup device and random seed."""
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    
    return torch.device(args.device)


def main():
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args)
    
    # Setup device and seed
    device = setup_device_and_seed(args)
    
    print(f"FlowEdit-Wan Video Editing")
    print(f"Input: {args.input_video}")
    print(f"Output: {args.output}")
    print(f"Source: {args.source_prompt}")
    print(f"Target: {args.target_prompt}")
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Parameters: steps={args.steps}, skip={args.skip_steps}, drift={args.drift_steps}")
    print()
    
    try:
        # Load input video
        print("Loading input video...")
        video_frames = load_video(
            args.input_video, 
            max_frames=args.max_frames,
            target_fps=args.frame_rate
        )
        print(f"Loaded video: {video_frames.shape} frames")
        
        # Check for memory optimization needs
        _, _, T, H, W = video_frames.shape
        estimated_memory_gb = (T * H * W * 3 * 4) / (1024**3)  # Rough estimate in GB
        print(f"Estimated video memory usage: {estimated_memory_gb:.2f} GB")
        
        # Reduce resolution if needed for memory
        if args.low_vram_mode or estimated_memory_gb > 4.0:
            max_resolution = 384 if args.low_vram_mode else 512
            video_frames = FlowEditWan.reduce_video_resolution(video_frames, max_resolution)
        
        # Clear any existing GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load Wan model
        print("Loading Wan 2.1 model...")
        pipeline = load_wan_model(
            args.model_path,
            model_type=args.model_type,
            device=device,
            force_offload=args.force_offload,
            use_cpu_text_encoder=args.use_cpu_text_encoder,
            low_vram_mode=args.low_vram_mode
        )
        print("Model loaded successfully")
        
        # Initialize FlowEdit
        flow_editor = FlowEditWan(pipeline, device=device)
        
        # Clear memory before processing
        flow_editor.clear_memory()
        
        # Perform editing
        print("Starting FlowEdit process...")
        edited_frames = flow_editor.edit_video(
            video_frames=video_frames,
            source_prompt=args.source_prompt,
            target_prompt=args.target_prompt,
            negative_prompt=args.negative_prompt,
            steps=args.steps,
            skip_steps=args.skip_steps,
            drift_steps=args.drift_steps,
            n_avg=args.n_avg,
            source_guidance_scale=args.source_guidance_scale,
            target_guidance_scale=args.target_guidance_scale,
            drift_guidance_scale=args.drift_guidance_scale,
            flow_shift=args.flow_shift,
            drift_flow_shift=args.drift_flow_shift,
            seed=args.seed
        )
        
        # Save result
        print("Saving edited video...")
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        save_video(edited_frames, args.output, fps=args.frame_rate)
        
        print(f"✅ Video editing completed successfully!")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error during video editing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()