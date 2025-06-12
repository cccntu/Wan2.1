#!/usr/bin/env python3
"""
Batch video editing script for FlowEdit-Wan.
Processes multiple videos based on a configuration file.
"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

import torch

# Add parent directory to path for wan imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wan_flow_edit import FlowEditWan
from utils.video_io import load_video, save_video
from utils.model_utils import load_wan_model, free_model_memory
from utils.config_utils import load_config, validate_config, print_config


def parse_args():
    parser = argparse.ArgumentParser(description="FlowEdit-Wan Batch Processing")
    
    parser.add_argument("--config_file", type=str, required=True,
                       help="Path to batch configuration YAML file")
    parser.add_argument("--resume", action="store_true",
                       help="Resume processing, skip existing outputs")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be processed without actually running")
    
    return parser.parse_args()


def process_single_edit(flow_editor, edit_config, global_config, output_dir):
    """
    Process a single editing task.
    
    Args:
        flow_editor: FlowEditWan instance
        edit_config: Individual edit configuration
        global_config: Global configuration
        output_dir: Output directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Merge edit config with defaults
        config = global_config.get('defaults', {}).copy()
        config.update(edit_config)
        
        # Prepare paths
        input_video = config['input_video']
        output_name = config['output']
        output_path = os.path.join(output_dir, output_name)
        
        print(f"\nProcessing: {input_video} -> {output_name}")
        print(f"Source: {config['source_prompt']}")
        print(f"Target: {config['target_prompt']}")
        
        # Load video
        video_frames = load_video(
            input_video,
            max_frames=global_config.get('max_frames'),
            target_fps=global_config.get('frame_rate', 16)
        )
        
        # Edit video
        edited_frames = flow_editor.edit_video(
            video_frames=video_frames,
            source_prompt=config['source_prompt'],
            target_prompt=config['target_prompt'],
            negative_prompt=config.get('negative_prompt', ''),
            steps=config.get('steps', 30),
            skip_steps=config.get('skip_steps', 8),
            drift_steps=config.get('drift_steps', 4),
            n_avg=config.get('n_avg', 1),
            source_guidance_scale=config.get('source_guidance_scale', 6.0),
            target_guidance_scale=config.get('target_guidance_scale', 12.0),
            drift_guidance_scale=config.get('drift_guidance_scale', 6.0),
            flow_shift=config.get('flow_shift', 6.0),
            drift_flow_shift=config.get('drift_flow_shift', 3.0),
            seed=config.get('seed', 42)
        )
        
        # Save result
        save_video(edited_frames, output_path, fps=global_config.get('frame_rate', 16))
        
        print(f"✅ Successfully completed: {output_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to process {edit_config.get('input_video', 'unknown')}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config_file)
        print_config(config, "Batch Processing Configuration")
        
        # Validate configuration
        validate_config(config)
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    # Extract components
    edits = config.get('edits', [])
    if not edits:
        print("❌ No edits specified in configuration")
        sys.exit(1)
    
    output_dir = config.get('output_dir', 'output/batch_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Dry run mode
    if args.dry_run:
        print(f"\n🔍 Dry run mode - showing what would be processed:")
        print(f"Output directory: {output_dir}")
        print(f"Total edits: {len(edits)}")
        
        for i, edit in enumerate(edits, 1):
            print(f"\n[{i}] {edit.get('input_video', 'unknown')} -> {edit.get('output', 'unknown')}")
            print(f"    Source: {edit.get('source_prompt', 'unknown')}")
            print(f"    Target: {edit.get('target_prompt', 'unknown')}")
        
        return
    
    # Filter edits if resuming
    if args.resume:
        original_count = len(edits)
        edits = [edit for edit in edits if not os.path.exists(os.path.join(output_dir, edit['output']))]
        skipped_count = original_count - len(edits)
        if skipped_count > 0:
            print(f"📋 Resuming: skipping {skipped_count} existing outputs")
    
    if not edits:
        print("✅ All edits already completed!")
        return
    
    # Setup device
    device = torch.device(config.get('device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    
    print(f"\n🚀 Starting batch processing on {device}")
    print(f"Processing {len(edits)} edits...")
    
    # Load model
    try:
        print("\n📦 Loading Wan 2.1 model...")
        pipeline = load_wan_model(
            config['model_path'],
            model_type=config.get('model_type'),
            device=device,
            force_offload=config.get('force_offload', False),
            use_cpu_text_encoder=config.get('use_cpu_text_encoder', False),
            low_vram_mode=config.get('low_vram_mode', False)
        )
        
        # Initialize FlowEdit
        flow_editor = FlowEditWan(pipeline, device=device)
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Process edits
    successful = 0
    failed = 0
    
    try:
        with tqdm(edits, desc="Processing edits") as pbar:
            for i, edit in enumerate(pbar):
                pbar.set_description(f"Processing {edit.get('output', f'edit_{i+1}')}")
                
                success = process_single_edit(flow_editor, edit, config, output_dir)
                
                if success:
                    successful += 1
                else:
                    failed += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Success': successful,
                    'Failed': failed,
                    'Rate': f"{successful}/{successful+failed}"
                })
                
                # Clean up GPU memory between edits
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    
    finally:
        # Clean up
        try:
            free_model_memory(pipeline)
        except:
            pass
    
    # Summary
    print(f"\n📊 Batch processing completed!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {output_dir}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()