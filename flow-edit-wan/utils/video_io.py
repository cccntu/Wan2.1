"""
Video input/output utilities for FlowEdit-Wan.
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
import imageio
from typing import Optional, Union, List


def load_video(video_path: str, 
               max_frames: Optional[int] = None,
               target_fps: Optional[int] = None,
               target_size: Optional[tuple] = None) -> torch.Tensor:
    """
    Load video from file and convert to tensor.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load
        target_fps: Target FPS for resampling
        target_size: Target size as (width, height)
        
    Returns:
        Video tensor of shape [1, C, T, H, W] normalized to [-1, 1]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Read video using imageio
    try:
        reader = imageio.get_reader(video_path)
        frames = []
        
        # Get video metadata
        fps = reader.get_meta_data().get('fps', 30)
        
        # Calculate frame sampling
        frame_skip = 1
        if target_fps and fps > target_fps:
            frame_skip = max(1, int(fps / target_fps))
        
        frame_count = 0
        for i, frame in enumerate(reader):
            if i % frame_skip != 0:
                continue
                
            # Convert to PIL Image for consistent processing
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            
            # Resize if needed
            if target_size:
                frame = frame.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            frame_array = np.array(frame)
            if frame_array.ndim == 2:  # Grayscale
                frame_array = np.stack([frame_array] * 3, axis=-1)
            elif frame_array.shape[-1] == 4:  # RGBA
                frame_array = frame_array[..., :3]  # Remove alpha
                
            frames.append(frame_array)
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break
                
        reader.close()
        
    except Exception as e:
        raise RuntimeError(f"Failed to load video {video_path}: {str(e)}")
    
    if not frames:
        raise ValueError(f"No frames could be loaded from {video_path}")
    
    # Convert to tensor
    frames_array = np.stack(frames, axis=0)  # [T, H, W, C]
    frames_tensor = torch.from_numpy(frames_array).float()
    
    # Normalize to [-1, 1]
    frames_tensor = frames_tensor / 127.5 - 1.0
    
    # Rearrange to [1, C, T, H, W]
    frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]
    
    # Ensure tensor is contiguous for better performance
    frames_tensor = frames_tensor.contiguous()
    
    print(f"Loaded video: {len(frames)} frames, shape: {frames_tensor.shape}")
    
    return frames_tensor


def save_video(video_tensor: torch.Tensor, 
               output_path: str,
               fps: int = 16,
               quality: int = 8) -> None:
    """
    Save video tensor to file.
    
    Args:
        video_tensor: Video tensor [1, C, T, H, W] or [C, T, H, W]
        output_path: Output file path
        fps: Frames per second
        quality: Video quality (1-10, higher is better)
    """
    # Handle different input shapes
    if video_tensor.dim() == 5:
        video_tensor = video_tensor.squeeze(0)  # Remove batch dimension
    
    if video_tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor [C, T, H, W], got shape {video_tensor.shape}")
    
    # Move to CPU and convert to numpy
    video_np = video_tensor.detach().cpu().numpy()
    
    # Denormalize from [-1, 1] to [0, 255]
    video_np = (video_np + 1.0) * 127.5
    video_np = np.clip(video_np, 0, 255).astype(np.uint8)
    
    # Rearrange from [C, T, H, W] to [T, H, W, C]
    video_np = video_np.transpose(1, 2, 3, 0)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Save video
    try:
        writer = imageio.get_writer(
            output_path, 
            fps=fps, 
            quality=quality,
            macro_block_size=None
        )
        
        for frame in video_np:
            writer.append_data(frame)
            
        writer.close()
        print(f"Video saved to: {output_path}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save video to {output_path}: {str(e)}")


def load_image(image_path: str, target_size: Optional[tuple] = None) -> torch.Tensor:
    """
    Load image and convert to tensor format compatible with video processing.
    
    Args:
        image_path: Path to image file
        target_size: Target size as (width, height)
        
    Returns:
        Image tensor of shape [1, C, 1, H, W] normalized to [-1, 1]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        image = Image.open(image_path).convert('RGB')
        
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).float()
        
        # Normalize to [-1, 1]
        image_tensor = image_tensor / 127.5 - 1.0
        
        # Rearrange to [1, C, 1, H, W] for video-like format
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        
        return image_tensor
        
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")


def save_frames(video_tensor: torch.Tensor, output_dir: str, prefix: str = "frame") -> None:
    """
    Save video frames as individual images.
    
    Args:
        video_tensor: Video tensor [1, C, T, H, W] or [C, T, H, W]
        output_dir: Output directory
        prefix: Filename prefix
    """
    if video_tensor.dim() == 5:
        video_tensor = video_tensor.squeeze(0)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy
    video_np = video_tensor.detach().cpu().numpy()
    video_np = (video_np + 1.0) * 127.5
    video_np = np.clip(video_np, 0, 255).astype(np.uint8)
    
    # Save each frame
    for t in range(video_np.shape[1]):
        frame = video_np[:, t, :, :].transpose(1, 2, 0)  # [H, W, C]
        frame_image = Image.fromarray(frame)
        frame_path = os.path.join(output_dir, f"{prefix}_{t:04d}.png")
        frame_image.save(frame_path)
    
    print(f"Saved {video_np.shape[1]} frames to {output_dir}")


def get_video_info(video_path: str) -> dict:
    """
    Get video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        
        # Count frames
        frame_count = 0
        for _ in reader:
            frame_count += 1
        reader.close()
        
        # Re-open to get first frame size
        reader = imageio.get_reader(video_path)
        first_frame = next(iter(reader))
        height, width = first_frame.shape[:2]
        reader.close()
        
        info = {
            'fps': meta.get('fps', 30),
            'frame_count': frame_count,
            'duration': frame_count / meta.get('fps', 30),
            'width': width,
            'height': height,
            'size': os.path.getsize(video_path)
        }
        
        return info
        
    except Exception as e:
        raise RuntimeError(f"Failed to get video info for {video_path}: {str(e)}")