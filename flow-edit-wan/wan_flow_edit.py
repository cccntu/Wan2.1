"""
Core FlowEdit implementation adapted for Wan 2.1 models.
Based on the FlowEdit paper: https://arxiv.org/abs/2412.08629
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from tqdm import tqdm
import numpy as np

from diffusers.utils.torch_utils import randn_tensor


class FlowEditWan:
    """FlowEdit implementation for Wan 2.1 video models."""
    
    def __init__(self, pipeline, device="cuda"):
        """
        Initialize FlowEdit with Wan 2.1 pipeline.
        
        Args:
            pipeline: Loaded Wan 2.1 pipeline (WanI2V, WanT2V, etc.)
            device: Device to run on
        """
        self.pipeline = pipeline
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Get model's native dtype for memory efficiency
        self.model_dtype = getattr(pipeline.config, 'param_dtype', torch.bfloat16)
        print(f"Using model dtype: {self.model_dtype}")
        
        # WanVAE doesn't have config, uses direct normalization
        # Scaling is handled internally by the VAE's encode/decode methods
        
        # Ensure key models are on the correct device
        self._ensure_model_device_consistency()
    
    def _ensure_model_device_consistency(self):
        """Ensure all model components are on the correct device with consistent dtypes."""
        try:
            # Ensure main model is on correct device
            if hasattr(self.pipeline, 'model'):
                self.pipeline.model.to(self.device)
                
            # Ensure VAE is on correct device  
            if hasattr(self.pipeline, 'vae'):
                self.pipeline.vae.model.to(self.device)
                
            # CLIP model will be moved on demand to avoid memory issues
            print(f"Models configured for device: {self.device}, dtype: {self.model_dtype}")
            
        except Exception as e:
            print(f"Warning: Could not ensure model device consistency: {e}")
        
    def encode_video(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames to latent space.
        
        Args:
            video_frames: Video tensor [B, C, T, H, W] or [B, T, C, H, W]
            
        Returns:
            Encoded latents
        """
        if video_frames.dim() == 5 and video_frames.shape[2] == 3:
            # Convert [B, T, C, H, W] to [B, C, T, H, W]
            video_frames = video_frames.permute(0, 2, 1, 3, 4)
        
        # WanVAE expects list of [C, T, H, W] tensors (no batch dimension)
        if video_frames.dim() == 5:
            # Remove batch dimension: [1, C, T, H, W] -> [C, T, H, W]
            video_frames = video_frames.squeeze(0)
        
        # Ensure tensor is on correct device - use model's native dtype for memory efficiency
        video_frames = video_frames.to(device=self.device, dtype=self.model_dtype)
        
        with torch.no_grad():
            # Wan VAE expects a list of [C, T, H, W] tensors
            latents = self.pipeline.vae.encode([video_frames])[0]
            
        return latents
    
    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to video frames.
        
        Args:
            latents: Latent tensor
            
        Returns:
            Decoded video frames [C, T, H, W]
        """
        # Ensure latents are on correct device and dtype
        latents = latents.to(device=self.device, dtype=self.model_dtype)
        
        with torch.no_grad():
            # Wan VAE decode method expects list of latents
            video_frames = self.pipeline.vae.decode([latents])[0]
            
        return video_frames
    
    def encode_prompts(self, prompt: str) -> dict:
        """
        Encode text prompt using Wan's text encoders.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Dictionary with encoded prompt embeddings
        """
        # Wan text encoder interface
        if not self.pipeline.t5_cpu:
            self.pipeline.text_encoder.model.to(self.device)
            context = self.pipeline.text_encoder([prompt], self.device)
            # Don't offload during FlowEdit to avoid repeated loading
        else:
            context = self.pipeline.text_encoder([prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
        
        return {
            "text_states": context[0],  # Primary text features
            "text_mask": context[1] if len(context) > 1 else None,  # Attention mask
            "text_states_2": context[2] if len(context) > 2 else None,  # Secondary text features
        }
    
    def scale_noise_forward(self, sample: torch.Tensor, timestep: float, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward process in flow-matching: x_t = (1-t) * x_0 + t * noise
        
        Args:
            sample: Clean sample x_0
            timestep: Time t (normalized to [0,1])
            noise: Random noise
            
        Returns:
            Noisy sample x_t
        """
        return (1.0 - timestep) * sample + timestep * noise
    
    def predict_velocity(self, 
                        latents: torch.Tensor,
                        timestep: torch.Tensor, 
                        prompt_embeds: dict,
                        guidance_scale: float = 7.5,
                        video_frames: torch.Tensor = None) -> torch.Tensor:
        """
        Predict velocity using Wan 2.1 transformer.
        
        Args:
            latents: Current latent state
            timestep: Current timestep
            prompt_embeds: Text embeddings
            guidance_scale: Guidance scale for CFG
            
        Returns:
            Predicted velocity
        """
        with torch.no_grad():
            # Remove batch dimension for model call: [1, C, T, H, W] -> [C, T, H, W]
            latents_no_batch = latents.squeeze(0) if latents.dim() == 5 else latents
            latent_model_input = [latents_no_batch]
            timestep_tensor = timestep.unsqueeze(0) if timestep.dim() == 0 else timestep
            
            # Calculate sequence length for positional encoding
            C, T, H, W = latents_no_batch.shape
            seq_len = T * H * W // (self.pipeline.patch_size[1] * self.pipeline.patch_size[2])
            
            # For I2V model, we need clip_fea and y
            clip_fea = None
            y = None
            
            if hasattr(self.pipeline, 'clip') and video_frames is not None:
                # Extract CLIP features from first frame
                first_frame = video_frames[:, :, 0:1, :, :] if video_frames.dim() == 5 else video_frames[:, 0:1, :, :]
                
                # Ensure CLIP model is on correct device
                self.pipeline.clip.model.to(self.device)
                
                # Get target device and dtype from CLIP model
                clip_param = next(self.pipeline.clip.model.parameters())
                target_device = clip_param.device
                target_dtype = clip_param.dtype
                
                # Ensure first frame is on the correct device and dtype
                first_frame_clip = first_frame.squeeze(0).to(device=target_device, dtype=target_dtype)
                
                # Extract CLIP features
                with torch.no_grad():
                    clip_fea = self.pipeline.clip.visual([first_frame_clip])
                
                # Create conditional input y (first frame + zeros + mask)
                video_no_batch = video_frames.squeeze(0) if video_frames.dim() == 5 else video_frames
                
                # Create mask: 1 for first frame, 0 for others
                lat_h, lat_w = H, W
                msk = torch.ones(1, T * 4, lat_h, lat_w, device=self.device)  # T*4 for VAE temporal compression
                msk[:, 4:] = 0  # Set all frames except first to 0
                
                # Encode first frame and pad with zeros
                first_frame_padded = torch.cat([
                    video_no_batch[:, 0:1, :, :],  # First frame
                    torch.zeros(3, T-1, video_no_batch.shape[2], video_no_batch.shape[3], device=self.device)  # Zeros for other frames
                ], dim=1)
                
                # Encode the padded video - ensure consistent device and dtype
                first_frame_padded = first_frame_padded.to(device=self.device, dtype=self.model_dtype)
                y_encoded = self.pipeline.vae.encode([first_frame_padded])[0]
                
                # Ensure mask and encoded tensor are on same device
                msk = msk.to(device=y_encoded.device, dtype=y_encoded.dtype)
                y = [torch.cat([msk, y_encoded], dim=0)]
            
            # Use WanModel's actual forward signature
            velocity = self.pipeline.model(
                x=latent_model_input,
                t=timestep_tensor,
                context=[prompt_embeds["text_states"]],
                seq_len=seq_len,
                clip_fea=clip_fea,
                y=y
            )[0]
            
            # Add batch dimension back: [C, T, H, W] -> [1, C, T, H, W]
            if velocity.dim() == 4:
                velocity = velocity.unsqueeze(0)
        
        return velocity
    
    def flow_edit_step(self,
                      x_src: torch.Tensor,
                      x_edit: torch.Tensor,
                      timestep: float,
                      timestep_prev: float,
                      src_embeds: dict,
                      tar_embeds: dict,
                      src_guidance_scale: float,
                      tar_guidance_scale: float,
                      n_avg: int = 1,
                      generator: Optional[torch.Generator] = None,
                      video_frames: torch.Tensor = None) -> torch.Tensor:
        """
        Perform one FlowEdit step.
        
        Args:
            x_src: Source video latents
            x_edit: Current editing trajectory  
            timestep: Current timestep (normalized)
            timestep_prev: Previous timestep (normalized)
            src_embeds: Source prompt embeddings
            tar_embeds: Target prompt embeddings
            src_guidance_scale: Source guidance scale
            tar_guidance_scale: Target guidance scale
            n_avg: Number of predictions to average
            generator: Random generator
            
        Returns:
            Updated editing trajectory
        """
        V_delta_avg = torch.zeros_like(x_src)
        
        for k in range(n_avg):
            # Sample noise
            noise = randn_tensor(x_src.shape, generator=generator, device=self.device, dtype=x_src.dtype)
            
            # Forward process: get noisy versions
            zt_src = self.scale_noise_forward(x_src, timestep, noise)
            zt_tar = x_edit + zt_src - x_src
            
            # Predict velocities
            t_tensor = torch.tensor(timestep, device=self.device, dtype=x_src.dtype)
            
            vt_src = self.predict_velocity(zt_src, t_tensor, src_embeds, src_guidance_scale, video_frames)
            vt_tar = self.predict_velocity(zt_tar, t_tensor, tar_embeds, tar_guidance_scale, video_frames)
            
            # Accumulate velocity difference
            V_delta_avg += (1.0 / n_avg) * (vt_tar - vt_src)
        
        # ODE integration step - use model's native dtype for memory efficiency
        original_dtype = x_edit.dtype
        compute_dtype = self.model_dtype if self.model_dtype != torch.float16 else torch.float32
        
        x_edit = x_edit.to(device=self.device, dtype=compute_dtype)
        V_delta_avg = V_delta_avg.to(device=self.device, dtype=compute_dtype)
        
        x_edit_new = x_edit + (timestep_prev - timestep) * V_delta_avg
        
        return x_edit_new.to(device=self.device, dtype=original_dtype)
    
    def drift_step(self,
                   x_edit: torch.Tensor,
                   timestep: float,
                   timestep_prev: float,
                   tar_embeds: dict,
                   guidance_scale: float,
                   video_frames: torch.Tensor = None) -> torch.Tensor:
        """
        Perform drift step (regular sampling).
        
        Args:
            x_edit: Current state
            timestep: Current timestep
            timestep_prev: Previous timestep
            tar_embeds: Target embeddings
            guidance_scale: Guidance scale
            
        Returns:
            Updated state
        """
        t_tensor = torch.tensor(timestep, device=self.device, dtype=x_edit.dtype)
        vt_tar = self.predict_velocity(x_edit, t_tensor, tar_embeds, guidance_scale, video_frames)
        
        original_dtype = x_edit.dtype
        compute_dtype = self.model_dtype if self.model_dtype != torch.float16 else torch.float32
        
        x_edit = x_edit.to(device=self.device, dtype=compute_dtype)
        vt_tar = vt_tar.to(device=self.device, dtype=compute_dtype)
        
        x_edit_new = x_edit + (timestep_prev - timestep) * vt_tar
        
        return x_edit_new.to(device=self.device, dtype=original_dtype)
    
    def edit_video(self,
                   video_frames: torch.Tensor,
                   source_prompt: str,
                   target_prompt: str,
                   negative_prompt: str = "",
                   steps: int = 30,
                   skip_steps: int = 8,
                   drift_steps: int = 4,
                   n_avg: int = 1,
                   source_guidance_scale: float = 6.0,
                   target_guidance_scale: float = 12.0,
                   drift_guidance_scale: float = 6.0,
                   flow_shift: float = 6.0,
                   drift_flow_shift: float = 3.0,
                   seed: int = 42) -> torch.Tensor:
        """
        Main FlowEdit video editing function.
        
        Args:
            video_frames: Input video frames [B, C, T, H, W]
            source_prompt: Description of input video
            target_prompt: Description of desired output
            negative_prompt: Negative prompt
            steps: Total denoising steps
            skip_steps: Steps to skip at beginning
            drift_steps: Drift steps at end
            n_avg: Number of velocity predictions to average
            source_guidance_scale: Source guidance scale
            target_guidance_scale: Target guidance scale  
            drift_guidance_scale: Drift guidance scale
            flow_shift: Flow shift parameter
            drift_flow_shift: Drift flow shift parameter
            seed: Random seed
            
        Returns:
            Edited video frames
        """
        # Set up generator
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Encode video to latents
        print("Encoding video to latents...")
        x_src = self.encode_video(video_frames)
        print(f"Encoded latents shape: {x_src.shape}")
        
        # Clear memory after encoding
        self.clear_memory()
        
        # Ensure latents have batch dimension for flow edit operations
        if x_src.dim() == 4:
            x_src = x_src.unsqueeze(0)  # [C, T, H, W] -> [1, C, T, H, W]
        
        # Encode prompts
        print("Encoding prompts...")
        src_embeds = self.encode_prompts(source_prompt)
        tar_embeds = self.encode_prompts(target_prompt)
        
        # Clear memory after prompt encoding
        self.clear_memory()
        
        # Setup timesteps - Wan uses different scheduling
        from wan.utils.fm_solvers import get_sampling_sigmas
        
        # Get sigmas for flow matching
        sigmas = get_sampling_sigmas(steps, shift=flow_shift)
        sigmas = torch.from_numpy(sigmas).to(device=self.device, dtype=self.model_dtype)  # Use model's native dtype
        timesteps = sigmas * 1000.0  # Wan expects timesteps in [0, 1000] range
        
        # Apply flow shift by modifying timestep schedule
        if flow_shift != 1.0:
            timesteps = timesteps ** (1.0 / flow_shift)
        
        # Setup drift timesteps
        if drift_steps > 0:
            drift_timesteps = timesteps.clone()
            if drift_flow_shift != flow_shift:
                drift_timesteps = (drift_timesteps ** flow_shift) ** (1.0 / drift_flow_shift)
            timesteps[-drift_steps:] = drift_timesteps[-drift_steps:]
        
        # Initialize editing trajectory
        x_edit = x_src.clone()
        
        print(f"Starting FlowEdit with {steps} steps...")
        print(f"Skip steps: {skip_steps}, Drift steps: {drift_steps}")
        
        # Main editing loop
        with tqdm(total=len(timesteps) - skip_steps, desc="FlowEdit") as pbar:
            for i, (t, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                
                # Skip initial steps
                if i < skip_steps:
                    continue
                
                t_val = t.item()
                t_prev_val = t_prev.item()
                
                # Determine if we're in drift phase
                is_drift_phase = (len(timesteps) - 1 - i) <= drift_steps
                
                if is_drift_phase:
                    # Drift step: regular sampling
                    if i == len(timesteps) - 1 - drift_steps:
                        # Initialize drift phase with some noise
                        noise = randn_tensor(x_src.shape, generator=generator, device=self.device, dtype=x_src.dtype)
                        xt_src = self.scale_noise_forward(x_src, t_val, noise)
                        x_edit = x_edit + xt_src - x_src
                    
                    x_edit = self.drift_step(
                        x_edit, t_val, t_prev_val, tar_embeds, drift_guidance_scale, video_frames
                    )
                else:
                    # FlowEdit step: differential velocity
                    x_edit = self.flow_edit_step(
                        x_src, x_edit, t_val, t_prev_val,
                        src_embeds, tar_embeds,
                        source_guidance_scale, target_guidance_scale,
                        n_avg, generator, video_frames
                    )
                
                pbar.update(1)
        
        # Clear memory after editing loop
        self.clear_memory()
        
        # Decode back to video
        print("Decoding latents to video...")
        edited_video = self.decode_video(x_edit)
        
        # Add batch dimension back for compatibility: [C, T, H, W] -> [1, C, T, H, W]
        if edited_video.dim() == 4:
            edited_video = edited_video.unsqueeze(0)
        
        return edited_video
    
    def clear_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def reduce_video_resolution(video_tensor: torch.Tensor, max_size: int = 512) -> torch.Tensor:
        """
        Reduce video resolution to save memory.
        
        Args:
            video_tensor: Video tensor [B, C, T, H, W]
            max_size: Maximum height/width
            
        Returns:
            Resized video tensor
        """
        B, C, T, H, W = video_tensor.shape
        
        if max(H, W) <= max_size:
            return video_tensor
            
        # Calculate new dimensions while maintaining aspect ratio
        if H > W:
            new_H = max_size
            new_W = int(W * (max_size / H))
        else:
            new_W = max_size
            new_H = int(H * (max_size / W))
            
        # Ensure dimensions are divisible by 8 for VAE
        new_H = (new_H // 8) * 8
        new_W = (new_W // 8) * 8
        
        print(f"Reducing video resolution from {H}x{W} to {new_H}x{new_W} to save memory")
        
        # Reshape for interpolation: [B*T, C, H, W]
        video_reshaped = video_tensor.transpose(1, 2).reshape(B * T, C, H, W)
        
        # Resize
        import torch.nn.functional as F
        video_resized = F.interpolate(video_reshaped, size=(new_H, new_W), mode='bilinear', align_corners=False)
        
        # Reshape back: [B, C, T, H, W]
        video_resized = video_resized.reshape(B, T, C, new_H, new_W).transpose(1, 2)
        
        return video_resized