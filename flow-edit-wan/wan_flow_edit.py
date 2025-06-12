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
        self.device = device
        # WanVAE doesn't have config, uses direct normalization
        # Scaling is handled internally by the VAE's encode/decode methods
        
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
        
        with torch.no_grad():
            # Wan VAE expects a list of [C, T, H, W] tensors
            latents = self.pipeline.vae.encode([video_frames.to(self.device)])[0]
            
        return latents
    
    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to video frames.
        
        Args:
            latents: Latent tensor
            
        Returns:
            Decoded video frames [C, T, H, W]
        """
        with torch.no_grad():
            # Wan VAE decode method expects list of latents
            video_frames = self.pipeline.vae.decode([latents.to(self.device)])[0]
            
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
                        guidance_scale: float = 7.5) -> torch.Tensor:
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
        # Prepare model arguments similar to how Wan I2V does it
        arg_dict = {
            'context': prompt_embeds["text_states"],
            'context_mask': prompt_embeds.get("text_mask"),
            'context_null': prompt_embeds.get("text_states"),  # Use same for now, CFG handled outside
            'context_null_mask': prompt_embeds.get("text_mask"),
            'context_clip': None,  # No CLIP context for FlowEdit
            'guide_scale': guidance_scale,
        }
        
        # Add secondary text states if available
        if prompt_embeds.get("text_states_2") is not None:
            arg_dict['context_2'] = prompt_embeds["text_states_2"]
            arg_dict['context_null_2'] = prompt_embeds["text_states_2"]
        
        with torch.no_grad():
            # Use Wan's actual model call interface
            # Remove batch dimension for model call: [1, C, T, H, W] -> [C, T, H, W]
            latents_no_batch = latents.squeeze(0) if latents.dim() == 5 else latents
            latent_model_input = [latents_no_batch]
            timestep_tensor = timestep.unsqueeze(0) if timestep.dim() == 0 else timestep
            
            velocity = self.pipeline.model(
                latent_model_input, 
                t=timestep_tensor, 
                **arg_dict
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
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
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
            
            vt_src = self.predict_velocity(zt_src, t_tensor, src_embeds, src_guidance_scale)
            vt_tar = self.predict_velocity(zt_tar, t_tensor, tar_embeds, tar_guidance_scale)
            
            # Accumulate velocity difference
            V_delta_avg += (1.0 / n_avg) * (vt_tar - vt_src)
        
        # ODE integration step
        x_edit = x_edit.to(torch.float32)
        V_delta_avg = V_delta_avg.to(torch.float32)
        
        x_edit_new = x_edit + (timestep_prev - timestep) * V_delta_avg
        
        return x_edit_new.to(x_src.dtype)
    
    def drift_step(self,
                   x_edit: torch.Tensor,
                   timestep: float,
                   timestep_prev: float,
                   tar_embeds: dict,
                   guidance_scale: float) -> torch.Tensor:
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
        vt_tar = self.predict_velocity(x_edit, t_tensor, tar_embeds, guidance_scale)
        
        x_edit = x_edit.to(torch.float32)
        vt_tar = vt_tar.to(torch.float32)
        
        x_edit_new = x_edit + (timestep_prev - timestep) * vt_tar
        
        return x_edit_new.to(x_edit.dtype)
    
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
        
        # Ensure latents have batch dimension for flow edit operations
        if x_src.dim() == 4:
            x_src = x_src.unsqueeze(0)  # [C, T, H, W] -> [1, C, T, H, W]
        
        # Encode prompts
        print("Encoding prompts...")
        src_embeds = self.encode_prompts(source_prompt)
        tar_embeds = self.encode_prompts(target_prompt)
        
        # Setup timesteps - Wan uses different scheduling
        from wan.utils.fm_solvers import get_sampling_sigmas
        
        # Get sigmas for flow matching
        sigmas = get_sampling_sigmas(steps, shift=flow_shift, device=self.device)
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
                        x_edit, t_val, t_prev_val, tar_embeds, drift_guidance_scale
                    )
                else:
                    # FlowEdit step: differential velocity
                    x_edit = self.flow_edit_step(
                        x_src, x_edit, t_val, t_prev_val,
                        src_embeds, tar_embeds,
                        source_guidance_scale, target_guidance_scale,
                        n_avg, generator
                    )
                
                pbar.update(1)
        
        # Decode back to video
        print("Decoding latents to video...")
        edited_video = self.decode_video(x_edit)
        
        # Add batch dimension back for compatibility: [C, T, H, W] -> [1, C, T, H, W]
        if edited_video.dim() == 4:
            edited_video = edited_video.unsqueeze(0)
        
        return edited_video