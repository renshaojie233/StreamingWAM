from typing import Any, Optional

import torch

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM

logger = get_logger(__name__)


class FastWAMJoint(FastWAM):
    """FastWAM variant where action attends to all video latent tokens."""

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError(
                "`video_dit_config` must be provided as dict for FastWAMJoint."
            )
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "FastWAMJoint requires `video_dit_config['action_conditioned']=false`."
            )
        return super().from_wan22_pretrained(**kwargs)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        condition_tokens = min(
            self._num_condition_latent_frames() * video_tokens_per_frame,
            video_seq_len,
        )
        if condition_tokens > 0:
            # Keep the clean visual history as a condition prefix.
            mask[:condition_tokens, condition_tokens:video_seq_len] = False
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> full video
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        if test_action_with_infer_action:
            logger.warning(
                "`FastWAMJoint.infer_joint` ignores `test_action_with_infer_action=True` "
                "and always runs with `test_action_with_infer_action=False`."
            )
        return super().infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            test_action_with_infer_action=False,
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        action_prefix: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        return_video: bool = False,
    ) -> dict[str, Any]:
        self.eval()

        input_video = self._normalize_condition_video_tensor(input_image)
        _, _, history_frames, height, width = input_video.shape
        if history_frames >= num_video_frames:
            raise ValueError(
                "`input_image` history length must leave future video frames, "
                f"got T={history_frames}, num_video_frames={num_video_frames}."
            )
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        action_prefix_clean, action_prefix_steps = self._prepare_action_prefix(
            action_prefix,
            action_horizon,
            dtype=latents_action.dtype,
        )
        if action_prefix_steps > 0:
            latents_action[:, :action_prefix_steps] = action_prefix_clean

        input_video = input_video.to(device=self.device, dtype=self.torch_dtype)
        condition_latents = self._encode_video_latents(input_video, tiled=tiled)
        num_condition_latent_frames = int(condition_latents.shape[2])
        latents_video[:, :, :num_condition_latent_frames] = condition_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            timestep_action_for_dit = timestep_action
            if action_prefix_steps > 0:
                timestep_action_for_dit = timestep_action[:, None].expand(1, action_horizon).clone()
                timestep_action_for_dit[:, :action_prefix_steps] = 0

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action_for_dit,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=None,
            )

            latents_video = self.infer_video_scheduler.step(pred_video_posi, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action_posi, step_delta_action, latents_action)
            latents_video[:, :, :num_condition_latent_frames] = condition_latents.clone()
            if action_prefix_steps > 0:
                latents_action[:, :action_prefix_steps] = action_prefix_clean

        result = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if return_video:
            result["video"] = self._decode_latents(latents_video, tiled=tiled)
        return result

    @torch.no_grad()
    def infer_action_sdp_rolling_step(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        action_prefix: Optional[torch.Tensor] = None,
        action_buffer: Optional[torch.Tensor] = None,
        action_steps: int = 1,
        sdp_num_noise_levels: int = 32,
        sdp_warm_start: bool = True,
        sdp_warm_start_inference_steps: int = 1,
        sdp_action_denoise_substeps: int = 1,
        rolling_step_idx: int = 0,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 1,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        return_video: bool = False,
    ) -> dict[str, Any]:
        """Run one SDP rolling update on the future action suffix.

        The clean prefix, when provided, is always re-injected as context.  Only
        `action[prefix:]` owns a persistent rolling noisy buffer.
        """
        del negative_prompt, text_cfg_scale
        self.eval()

        if int(num_inference_steps) != 1:
            raise ValueError(
                "SDP rolling inference expects EVALUATION.num_inference_steps=1; "
                f"got {num_inference_steps}."
            )
        sdp_action_denoise_substeps = int(sdp_action_denoise_substeps)
        if sdp_action_denoise_substeps <= 0:
            raise ValueError(
                "`sdp_action_denoise_substeps` must be positive, "
                f"got {sdp_action_denoise_substeps}."
            )

        raw_proprio = proprio
        raw_context = context
        raw_context_mask = context_mask

        input_video = self._normalize_condition_video_tensor(input_image)
        _, _, history_frames, height, width = input_video.shape
        if history_frames >= num_video_frames:
            raise ValueError(
                "`input_image` history length must leave future video frames, "
                f"got T={history_frames}, num_video_frames={num_video_frames}."
            )
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        seed_base = None if seed is None else int(seed) + int(rolling_step_idx) * 9973
        video_generator = None if seed_base is None else torch.Generator(device=rand_device).manual_seed(seed_base + 1)
        action_generator = None if seed_base is None else torch.Generator(device=rand_device).manual_seed(seed_base + 2)
        tail_generator = None if seed_base is None else torch.Generator(device=rand_device).manual_seed(seed_base + 3)

        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        action_prefix_clean, action_prefix_steps = self._prepare_action_prefix(
            action_prefix,
            action_horizon,
            dtype=self.torch_dtype,
        )
        future_horizon = int(action_horizon) - int(action_prefix_steps)
        if future_horizon <= 0:
            raise ValueError(
                f"action_horizon={action_horizon} must leave future actions after prefix={action_prefix_steps}."
            )
        action_steps = min(max(1, int(action_steps)), future_horizon)

        input_video = input_video.to(device=self.device, dtype=self.torch_dtype)
        condition_latents = self._encode_video_latents(input_video, tiled=tiled)
        num_condition_latent_frames = int(condition_latents.shape[2])
        latents_video[:, :, :num_condition_latent_frames] = condition_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        future_timestep, future_delta = self._build_sdp_future_action_ladder(
            future_horizon=future_horizon,
            action_steps=action_steps,
            num_noise_levels=int(sdp_num_noise_levels),
            device=self.device,
            dtype=self.torch_dtype,
        )

        if action_buffer is None:
            if sdp_warm_start:
                warm = self.infer_action(
                    prompt=prompt,
                    input_image=input_video.clone(),
                    action_horizon=action_horizon,
                    num_video_frames=num_video_frames,
                    action_prefix=action_prefix,
                    proprio=raw_proprio,
                    context=raw_context,
                    context_mask=raw_context_mask,
                    num_inference_steps=int(sdp_warm_start_inference_steps),
                    sigma_shift=sigma_shift,
                    seed=seed,
                    rand_device=rand_device,
                    tiled=tiled,
                )["action"].unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
                future_clean = warm[:, action_prefix_steps:]
                warm_sigma = future_timestep / float(self.infer_action_scheduler.num_train_timesteps)
                warm_noise = torch.randn(
                    future_clean.shape,
                    generator=action_generator,
                    device=rand_device,
                    dtype=torch.float32,
                ).to(device=self.device, dtype=self.torch_dtype)
                future_buffer = (1.0 - warm_sigma.unsqueeze(-1)) * future_clean + warm_sigma.unsqueeze(-1) * warm_noise
            else:
                future_buffer = torch.randn(
                    (1, future_horizon, self.action_expert.action_dim),
                    generator=action_generator,
                    device=rand_device,
                    dtype=torch.float32,
                ).to(device=self.device, dtype=self.torch_dtype)
        else:
            if action_buffer.ndim == 2:
                action_buffer = action_buffer.unsqueeze(0)
            if action_buffer.shape == (1, action_horizon, self.action_expert.action_dim):
                action_buffer = action_buffer[:, action_prefix_steps:]
            if action_buffer.shape != (1, future_horizon, self.action_expert.action_dim):
                raise ValueError(
                    "`action_buffer` must be future suffix [T,D]/[1,T,D] or full action buffer matching "
                    f"future ({future_horizon}, {self.action_expert.action_dim}); got {tuple(action_buffer.shape)}"
                )
            future_buffer = action_buffer.to(device=self.device, dtype=self.torch_dtype)

        latents_action = torch.empty(
            (1, action_horizon, self.action_expert.action_dim),
            device=self.device,
            dtype=self.torch_dtype,
        )
        if action_prefix_steps > 0:
            latents_action[:, :action_prefix_steps] = action_prefix_clean
        latents_action[:, action_prefix_steps:] = future_buffer

        timestep_action = torch.zeros((1, action_horizon), device=self.device, dtype=self.torch_dtype)
        timestep_action[:, action_prefix_steps:] = future_timestep

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=1,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        timestep_video = infer_timesteps_video[0].unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)

        if sdp_action_denoise_substeps == 1:
            # Preserve the original one-step path exactly for backward compatibility.
            pred_video, pred_action = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=None,
            )
            latents_video = self.infer_video_scheduler.step(pred_video, infer_deltas_video[0], latents_video)
            latents_video[:, :, :num_condition_latent_frames] = condition_latents.clone()
            future_buffer = future_buffer + pred_action[:, action_prefix_steps:] * future_delta.unsqueeze(-1)
            latents_action[:, action_prefix_steps:] = future_buffer
            if action_prefix_steps > 0:
                latents_action[:, :action_prefix_steps] = action_prefix_clean
        else:
            # Advance video once, then hold its denoised latent at t=0 while the
            # action traverses the same SDP stair in smaller solver steps.
            action_delta = future_delta / float(sdp_action_denoise_substeps)
            action_timestep_delta = (
                future_delta
                * float(self.infer_action_scheduler.num_train_timesteps)
                / float(sdp_action_denoise_substeps)
            )
            for substep in range(sdp_action_denoise_substeps):
                timestep_action[:, action_prefix_steps:] = (
                    future_timestep + action_timestep_delta * float(substep)
                )
                pred_video, pred_action = self._predict_joint_noise(
                    latents_video=latents_video,
                    latents_action=latents_action,
                    timestep_video=timestep_video,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    gt_action=None,
                )
                if substep == 0:
                    latents_video = self.infer_video_scheduler.step(
                        pred_video,
                        infer_deltas_video[0],
                        latents_video,
                    )
                    latents_video[:, :, :num_condition_latent_frames] = condition_latents.clone()
                    timestep_video = torch.zeros_like(timestep_video)
                future_buffer = (
                    future_buffer
                    + pred_action[:, action_prefix_steps:] * action_delta.unsqueeze(-1)
                )
                latents_action[:, action_prefix_steps:] = future_buffer
                if action_prefix_steps > 0:
                    latents_action[:, :action_prefix_steps] = action_prefix_clean
        predicted_video = self._decode_latents(latents_video, tiled=tiled) if return_video else None
        del latents_video

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)

        next_future_buffer = torch.empty_like(future_buffer)
        if action_steps < future_horizon:
            next_future_buffer[:, :-action_steps] = future_buffer[:, action_steps:]
        tail_noise = torch.randn(
            (1, action_steps, self.action_expert.action_dim),
            generator=tail_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        next_future_buffer[:, -action_steps:] = tail_noise

        result = {
            "action": action_out,
            "action_buffer": next_future_buffer.detach().to(device="cpu", dtype=torch.float32),
            "sdp_action_steps": action_steps,
            "sdp_action_denoise_substeps": sdp_action_denoise_substeps,
            "sdp_pre_sigma": (
                future_timestep[0].detach().to(device="cpu", dtype=torch.float32)
                / float(self.infer_action_scheduler.num_train_timesteps)
            ),
            "sdp_delta_sigma": future_delta[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if predicted_video is not None:
            result["video"] = predicted_video
        return result
