from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .fastwam_joint import FastWAMJoint


class FastWAMSDPJoint(FastWAMJoint):
    """FastWAM-Joint trained with Streaming Diffusion Policy action noise.

    SDP keeps a rolling action trajectory at different denoising levels: near
    actions are almost clean, distant actions are noisier.  This class keeps the
    video objective unchanged and applies the SDP chunk-wise noise schedule only
    to the action branch.
    """

    @classmethod
    def from_wan22_pretrained(
        cls,
        *args,
        sdp_action_train_prob: float = 1.0,
        sdp_action_steps_choices: Optional[list[int]] = None,
        sdp_num_noise_levels: int = 32,
        sdp_random_offset: bool = True,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(*args, **kwargs)
        model.sdp_action_train_prob = float(sdp_action_train_prob)
        model.sdp_action_steps_choices = [
            int(x) for x in (sdp_action_steps_choices or [1, 2, 4])
        ]
        model.sdp_num_noise_levels = int(sdp_num_noise_levels)
        model.sdp_random_offset = bool(sdp_random_offset)
        if model.sdp_num_noise_levels <= 0:
            raise ValueError("`sdp_num_noise_levels` must be positive.")
        if not model.sdp_action_steps_choices:
            raise ValueError("`sdp_action_steps_choices` must not be empty.")
        return model

    def _sample_sdp_action_timestep(
        self,
        batch_size: int,
        action_horizon: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return token-wise action timesteps [B,T] and selected K [B].

        This mirrors SDP's chunk-wise regime: positions are grouped into chunks
        of K actions; later chunks receive larger noise indices.  A random offset
        j represents the partially denoised rolling-buffer state seen during
        streaming execution.
        """
        valid_steps = [k for k in self.sdp_action_steps_choices if 1 <= k <= action_horizon]
        if not valid_steps:
            valid_steps = [1]

        step_table = torch.tensor(valid_steps, device=device, dtype=torch.long)
        step_idx = torch.randint(len(valid_steps), (batch_size,), device=device)
        action_steps = step_table[step_idx]

        positions = torch.arange(action_horizon, device=device, dtype=torch.long).unsqueeze(0)
        chunk_idx = torch.div(positions, action_steps.unsqueeze(1), rounding_mode="floor")
        num_chunks = torch.div(
            torch.full_like(action_steps, action_horizon) + action_steps - 1,
            action_steps,
            rounding_mode="floor",
        ).clamp(min=1)
        chunk_idx = torch.minimum(chunk_idx, (num_chunks - 1).unsqueeze(1))

        levels = int(self.sdp_num_noise_levels)
        base_idx = (
            torch.floor(
                levels
                * (chunk_idx.to(torch.float32) + 1.0)
                / num_chunks.to(torch.float32).unsqueeze(1)
            ).to(torch.long)
            - 1
        )

        if self.sdp_random_offset:
            upper = torch.div(
                torch.full_like(num_chunks, levels),
                num_chunks,
                rounding_mode="floor",
            ).clamp(min=1)
            rand = torch.rand((batch_size,), device=device)
            offset = torch.floor(rand * upper.to(torch.float32)).to(torch.long)
            base_idx = base_idx - offset.unsqueeze(1)

        noise_idx = base_idx.clamp(min=0, max=levels - 1)
        sigma = (noise_idx.to(torch.float32) + 1.0) / float(levels)
        timestep = sigma * float(self.train_action_scheduler.num_train_timesteps)
        return timestep.to(dtype=dtype), action_steps

    @staticmethod
    def _add_tokenwise_flow_noise(
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        num_train_timesteps: int,
    ) -> torch.Tensor:
        sigma = (timestep / float(num_train_timesteps)).to(device=sample.device, dtype=sample.dtype)
        return (1.0 - sigma.unsqueeze(-1)) * sample + sigma.unsqueeze(-1) * noise

    def _build_sdp_action_ladder(
        self,
        action_horizon: int,
        action_steps: int,
        *,
        num_noise_levels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build token-wise pre-update timesteps and one-step deltas.

        For action_steps=2 and action_horizon=32 this gives pre-update noise
        levels [2,2,4,4,...,32,32] / 32 and deltas [-2,-2,...] / 32, so
        one model step turns the first two actions clean and shifts the rest to
        the next rolling state.
        """
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")
        if action_steps <= 0:
            raise ValueError(f"`action_steps` must be positive, got {action_steps}")
        if num_noise_levels <= 0:
            raise ValueError(f"`num_noise_levels` must be positive, got {num_noise_levels}")

        action_steps = min(int(action_steps), int(action_horizon))
        num_chunks = max(1, math.ceil(int(action_horizon) / action_steps))
        positions = torch.arange(action_horizon, device=device, dtype=torch.long)
        chunk_idx = torch.div(positions, action_steps, rounding_mode="floor")

        levels = float(num_noise_levels)
        pre_level = torch.floor(levels * (chunk_idx.to(torch.float32) + 1.0) / float(num_chunks))
        post_level = torch.floor(levels * chunk_idx.to(torch.float32) / float(num_chunks))
        pre_level = pre_level.to(torch.long).clamp(min=1, max=num_noise_levels)
        post_level = post_level.to(torch.long).clamp(min=0, max=num_noise_levels)

        timestep = pre_level.to(torch.float32).unsqueeze(0) / levels
        timestep = timestep * float(self.infer_action_scheduler.num_train_timesteps)
        delta = (post_level.to(torch.float32) - pre_level.to(torch.float32)).unsqueeze(0) / levels
        return timestep.to(device=device, dtype=dtype), delta.to(device=device, dtype=dtype)

    @torch.no_grad()
    def infer_action_sdp_rolling_step(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        action_buffer: Optional[torch.Tensor] = None,
        action_steps: int = 1,
        sdp_num_noise_levels: int = 32,
        sdp_warm_start: bool = True,
        sdp_warm_start_inference_steps: int = 1,
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
    ) -> dict[str, Any]:
        """Run one SDP-style rolling action denoising step.

        This is intentionally separate from `infer_action`: the original path
        samples a fresh action chunk every call, while this path consumes and
        returns a persistent noisy action buffer.
        """
        del negative_prompt, text_cfg_scale
        self.eval()

        if int(num_inference_steps) != 1:
            raise ValueError(
                "SDP rolling inference expects EVALUATION.num_inference_steps=1; "
                f"got {num_inference_steps}."
            )
        action_steps = min(max(1, int(action_steps)), int(action_horizon))

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")

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

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
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

        if action_buffer is None:
            if sdp_warm_start:
                warm = super().infer_action(
                    prompt=prompt,
                    input_image=input_image.clone(),
                    action_horizon=action_horizon,
                    num_video_frames=num_video_frames,
                    proprio=proprio.clone() if proprio is not None else None,
                    context=context.clone() if not use_prompt else None,
                    context_mask=context_mask.clone() if not use_prompt else None,
                    num_inference_steps=int(sdp_warm_start_inference_steps),
                    sigma_shift=sigma_shift,
                    seed=seed,
                    rand_device=rand_device,
                    tiled=tiled,
                )["action"].unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
                warm_timestep, _ = self._build_sdp_action_ladder(
                    action_horizon=action_horizon,
                    action_steps=action_steps,
                    num_noise_levels=int(sdp_num_noise_levels),
                    device=self.device,
                    dtype=self.torch_dtype,
                )
                warm_sigma = warm_timestep / float(self.infer_action_scheduler.num_train_timesteps)
                warm_noise = torch.randn(
                    warm.shape,
                    generator=action_generator,
                    device=rand_device,
                    dtype=torch.float32,
                ).to(device=self.device, dtype=self.torch_dtype)
                latents_action = (1.0 - warm_sigma.unsqueeze(-1)) * warm + warm_sigma.unsqueeze(-1) * warm_noise
            else:
                latents_action = torch.randn(
                    (1, action_horizon, self.action_expert.action_dim),
                    generator=action_generator,
                    device=rand_device,
                    dtype=torch.float32,
                ).to(device=self.device, dtype=self.torch_dtype)
        else:
            if action_buffer.ndim == 2:
                action_buffer = action_buffer.unsqueeze(0)
            if action_buffer.shape != (1, action_horizon, self.action_expert.action_dim):
                raise ValueError(
                    "`action_buffer` must have shape [T,D] or [1,T,D] matching "
                    f"({action_horizon}, {self.action_expert.action_dim}), got {tuple(action_buffer.shape)}"
                )
            latents_action = action_buffer.to(device=self.device, dtype=self.torch_dtype)

        timestep_action, delta_action = self._build_sdp_action_ladder(
            action_horizon=action_horizon,
            action_steps=action_steps,
            num_noise_levels=int(sdp_num_noise_levels),
            device=self.device,
            dtype=latents_action.dtype,
        )
        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=1,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        timestep_video = infer_timesteps_video[0].unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)

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
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        del latents_video

        latents_action = latents_action + pred_action * delta_action.unsqueeze(-1)
        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)

        next_buffer = torch.empty_like(latents_action)
        if action_steps < action_horizon:
            next_buffer[:, :-action_steps] = latents_action[:, action_steps:]
        tail_noise = torch.randn(
            (1, action_steps, self.action_expert.action_dim),
            generator=tail_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        next_buffer[:, -action_steps:] = tail_noise

        return {
            "action": action_out,
            "action_buffer": next_buffer.detach().to(device="cpu", dtype=torch.float32),
            "sdp_action_steps": action_steps,
            "sdp_pre_sigma": (
                timestep_action[0].detach().to(device="cpu", dtype=torch.float32)
                / float(self.infer_action_scheduler.num_train_timesteps)
            ),
            "sdp_delta_sigma": delta_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        use_sdp_action = (
            torch.is_grad_enabled()
            and self.sdp_action_train_prob > 0.0
            and torch.rand((), device=action.device).item() < self.sdp_action_train_prob
        )
        sdp_action_steps = None
        if use_sdp_action:
            timestep_action, sdp_action_steps = self._sample_sdp_action_timestep(
                batch_size=batch_size,
                action_horizon=action.shape[1],
                device=action.device,
                dtype=action.dtype,
            )
        else:
            # SDP's constant schedule is still token-wise: every action token in
            # a sample gets the same timestep. Keeping the tensor shape [B,T]
            # makes the distributed graph and metric gathers identical to the
            # chunk-wise branch.
            timestep_scalar = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=action.dtype,
            )
            timestep_action = timestep_scalar[:, None].expand(batch_size, action.shape[1])

        noisy_action = self._add_tokenwise_flow_noise(
            action,
            noise_action,
            timestep_action,
            self.train_action_scheduler.num_train_timesteps,
        )
        target_action = noise_action - action

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_tokens, "action": action_tokens},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_token.device, dtype=action_loss_token.dtype
        )
        if action_weight.ndim == 2:
            weighted_action_loss_token = action_loss_token * action_weight
            if action_is_pad is not None:
                valid = (~action_is_pad).to(weighted_action_loss_token.device, dtype=weighted_action_loss_token.dtype)
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                loss_action = ((weighted_action_loss_token * valid).sum(dim=1) / valid_sum).mean()
            else:
                loss_action = weighted_action_loss_token.mean()
        else:
            if action_is_pad is not None:
                valid = (~action_is_pad).to(action_loss_token.device, dtype=action_loss_token.dtype)
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
            else:
                action_loss_per_sample = action_loss_token.mean(dim=1)
            loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        sigma_mean = (
            timestep_action.float()
            / float(self.train_action_scheduler.num_train_timesteps)
        ).mean()
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "sdp_action_train": float(use_sdp_action),
            "sdp_action_k_mean": (
                float(sdp_action_steps.float().mean().detach().item())
                if sdp_action_steps is not None
                else 0.0
            ),
            "sdp_action_sigma_mean": float(sigma_mean.detach().item()),
        }
        return loss_total, loss_dict
