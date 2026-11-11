# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].
# Action repeat is inspired by CogACT
# Modified by [Teli Ma/ HKUST GZ] in [2025]. 
# Modification: [modify to match with DiT4DiT].


from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
import os

from DiT4DiT.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from DiT4DiT.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT
from DiT4DiT.model.modules.action_model.streaming import (
    euler_step, interpolate, noise_ladder, position_ids, roll_buffer,
)


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,) or (B, T)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        elif timesteps.dim() != 2 or timesteps.shape != (B, T):
            raise ValueError(
                "Expected `timesteps` to have shape (B,) or (B,T)."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x



class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}

class FlowmatchingActionHead(nn.Module):
    """Official head with an opt-in stateless H8/K4 streaming branch.

    Set action_model.streaming=true; optional streaming_ladder_probability defaults
    to 0.8. Layout knobs streaming_prefix_steps/execution_steps/warm_start_steps
    currently require 8/4/4; streaming_noise_levels defaults to 32 (must be even).
    A streaming mapping with enabled and the same unprefixed keys is also accepted.
    """
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.hidden_size = config.hidden_size
        self.full_config = full_config
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = MLP(
            input_dim=config.state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        ) if config.state_dim else None

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        streaming = config.get("streaming", False)
        assert isinstance(streaming, bool) or hasattr(streaming, "get"), "streaming must be a bool or config mapping"
        self.streaming_enabled = streaming if isinstance(streaming, bool) else bool(streaming.get("enabled", False))
        if self.streaming_enabled:
            def option(name, default):
                fallback = config.get("streaming_" + name, default)
                return fallback if isinstance(streaming, bool) else streaming.get(name, fallback)

            self.streaming_prefix_steps = option("prefix_steps", 8)
            self.streaming_execution_steps = option("execution_steps", 4)
            self.streaming_warm_start_steps = option("warm_start_steps", 4)
            self.streaming_ladder_probability = option("ladder_probability", 0.8)
            self.streaming_noise_levels = option("noise_levels", 32)
            assert self.action_dim == 8, "streaming requires D=8 (first seven active, last padded)"
            assert self.action_horizon == config.action_horizon == 8, "streaming requires H=8"
            assert isinstance(self.streaming_prefix_steps, int) and self.streaming_prefix_steps == 8, "streaming requires P=8"
            assert isinstance(self.streaming_execution_steps, int) and self.streaming_execution_steps == 4, "streaming currently supports only K=4"
            assert isinstance(self.streaming_warm_start_steps, int) and self.streaming_warm_start_steps == 4, "streaming requires four ordinary warm-start steps"
            assert 0 <= self.streaming_ladder_probability <= 1, "ladder_probability must be in [0,1]"
            assert isinstance(self.streaming_noise_levels, int) and self.streaming_noise_levels >= 2, "invalid noise levels"
            assert self.streaming_noise_levels % 2 == 0, "H8/K4 requires an exact half-noise level"
            if config.add_pos_embed:
                assert config.max_seq_len >= 16, "streaming requires position IDs 0..15"

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return sample / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)


    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        if self.streaming_enabled:
            return self._forward_streaming(vl_embs, actions, action_mask, state, encoder_attention_mask)
        device = vl_embs.device

        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * actions + t * noise
        velocity = noise - actions

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)


        # embed state
        state_features = self.state_encoder(state) if state is not None else None


        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1) \
            if state_features is not None else action_features

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )


        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2) * action_mask
        loss = loss.sum() / action_mask.sum()
        return loss

    def _streaming_mask(self, actions, action_mask=None):
        """Full masks include episode validity; dimension 7 is never a model action."""
        assert actions.ndim == 3 and actions.shape[1:] == (16, 8), "streaming actions must be [B,16,8]"
        mask = torch.ones_like(actions, dtype=torch.bool)
        if action_mask is not None:
            assert action_mask.shape in (actions.shape, (actions.shape[0], 8, 8)), "action_mask must be [B,16,8] or future-only [B,8,8]"
            action_mask = action_mask.to(device=actions.device, dtype=torch.bool)
            if action_mask.shape == actions.shape:
                mask = action_mask.clone()
            else:
                mask[:, 8:] = action_mask
        mask[:, :, 7] = False
        return mask

    def _streaming_velocity(self, vl_embs, prefix, future, future_times, state=None,
                            encoder_attention_mask=None):
        batch = vl_embs.shape[0]
        assert prefix.shape == future.shape == (batch, 8, 8), "prefix/future must be [B,8,8]"
        assert future_times.shape == (batch, 8), "future times must be [B,8]"
        # The prefix is rebuilt from the caller's clean commands on every model pass.
        actions = torch.cat((prefix, future), dim=1)
        action_times = torch.cat((torch.zeros_like(future_times), future_times), dim=1)
        # BF16 multiplication would incorrectly round the ordinary 0.75 bucket to 752.
        action_times = (action_times.float() * self.num_timestep_buckets).long()
        action_features = self.action_encoder(actions, action_times)
        if self.config.add_pos_embed:
            ids = position_ids(device=actions.device)
            action_features = action_features + self.position_embedding(ids).unsqueeze(0)

        state_features = None
        if state is not None:
            assert self.state_encoder is not None, "state provided but state encoder is disabled"
            assert state.shape == (batch, 1, self.config.state_dim), "state must be [B,1,state_dim]"
            state_features = self.state_encoder(state)
        sa_embs = action_features
        model_times = action_times
        if state_features is not None:
            sa_embs = torch.cat((state_features, action_features), dim=1)
            model_times = torch.cat((action_times.new_zeros((batch, 1)), action_times), dim=1)
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=model_times,
            return_all_hidden_states=False,
        )
        return self.action_decoder(model_output)[:, -8:]

    def _forward_streaming(self, vl_embs, actions, action_mask, state, encoder_attention_mask):
        mask = self._streaming_mask(actions, action_mask)
        assert actions.shape[0] == vl_embs.shape[0], "action/feature batch mismatch"
        actions = torch.where(mask, actions, 0)
        prefix, clean = actions[:, :8], actions[:, 8:]
        future_mask = mask[:, 8:]
        noise = torch.where(future_mask, torch.randn_like(clean), 0)
        pre, _ = noise_ladder(noise_levels=self.streaming_noise_levels,
                              device=clean.device, dtype=clean.dtype)
        # The ordinary component retains the official per-sample Beta time sampler.
        ordinary = self.sample_time(clean.shape[0], clean.device, clean.dtype)[:, None].expand(-1, 8)
        use_ladder = torch.rand((clean.shape[0], 1), device=clean.device) < self.streaming_ladder_probability
        times = torch.where(use_ladder, pre.unsqueeze(0), ordinary)
        future = interpolate(clean, noise, times)
        pred = self._streaming_velocity(vl_embs, prefix, future, times, state, encoder_attention_mask)
        # Accumulate in FP32; no prefix, episode-padding, or padded-dimension loss.
        error = torch.where(future_mask, pred.float() - (noise - clean).float(), 0)
        return error.square().sum() / future_mask.sum().clamp_min(1)

    @torch.no_grad()
    def predict_streaming_action(self, vl_embs: torch.Tensor, prefix: torch.Tensor,
                                 state: torch.Tensor = None, buffer: torch.Tensor = None,
                                 action_mask: torch.Tensor = None, encoder_attention_mask=None,
                                 generator: torch.Generator = None) -> dict:
        """Stateless H8/K4 inference in normalized model coordinates.

        prefix: [B,8,8] actually executed commands, oldest first (time zero).
        buffer: previous FULL POST-update [B,8,8], or None on episode reset.
        action_mask: optional [B,16,8] prefix+future validity, or [B,8,8] future validity.
        Returns normalized_actions [B,4,8] and action_buffer [B,8,8] (post-update).
        Feed back action_buffer exactly once, only after executing all four commands.
        No clipping, normalization, environment conversion, or mutable episode state.
        """
        assert self.streaming_enabled, "enable framework.action_model.streaming first"
        batch = vl_embs.shape[0]
        assert prefix.shape == (batch, 8, 8), "prefix must be [B,8,8]"
        device, dtype = vl_embs.device, vl_embs.dtype
        prefix = prefix.to(device=device, dtype=dtype)
        template = torch.cat((prefix, torch.zeros_like(prefix)), dim=1)
        mask = self._streaming_mask(template, action_mask)
        prefix = torch.where(mask[:, :8], prefix, 0)
        future_mask = mask[:, 8:]
        pre, post = noise_ladder(noise_levels=self.streaming_noise_levels, device=device, dtype=dtype)

        def fresh(length):
            return torch.randn((batch, length, 8), device=device, dtype=dtype, generator=generator)

        if buffer is None:
            future = torch.where(future_mask, fresh(8), 0)
            steps = self.streaming_warm_start_steps
            # All future tokens take ordinary 1 -> 0 Euler steps before ladder re-noising.
            for step in range(steps):
                times = future.new_full((batch, 8), 1.0 - step / steps)
                velocity = self._streaming_velocity(vl_embs, prefix, future, times, state, encoder_attention_mask)
                future = torch.where(future_mask, future - velocity / steps, 0)
            future = interpolate(future, fresh(8), pre)
        else:
            assert buffer.shape == (batch, 8, 8), "buffer must be full post-update [B,8,8], not a suffix"
            buffer = buffer.to(device=device, dtype=dtype)
            future = roll_buffer(buffer, fresh(4), self.streaming_execution_steps)
        future = torch.where(future_mask, future, 0)
        times = pre.unsqueeze(0).expand(batch, -1)
        velocity = self._streaming_velocity(vl_embs, prefix, future, times, state, encoder_attention_mask)
        future = torch.where(future_mask, euler_step(future, velocity, pre, post), 0)
        return {"normalized_actions": future[:, :4].clone(), "action_buffer": future}

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn( # yes, here make sure action_horizon align with data loader? or share from clinet?
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        
        state_features = self.state_encoder(state) if state is not None else None

        # Run denoising steps (t goes from 1 -> 0, i.e. noise -> clean).
        for t in range(num_steps):
            t_cont = 1.0 - t / float(num_steps)  # e.g. goes 1, (N-1)/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1) \
                if state_features is not None else action_features


            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration (stepping from noise toward clean).
            actions = actions - dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype



def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return FlowmatchingActionHead(
        full_config=config
    )


if __name__ == "__main__":
    pass
