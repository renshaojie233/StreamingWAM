"""Stateless future-only flow-matching helpers (no model/backbone dependencies)."""

import torch


def noise_ladder(horizon=8, execution_steps=4, noise_levels=32, *, device=None, dtype=None):
    """Return pre/post sigmas; only divisor K supports a pure-noise appended tail."""
    assert isinstance(horizon, int) and horizon > 0, "H must be a positive integer"
    assert isinstance(execution_steps, int) and 0 < execution_steps <= horizon, "invalid K"
    assert horizon % execution_steps == 0, "H must be divisible by K"
    groups = horizon // execution_steps
    assert isinstance(noise_levels, int) and noise_levels >= groups, "too few noise levels"
    group = torch.arange(horizon, device=device) // execution_steps
    dtype = torch.float32 if dtype is None else dtype
    pre = ((noise_levels * (group + 1)) // groups).to(dtype) / noise_levels
    post = ((noise_levels * group) // groups).to(dtype) / noise_levels
    return pre, post


def position_ids(prefix_steps=8, horizon=8, *, device=None):
    """Preserve checkpoint future IDs 0..H-1; place clean history at H..H+P-1."""
    assert prefix_steps >= 0 and horizon > 0, "invalid prefix/future lengths"
    return torch.cat((torch.arange(horizon, horizon + prefix_steps, device=device),
                      torch.arange(horizon, device=device)))


def interpolate(clean, noise, sigma):
    assert clean.shape == noise.shape and clean.ndim == 3, "clean/noise must match [B,H,D]"
    assert sigma.shape in ((clean.shape[1],), clean.shape[:2]), "sigma must be [H] or [B,H]"
    sigma = sigma.to(device=clean.device, dtype=clean.dtype).unsqueeze(-1)
    return (1 - sigma) * clean + sigma * noise


def euler_step(future, velocity, sigma_pre, sigma_post):
    assert future.shape == velocity.shape and future.ndim == 3, "future/velocity must match [B,H,D]"
    assert sigma_pre.shape == sigma_post.shape, "pre/post sigma shapes must match"
    assert sigma_pre.shape in ((future.shape[1],), future.shape[:2]), "sigma must be [H] or [B,H]"
    delta = (sigma_post - sigma_pre).to(device=future.device, dtype=future.dtype).unsqueeze(-1)
    return future + delta * velocity


def roll_buffer(post_buffer, fresh_noise, execution_steps=4):
    """Consume exactly K executed commands; retain the *unclipped* noisy suffix."""
    assert post_buffer.ndim == 3 and fresh_noise.ndim == 3, "buffers must be [B,H,D]"
    batch, horizon, dim = post_buffer.shape
    assert 0 < execution_steps <= horizon and horizon % execution_steps == 0, "unsupported K"
    assert fresh_noise.shape == (batch, execution_steps, dim), "fresh noise must be [B,K,D]"
    assert fresh_noise.device == post_buffer.device and fresh_noise.dtype == post_buffer.dtype, "buffer dtype/device mismatch"
    return torch.cat((post_buffer[:, execution_steps:], fresh_noise), dim=1)
