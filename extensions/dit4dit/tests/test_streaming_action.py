"""CPU-only action SDP checks; no checkpoints, backbone, simulator, or training."""

import importlib
from pathlib import Path
import sys
import types

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
action_module = importlib.import_module("DiT4DiT.model.modules.action_model.ActionDiT")
from DiT4DiT.model.modules.action_model.flow_matching_head.cross_attention_dit import (
    AdaLayerNorm, DiT, TimestepEncoder,
)
from DiT4DiT.model.modules.action_model.streaming import (
    euler_step, interpolate, noise_ladder, position_ids, roll_buffer,
)


def make_head(monkeypatch, streaming=True, **options):
    monkeypatch.setitem(action_module.DiTConfig, "test", {
        "input_embedding_dim": 8, "attention_head_dim": 4, "num_attention_heads": 2,
    })
    config = OmegaConf.create({"framework": {"action_model": {
        "action_model_type": "test", "hidden_size": 12,
        "diffusion_model_cfg": {"output_dim": 8, "cross_attention_dim": 6,
                                "num_layers": 2, "dropout": 0.0, "final_dropout": False,
                                "positional_embeddings": None, "interleave_self_attention": True},
        "state_dim": 16, "action_dim": 8, "future_action_window_size": 7,
        "action_horizon": 8, "num_inference_timesteps": 4, "add_pos_embed": True,
        "max_seq_len": 32, "noise_beta_alpha": 1.5, "noise_beta_beta": 1.0,
        "noise_s": 0.999, "num_timestep_buckets": 1000,
        "streaming": streaming, **options,
    }}})
    return action_module.FlowmatchingActionHead(config).eval()


def inputs(batch=2, dtype=torch.float32):
    return (torch.randn(batch, 3, 6, dtype=dtype), torch.randn(batch, 8, 8, dtype=dtype),
            torch.randn(batch, 1, 16, dtype=dtype))


def test_ladder_oracle_and_rolling_alignment():
    pre, post = noise_ladder()
    torch.testing.assert_close(pre, torch.tensor([0.5] * 4 + [1.0] * 4))
    torch.testing.assert_close(post, torch.tensor([0.0] * 4 + [0.5] * 4))
    torch.testing.assert_close(position_ids(), torch.tensor(list(range(8, 16)) + list(range(8))))
    clean, noise = torch.randn(2, 8, 8), torch.randn(2, 8, 8)
    noisy = interpolate(clean, noise, pre)
    updated = euler_step(noisy, noise - clean, pre, post)
    torch.testing.assert_close(updated, interpolate(clean, noise, post))
    fresh_noise = torch.randn(2, 4, 8)
    rolled = roll_buffer(updated, fresh_noise)
    torch.testing.assert_close(rolled[:, :4], updated[:, 4:])
    torch.testing.assert_close(rolled[:, 4:], fresh_noise)
    next_clean = torch.cat((clean[:, 4:], torch.randn(2, 4, 8)), dim=1)
    next_noise = torch.cat((noise[:, 4:], fresh_noise), dim=1)
    torch.testing.assert_close(rolled, interpolate(next_clean, next_noise, pre))
    next_updated = euler_step(rolled, next_noise - next_clean, pre, post)
    torch.testing.assert_close(next_updated[:, :4], clean[:, 4:])


@pytest.mark.parametrize("k", [0, 3, 9])
def test_nondivisor_schedule_rejected(k):
    with pytest.raises(AssertionError):
        noise_ladder(execution_steps=k)


def test_action_encoder_tokenwise_parity():
    encoder = action_module.ActionEncoder(8, 16)
    actions = torch.randn(2, 8, 8)
    times = torch.tensor([125, 750])
    scalar = encoder(actions, times)
    tokens = encoder(actions, times[:, None].expand(-1, 8))
    torch.testing.assert_close(scalar, tokens, rtol=0, atol=0)
    variable = times[:, None].expand(-1, 8).clone()
    variable[:, :4] = 0
    assert not torch.equal(encoder(actions, variable)[:, :4], scalar[:, :4])
    with pytest.raises(ValueError):
        encoder(actions, torch.zeros(2, 7))


def test_timestep_adaln_and_final_tokenwise_parity():
    times = torch.tensor([0, 500])
    encoder = TimestepEncoder(8)
    scalar = encoder(times)
    tokens = encoder(times[:, None].expand(-1, 9))
    torch.testing.assert_close(tokens, scalar[:, None].expand(-1, 9, -1))
    norm = AdaLayerNorm(8)
    x = torch.randn(2, 9, 8)
    scale, shift = norm.linear(norm.silu(scalar)).chunk(2, dim=1)
    reference = norm.norm(x) * (1 + scale[:, None]) + shift[:, None]
    torch.testing.assert_close(norm(x, scalar), reference, rtol=0, atol=0)
    torch.testing.assert_close(norm(x, tokens), reference)
    model = DiT(num_attention_heads=2, attention_head_dim=4, output_dim=8,
                num_layers=2, dropout=0.0, final_dropout=False,
                positional_embeddings=None, cross_attention_dim=6,
                interleave_self_attention=True).eval()
    vl = torch.randn(2, 3, 6)
    torch.testing.assert_close(model(x, vl, times), model(x, vl, times[:, None].expand(-1, 9)))
    with pytest.raises(AssertionError, match="include state"):
        model(x, vl, times[:, None].expand(-1, 8))


def legacy_model_forward(model, hidden_states, vl, times, encoder_attention_mask=None):
    """Independent original [B] timestep/AdaLN/final formulas."""
    temb = model.timestep_encoder.timestep_embedder(
        model.timestep_encoder.time_proj(times).to(next(model.parameters()).dtype))
    for index, block in enumerate(model.transformer_blocks):
        scale, shift = block.norm1.linear(block.norm1.silu(temb)).chunk(2, dim=1)
        normed = block.norm1.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if block.pos_embed is not None:
            normed = block.pos_embed(normed)
        self_attention = index % 2 == 1 and model.config.interleave_self_attention
        attended = block.attn1(normed, encoder_hidden_states=None if self_attention else vl,
                               attention_mask=None if self_attention else encoder_attention_mask)
        if block.final_dropout:
            attended = block.final_dropout(attended)
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + block.ff(block.norm3(hidden_states))
    shift, scale = model.proj_out_1(torch.nn.functional.silu(temb)).chunk(2, dim=1)
    return model.proj_out_2(model.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None])


def test_disabled_head_checkpoint_and_legacy_parity(monkeypatch):
    head = make_head(monkeypatch, streaming=False)
    streaming = make_head(monkeypatch)
    assert {k: v.shape for k, v in head.state_dict().items()} == {
        k: v.shape for k, v in streaming.state_dict().items()}
    streaming.load_state_dict(head.state_dict(), strict=True)
    vl, actions, state = inputs()
    times = torch.tensor([250, 750])
    features = head.action_encoder(actions, times) + head.position_embedding(torch.arange(8))[None]
    hidden = torch.cat((head.state_encoder(state), features), dim=1)
    torch.testing.assert_close(head.model(hidden, vl, times), legacy_model_forward(head.model, hidden, vl, times), rtol=0, atol=0)

    # Full disabled loss and ordinary Euler inference reproduce the original path.
    mask = torch.ones_like(actions)
    torch.manual_seed(41)
    noise = torch.randn_like(actions)
    time = head.sample_time(2, actions.device, actions.dtype)
    noisy = (1 - time[:, None, None]) * actions + time[:, None, None] * noise
    discrete = (time * head.num_timestep_buckets).long()
    features = head.action_encoder(noisy, discrete) + head.position_embedding(torch.arange(8))[None]
    hidden = torch.cat((head.state_encoder(state), features), dim=1)
    predicted = head.action_decoder(legacy_model_forward(head.model, hidden, vl, discrete))[:, -8:]
    expected = ((predicted - (noise - actions)).square() * mask).sum() / mask.sum()
    torch.manual_seed(41)
    torch.testing.assert_close(head(vl, actions, mask, state), expected, rtol=0, atol=0)
    torch.manual_seed(42)
    expected = torch.randn_like(actions)
    for step in range(4):
        discrete = torch.full((2,), int((1 - step / 4) * head.num_timestep_buckets))
        features = head.action_encoder(expected, discrete) + head.position_embedding(torch.arange(8))[None]
        hidden = torch.cat((head.state_encoder(state), features), dim=1)
        predicted = head.action_decoder(legacy_model_forward(head.model, hidden, vl, discrete))[:, -8:]
        expected = expected - predicted / 4
    torch.manual_seed(42)
    torch.testing.assert_close(head.predict_action(vl, state), expected, rtol=0, atol=0)


@pytest.mark.parametrize("ladder_probability", [0.0, 1.0])
def test_training_clean_prefix_mask_target_and_times(monkeypatch, ladder_probability):
    head = make_head(monkeypatch, streaming_ladder_probability=ladder_probability)
    vl, prefix, state = inputs()
    actions = torch.cat((prefix, torch.randn_like(prefix)), dim=1)
    mask = torch.ones_like(actions, dtype=torch.bool)
    mask[:, 0] = False
    mask[:, -1] = False
    mask[:, :, 7] = False
    actions[~mask] = float("nan")
    saved = []
    def zero_velocity(self, vl, clean_prefix, future, times, state, encoder_attention_mask):
        saved.append((clean_prefix.clone(), future.clone(), times.clone()))
        return torch.zeros_like(future)
    head._streaming_velocity = types.MethodType(zero_velocity, head)
    head.sample_time = lambda batch, device, dtype: torch.full((batch,), 0.75, device=device, dtype=dtype)
    torch.manual_seed(43)
    loss = head(vl, actions, mask, state)
    clean_prefix, future, times = saved[-1]
    assert torch.isfinite(loss) and torch.isfinite(future).all()
    assert torch.count_nonzero(clean_prefix[:, 0]) == 0
    torch.testing.assert_close(clean_prefix[:, 1:, :7], prefix[:, 1:, :7])
    assert torch.count_nonzero(future[:, -1]) == 0
    assert torch.count_nonzero(future[:, :, 7]) == 0
    target_times = noise_ladder()[0][None].expand(2, -1) if ladder_probability else torch.full((2, 8), 0.75)
    torch.testing.assert_close(times, target_times)
    clean = torch.where(mask[:, 8:], actions[:, 8:], 0)
    noise = (future - (1 - times[:, :, None]) * clean) / times[:, :, None]
    expected = torch.where(mask[:, 8:], (noise - clean).square(), 0).sum() / mask[:, 8:].sum()
    torch.testing.assert_close(loss, expected)
    # Alter every prefix while keeping the future/noise fixed: prefix has no direct target loss.
    actions[:, :8] = 100
    torch.manual_seed(43)
    torch.testing.assert_close(head(vl, actions, mask, state), loss)
    empty_mask = torch.zeros_like(mask)
    assert head(vl, actions, empty_mask, state).item() == 0


def test_warm_start_oracle_and_stateless_rolling(monkeypatch):
    head = make_head(monkeypatch)
    vl, prefix, state = inputs()
    prefix[:, :, 7] = 99
    original_prefix = prefix.clone()
    clean = torch.randn_like(prefix)
    clean[:, :, 7] = 0
    calls = []
    def oracle(self, vl, clean_prefix, future, times, state, encoder_attention_mask):
        assert torch.count_nonzero(clean_prefix[:, :, 7]) == 0
        torch.testing.assert_close(clean_prefix[:, :, :7], prefix[:, :, :7])
        calls.append((future.clone(), times.clone()))
        return (future - clean) / times[:, :, None]
    head._streaming_velocity = types.MethodType(oracle, head)
    result = head.predict_streaming_action(vl, prefix=prefix, state=state,
                                          generator=torch.Generator().manual_seed(7))
    assert len(calls) == 5
    for (_, times), time in zip(calls[:4], [1.0, 0.75, 0.5, 0.25]):
        torch.testing.assert_close(times, torch.full((2, 8), time))
    torch.testing.assert_close(calls[4][1], noise_ladder()[0][None].expand(2, -1))
    torch.testing.assert_close(result["normalized_actions"], clean[:, :4])
    assert result["action_buffer"].shape == (2, 8, 8)
    buffer = result["action_buffer"].clone()
    clean = torch.cat((clean[:, 4:], torch.randn_like(clean[:, :4])), dim=1)
    clean[:, :, 7] = 0
    calls.clear()
    second = head.predict_streaming_action(vl, prefix=prefix, state=state, buffer=buffer,
                                          generator=torch.Generator().manual_seed(8))
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0][:, :4], buffer[:, 4:])
    torch.testing.assert_close(second["normalized_actions"], clean[:, :4])
    torch.testing.assert_close(buffer, result["action_buffer"], rtol=0, atol=0)
    torch.testing.assert_close(prefix, original_prefix, rtol=0, atol=0)
    assert torch.count_nonzero(second["action_buffer"][:, :, 7]) == 0
    assert not any(name in vars(head) for name in ("buffer", "future_buffer", "action_buffer"))
    # Executable commands must not alias the caller-carried noisy buffer.
    executable = second["normalized_actions"].clone()
    second["action_buffer"].zero_()
    torch.testing.assert_close(second["normalized_actions"], executable)


def test_streaming_velocity_state_times_positions_and_prefix_clamping(monkeypatch):
    head = make_head(monkeypatch)
    vl, prefix, state = inputs()
    seen_times, seen_positions, seen_actions = [], [], []
    handles = [
        head.model.register_forward_pre_hook(lambda module, args, kwargs: seen_times.append(kwargs["timestep"].clone()), with_kwargs=True),
        head.position_embedding.register_forward_pre_hook(lambda module, args: seen_positions.append(args[0].clone())),
        head.action_encoder.register_forward_pre_hook(lambda module, args: seen_actions.append(args[0].clone())),
    ]
    result = head.predict_streaming_action(vl, prefix=prefix, state=state)
    for handle in handles:
        handle.remove()
    assert len(seen_times) == 5
    for times, positions, actions in zip(seen_times, seen_positions, seen_actions):
        assert times.shape == (2, 17)
        assert torch.count_nonzero(times[:, :9]) == 0
        torch.testing.assert_close(positions, position_ids())
        torch.testing.assert_close(actions[:, :8, :7], prefix[:, :, :7])
        assert torch.count_nonzero(actions[:, :, 7]) == 0
    assert torch.isfinite(result["action_buffer"]).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_streaming_model_dtype_and_finite_loss(monkeypatch, dtype):
    head = make_head(monkeypatch).to(dtype=dtype)
    vl, prefix, state = inputs(dtype=dtype)
    actions = torch.cat((prefix, torch.randn_like(prefix)), dim=1)
    mask = torch.ones_like(actions, dtype=torch.bool)
    assert torch.isfinite(head(vl, actions, mask, state))
    times = []
    handle = head.model.register_forward_pre_hook(
        lambda module, args, kwargs: times.append(kwargs["timestep"].clone()), with_kwargs=True)
    result = head.predict_streaming_action(vl, prefix=prefix, state=state)
    handle.remove()
    for token_times, bucket in zip(times[:4], [1000, 750, 500, 250]):
        torch.testing.assert_close(token_times[:, -8:], torch.full((2, 8), bucket))
    assert result["normalized_actions"].dtype == result["action_buffer"].dtype == dtype
    assert torch.isfinite(result["action_buffer"]).all()
    assert torch.count_nonzero(result["action_buffer"][:, :, 7]) == 0


def test_streaming_options_and_shape_errors(monkeypatch):
    legacy = make_head(monkeypatch, streaming=False)
    del legacy.full_config.framework.action_model["streaming"]
    assert not action_module.FlowmatchingActionHead(legacy.full_config).streaming_enabled
    assert not make_head(monkeypatch, streaming={"enabled": False}).streaming_enabled
    head = make_head(monkeypatch, streaming={"enabled": True, "ladder_probability": 0.6})
    assert head.streaming_ladder_probability == 0.6
    vl, prefix, state = inputs()
    for name, value in [("execution_steps", 3), ("prefix_steps", 16), ("warm_start_steps", 0),
                        ("ladder_probability", 1.1), ("noise_levels", 31)]:
        with pytest.raises(AssertionError):
            make_head(monkeypatch, **{"streaming_" + name: value})
    with pytest.raises(AssertionError, match="full post-update"):
        head.predict_streaming_action(vl, prefix=prefix, buffer=prefix[:, :4], state=state)
    with pytest.raises(AssertionError, match="prefix"):
        head.predict_streaming_action(vl, prefix=prefix[:, :4], state=state)
    with pytest.raises(AssertionError, match="16,8"):
        head(vl, prefix, torch.ones_like(prefix), state)
