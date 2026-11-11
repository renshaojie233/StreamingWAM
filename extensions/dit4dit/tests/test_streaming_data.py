import numpy as np
import torch

from DiT4DiT.dataloader.gr00t_lerobot.streaming_data import (
    ACTION_DELTAS, VIDEO_DELTAS, encode_libero_state, pack_rgb_views, streaming_enabled, temporal_validity,
)


def test_opt_in_keeps_legacy_and_accepts_boolean_or_mapping():
    assert not streaming_enabled(None)
    assert not streaming_enabled({})
    assert not streaming_enabled({"streaming": False})
    assert not streaming_enabled({"streaming": {"enabled": False}})
    assert streaming_enabled({"streaming": True})
    assert streaming_enabled({"streaming": {"enabled": True}})


def test_exact_temporal_contract_and_episode_boundaries():
    assert VIDEO_DELTAS == (-4, -3, -2, -1, 0, 2, 4, 6, 8)
    assert ACTION_DELTAS == tuple(range(-8, 8))
    valid = temporal_validity(0, 100, ACTION_DELTAS)
    assert not valid[:8].any() and valid[8:].all()
    valid = temporal_validity(99, 100, ACTION_DELTAS)
    assert valid[:9].all() and not valid[9:].any()
    assert temporal_validity(8, 100, ACTION_DELTAS).all()


def test_train_infer_rgb_view_packing_has_identical_values():
    primary = np.full((256, 256, 3), 25, dtype=np.uint8)
    wrist = np.full((256, 256, 3), 210, dtype=np.uint8)
    array_input = pack_rgb_views(primary, wrist)
    tensor_input = pack_rgb_views(torch.from_numpy(primary).permute(2, 0, 1),
                                  torch.from_numpy(wrist).permute(2, 0, 1))
    torch.testing.assert_close(array_input, tensor_input, rtol=0, atol=0)
    assert array_input.shape == (3, 224, 448)
    torch.testing.assert_close(array_input[:, :, :224], torch.full((3, 224, 224), 25 / 255))
    torch.testing.assert_close(array_input[:, :, 224:], torch.full((3, 224, 224), 210 / 255))


def test_state_order_and_storage_match_training():
    raw = np.linspace(-1, 1, 8, dtype=np.float32)
    expected = np.stack((np.sin(raw), np.cos(raw)), axis=-1).reshape(1, 16).astype(np.float16)
    np.testing.assert_array_equal(encode_libero_state(raw), expected)
