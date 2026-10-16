from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import torch

from .contracts import ACTION_DIM
from .contracts import ACTION_PREFIX_STEPS
from .contracts import CHECKPOINT_NAME
from .contracts import CHECKPOINT_SHA256
from .contracts import FULL_ACTION_HORIZON
from .contracts import FUTURE_ACTION_HORIZON
from .contracts import IMAGE_HISTORY_FRAMES
from .contracts import IMAGE_HISTORY_SHAPE
from .contracts import MODEL_VIDEO_FRAMES
from .contracts import ROLLING_DENOISE_SUBSTEPS
from .contracts import SDP_NOISE_LEVELS
from .contracts import STATS_SHA256
from .contracts import TEXT_EMBED_SHA256
from .contracts import TRAINED_REPLAN_K
from .contracts import WARM_START_DENOISE_STEPS
from .contracts import prompt_cache_name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StreamingWAMPolicy:
    """One in-process StreamingWAM rolling session."""

    def __init__(self, asset_root: Path, *, base_model_root: Path | None = None) -> None:
        self.asset_root = Path(asset_root).resolve()
        checkpoint = self.asset_root / CHECKPOINT_NAME
        release_root = Path(__file__).resolve().parents[1]
        config_path = release_root / "assets/streamingwam_realworld_model.yaml"
        stats_path = self.asset_root / "train_split_zscore_stats.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        for path, expected in (
            (checkpoint, CHECKPOINT_SHA256),
            (stats_path, STATS_SHA256),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"SHA-256 mismatch for {path.name}: {actual}")

        if base_model_root is not None:
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(Path(base_model_root).resolve())
        os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        self.state_mean = np.asarray(stats["state"]["default"]["global_mean"], dtype=np.float32)
        self.state_std = np.asarray(stats["state"]["default"]["global_std"], dtype=np.float32)
        self.action_mean = np.asarray(stats["action"]["default"]["global_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action"]["default"]["global_std"], dtype=np.float32)
        for name, value in (
            ("state_mean", self.state_mean),
            ("state_std", self.state_std),
            ("action_mean", self.action_mean),
            ("action_std", self.action_std),
        ):
            if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise RuntimeError(f"invalid {name}")
        if np.any(self.state_std <= 0) or np.any(self.action_std <= 0):
            raise RuntimeError("normalization standard deviations must be positive")

        cfg = OmegaConf.load(config_path)
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        if not isinstance(model_cfg, dict):
            raise RuntimeError("config.model must resolve to a mapping")
        target = model_cfg.pop("_target_", None)
        if target != "fastwam.runtime.create_fastwam_joint":
            raise RuntimeError(f"unexpected model target: {target}")
        if int(model_cfg.get("num_history_frames", -1)) != IMAGE_HISTORY_FRAMES:
            raise RuntimeError("checkpoint must use five history frames")
        model_cfg["load_text_encoder"] = False
        model_cfg["skip_dit_load_from_pretrain"] = True
        model_cfg["action_dit_pretrained_path"] = None
        model_cfg["mot_checkpoint_mixed_attn"] = False
        model_cfg["video_dit_config"]["use_gradient_checkpointing"] = False
        model_cfg["action_dit_config"]["use_gradient_checkpointing"] = False

        from fastwam.runtime import create_fastwam_joint

        self.model = create_fastwam_joint(**model_cfg, model_dtype=torch.bfloat16, device="cuda")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if set(payload) != {"mot", "step", "torch_dtype", "proprio_encoder"}:
            raise RuntimeError(f"unexpected checkpoint keys: {sorted(payload)}")
        if int(payload["step"]) != 30_000:
            raise RuntimeError(f"unexpected checkpoint step: {payload['step']}")
        self.model.mot.load_state_dict(payload["mot"], strict=True)
        self.model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        del payload
        self.model.eval()

        tasks_path = release_root / "assets/tasks.json"
        self.tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        self.aliases = {name: name for name in self.tasks}
        for name, aliases in self.tasks.items():
            for alias in aliases:
                self.aliases[str(alias)] = name
        self.contexts = self._load_contexts()
        self.action_buffer: torch.Tensor | None = None
        self.rolling_step_idx = 0

    def _load_contexts(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        cache = self.asset_root / "text_embeds_cache"
        for task in sorted(self.tasks):
            filename = prompt_cache_name(task)
            path = cache / filename
            expected = TEXT_EMBED_SHA256.get(filename)
            if expected is None or not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"missing or corrupt cached prompt embedding for {task}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            context = payload["context"].to(torch.bfloat16)
            mask = payload["mask"].to(torch.bool)
            if context.shape != (128, 4096) or mask.shape != (128,):
                raise RuntimeError(f"invalid cached context for {task}")
            context = context.clone()
            context[~mask] = 0
            result[task] = (context, torch.ones_like(mask))
        return result

    def reset(self) -> None:
        self.action_buffer = None
        self.rolling_step_idx = 0

    @torch.inference_mode()
    def infer(
        self,
        task: str,
        image_history: np.ndarray,
        state: np.ndarray,
        action_prefix: np.ndarray,
        *,
        replan_k: int,
        seed: int,
    ) -> np.ndarray:
        if replan_k not in TRAINED_REPLAN_K:
            raise ValueError(f"replan_k must be one of {TRAINED_REPLAN_K}")
        canonical = self.aliases.get(task.strip())
        if canonical is None:
            raise ValueError(f"unsupported task {task!r}; choose from {sorted(self.tasks)}")
        image_history = np.ascontiguousarray(image_history, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        action_prefix = np.asarray(action_prefix, dtype=np.float32)
        if image_history.shape != IMAGE_HISTORY_SHAPE or not np.isfinite(image_history).all():
            raise ValueError(f"image_history must be finite {IMAGE_HISTORY_SHAPE}")
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ValueError("state must be finite shape (8,)")
        if action_prefix.shape != (ACTION_PREFIX_STEPS, ACTION_DIM) or not np.isfinite(action_prefix).all():
            raise ValueError("action_prefix must be finite shape (16,8)")

        normalized_state = np.clip((state - self.state_mean) / (self.state_std + 1e-8), -5.0, 5.0)
        normalized_prefix = (action_prefix - self.action_mean) / (self.action_std + 1e-8)
        context, mask = self.contexts[canonical]
        output = self.model.infer_action_sdp_rolling_step(
            prompt=None,
            input_image=torch.from_numpy(image_history),
            action_horizon=FULL_ACTION_HORIZON,
            num_video_frames=MODEL_VIDEO_FRAMES,
            action_prefix=torch.from_numpy(normalized_prefix),
            action_buffer=self.action_buffer,
            action_steps=replan_k,
            sdp_num_noise_levels=SDP_NOISE_LEVELS,
            sdp_warm_start=True,
            sdp_warm_start_inference_steps=WARM_START_DENOISE_STEPS,
            sdp_action_denoise_substeps=ROLLING_DENOISE_SUBSTEPS,
            rolling_step_idx=self.rolling_step_idx,
            proprio=torch.from_numpy(normalized_state),
            context=context,
            context_mask=mask,
            num_inference_steps=1,
            seed=int(seed),
            rand_device="cpu",
            tiled=False,
            return_video=False,
        )
        normalized = output["action"].to(device="cpu", dtype=torch.float32).numpy()
        if normalized.shape != (FULL_ACTION_HORIZON, ACTION_DIM):
            raise RuntimeError(f"invalid model output {normalized.shape}")
        future = normalized[ACTION_PREFIX_STEPS:] * self.action_std + self.action_mean
        if future.shape != (FUTURE_ACTION_HORIZON, ACTION_DIM) or not np.isfinite(future).all():
            raise RuntimeError("non-finite or malformed future actions")
        self.action_buffer = output["action_buffer"]
        self.rolling_step_idx += 1
        return np.ascontiguousarray(future, dtype=np.float32)
