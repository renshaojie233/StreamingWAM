#!/usr/bin/env python3
"""Continued DiT4DiT training with BF16 parameters and ZeRO-1 optimizer state.

Both recipes start from the upstream LIBERO checkpoint. A stop at 10,000 updates
uses the original 20,000-update cosine schedule, as in the released weights.
"""

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bootstrap import activate
activate("streaming")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=("joint", "streaming"))
    p.add_argument("--config", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True, help="Upstream DiT4DiT LIBERO .pt initialization")
    p.add_argument("--checkpoint-stats", type=Path)
    p.add_argument("--cosmos-path", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frame-cache", type=Path)
    p.add_argument("--max-steps", type=int, help="Cosine schedule length; leave at 20,000 for this release")
    p.add_argument("--stop-after", type=int, default=10000, help="Stop without changing the cosine schedule")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--grad-accum", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--lr-preset", choices=("official", "low-backbone"), default="official")
    p.add_argument("--zero-config", type=Path, default=ROOT / "scripts/zero1_streaming.json")
    p.add_argument("--hash-checkpoint", action="store_true")
    p.add_argument("--checkpoint-hash-cache", type=Path)
    p.add_argument("--preflight-only", action="store_true", help="Validate paths/config; do not allocate a model")
    p.add_argument("--resume", type=Path, help="Complete native training state, not inference weights")
    return p


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"Expected file: {path}")
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_manifest():
    """Hash source, not checkpoints, generated results, caches, or Git internals."""
    directories = {"vendor", "scripts", "tests", "configs", "evaluation"}
    extensions = {".py", ".yaml", ".yml", ".json", ".sh", ".toml", ".txt", ".md"}
    root_files = {"bootstrap.py", "release.json", "README.md", "requirements.txt"}
    result = {}
    for directory, subdirs, files in os.walk(ROOT):
        rel = Path(directory).relative_to(ROOT)
        subdirs[:] = sorted(d for d in subdirs if not d.startswith(".") and d != "__pycache__"
                           and (rel.parts or d in directories))
        for name in sorted(files):
            if not rel.parts and name not in root_files:
                continue
            path = Path(directory) / name
            if path.suffix in extensions or name == "LICENSE":
                result[str(path.relative_to(ROOT))] = sha256_file(path)
    return result


def resolve_streaming_config(args):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    overrides = {
        "output_dir": str(args.output.resolve()),
        "trainer.pretrained_checkpoint": str(args.checkpoint.resolve()),
        "trainer.lr_preset": args.lr_preset,
        "trainer.learning_rate.backbone_interface": 1e-6 if args.lr_preset == "low-backbone" else 1e-5,
        "trainer.learning_rate.action_model": 1e-4,
    }
    for key, value in (
        ("framework.cosmos25.base_model", str(args.cosmos_path.resolve()) if args.cosmos_path else None),
        ("datasets.vla_data.data_root_dir", str(args.data_root.resolve()) if args.data_root else None),
        ("trainer.max_train_steps", args.max_steps),
        ("datasets.vla_data.per_device_batch_size", args.batch_size),
        ("trainer.gradient_accumulation_steps", args.grad_accum),
        ("trainer.num_workers", args.num_workers), ("seed", args.seed),
    ):
        if value is not None:
            overrides[key] = value
    if args.max_steps is not None:
        overrides["provenance.schedule_overridden"] = True
    if args.frame_cache:
        overrides["datasets.vla_data.streaming_frame_cache_root"] = str(args.frame_cache.resolve())
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    d, a, t, c = cfg.datasets.vla_data, cfg.framework.action_model, cfg.trainer, cfg.framework.cosmos25
    checks = {
        "dataset streaming must be literal true": d.streaming is True,
        "state/action dimensions must be 16/8": (d.max_state_dim, d.max_action_dim, a.state_dim, a.action_dim) == (16, 8, 16, 8),
        "eight past, eight future actions required": (a.past_action_window_size, a.future_action_window_size, a.action_horizon) == (8, 7, 8),
        "explicit non-resampled nine-frame timeline required": list(d.video_delta_indices) == [-4, -3, -2, -1, 0, 2, 4, 6, 8] and d.action_video_freq_ratio == 1,
        "explicit -8..7 action timeline required": list(d.action_delta_indices) == list(range(-8, 8)),
        "joint video FM training required": c.training == "joint" and c.future_loss_type == "flow_matching" and c.trainable is True,
        "BF16 and explicit checkpointing required": t.mixed_precision == "bf16" and isinstance(t.enable_gradient_checkpointing, bool),
        "dataset must use direct LeRobot builder": d.dataset_py == "lerobot_datasets",
        "LeRobot v3 must be explicit": d.get("lerobot_version") == "v3.0",
        "workers cannot be negative": t.num_workers >= 0,
    }
    streaming = a.streaming
    def option(name):
        return a.get("streaming_" + name) if isinstance(streaming, bool) else streaming.get(name)
    checks["action streaming must be enabled"] = streaming is True or (
        not isinstance(streaming, bool) and streaming.get("enabled") is True)
    checks["P8/K4/W4/L32 and proposed 80/20 mix required"] = tuple(
        option(k) for k in ("prefix_steps", "execution_steps", "warm_start_steps", "noise_levels", "ladder_probability")) == (8, 4, 4, 32, 0.8)
    for name in ("max_train_steps", "save_interval", "logging_frequency", "gradient_accumulation_steps"):
        checks[name + " must be positive"] = isinstance(t[name], int) and t[name] > 0
    checks["batch size must be positive"] = isinstance(d.per_device_batch_size, int) and d.per_device_batch_size > 0
    checks["future video loss must contribute"] = math.isfinite(t.loss_scale.future_video) and t.loss_scale.future_video > 0
    for description, passed in checks.items():
        if not passed:
            raise ValueError(description)
    zero = json.loads(args.zero_config.read_text())
    if zero.get("zero_optimization", {}).get("stage") != 1 or zero.get("bf16", {}).get("enabled") is not True:
        raise ValueError("Only BF16 DeepSpeed ZeRO stage 1 is supported")
    if zero.get("gradient_clipping") != t.gradient_clipping:
        raise ValueError("Trainer and DeepSpeed gradient_clipping must agree")
    return cfg


def validation_context(args, cfg):
    from omegaconf import OmegaConf
    manifest = source_manifest()
    stats_path = args.checkpoint_stats
    if stats_path is None:
        candidates = [args.checkpoint.parent / "dataset_statistics.json",
                      args.checkpoint.parent.parent / "dataset_statistics.json"]
        stats_path = next((p for p in candidates if p.is_file()), None)
    if stats_path is None:
        raise FileNotFoundError("Official dataset_statistics.json is required; use --checkpoint-stats")
    args.checkpoint_stats = stats_path.resolve(strict=True)
    checkpoint_identity = identity(args.checkpoint)
    if checkpoint_identity["size"] != cfg.provenance.official_checkpoint_bytes:
        raise ValueError("Official checkpoint size mismatch: transfer may be incomplete; training is blocked")
    context = {
        "schema_version": 1,
        "source_sha256": digest_json(manifest),
        "config_sha256": digest_json(OmegaConf.to_container(cfg, resolve=True)),
        "zero_config_sha256": sha256_file(args.zero_config),
        "checkpoint": checkpoint_identity,
        "checkpoint_stats_sha256": sha256_file(args.checkpoint_stats),
        "world_size": 4, "mixed_precision": "bf16", "zero_stage": 1,
        "stop_after": args.stop_after,
    }
    return context, manifest




def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def load_official_weights(model, path):
    import torch
    kwargs = {"map_location": "cpu"}
    supported = inspect.signature(torch.load).parameters
    if "weights_only" in supported:
        kwargs["weights_only"] = True
    if "mmap" in supported:
        kwargs["mmap"] = True
    try:
        state = torch.load(path, **kwargs)
    except RuntimeError as exc:
        # Legacy non-ZIP files cannot be memory-mapped. Never downgrade safe loading.
        if not kwargs.get("mmap") or "mmap" not in str(exc).lower():
            raise
        kwargs.pop("mmap")
        state = torch.load(path, **kwargs)
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise ValueError("Official checkpoint must be a raw, complete tensor state_dict; no key rewriting")
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    shapes = [k for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape]
    if missing or unexpected or shapes:
        raise RuntimeError(f"Strict transplant rejected: missing={missing}, unexpected={unexpected}, shape_mismatch={shapes}")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Strict transplant failed: {result}")
    summary = {"missing_keys": [], "unexpected_keys": [], "tensor_count": len(state), "torch_load_options": kwargs}
    del state, expected
    gc.collect()
    return summary


def freeze_and_group(model, cfg):
    import torch
    extractor = model.backbone_interface.extractor
    model.requires_grad_(False)
    extractor.transformer.requires_grad_(True)
    model.action_model.requires_grad_(True)
    extractor.vae.requires_grad_(False).eval()
    extractor.text_encoder.requires_grad_(False).eval()
    extractor.trainable = True
    if cfg.trainer.enable_gradient_checkpointing:
        extractor.transformer.enable_gradient_checkpointing()
    else:
        extractor.transformer.disable_gradient_checkpointing()
    if bool(getattr(extractor.transformer, "is_gradient_checkpointing", False)) != cfg.trainer.enable_gradient_checkpointing:
        raise RuntimeError("Cosmos gradient checkpointing does not match config")
    # DeepSpeed BF16 converts the module anyway; do it explicitly and validate.
    model.to(dtype=torch.bfloat16)
    groups = [
        {"name": "backbone_interface", "params": list(extractor.transformer.parameters()),
         "lr": cfg.trainer.learning_rate.backbone_interface},
        {"name": "action_model", "params": list(model.action_model.parameters()),
         "lr": cfg.trainer.learning_rate.action_model},
    ]
    intended = {id(p) for group in groups for p in group["params"]}
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if intended != actual or sum(len(g["params"]) for g in groups) != len(intended):
        raise RuntimeError("Optimizer coverage mismatch, duplicate parameters, or unintended trainability")
    for group in groups:
        if not group["params"] or not all(p.requires_grad and p.dtype == torch.bfloat16 for p in group["params"]):
            raise RuntimeError(f"Empty, frozen, or non-BF16 trainable group: {group['name']}")
    def detached_features(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]) or inputs[0].requires_grad:
            raise RuntimeError("Action features must remain detached from Cosmos")
    model.action_model.register_forward_pre_hook(detached_features)
    return groups


class GradientMonitor:
    """Track each backward and its accumulation window before ZeRO clears .grad."""

    def __init__(self, groups, accelerator):
        import torch
        self.accelerator = accelerator
        # Columns: observed gradients, any nonfinite, any nonzero, valid targets.
        self.flags = torch.zeros((len(groups), 4), device=accelerator.device, dtype=torch.int32)
        self.window_flags = torch.zeros_like(self.flags)
        self._optimizer_step_seen = False
        self.active = True
        self.handles = []
        for i, group in enumerate(groups):
            for parameter in group["params"]:
                def observe(grad, index=i):
                    if not self.active:
                        return grad
                    value = grad.detach()
                    nonfinite = (~torch.isfinite(value).all()).to(torch.int32)
                    nonzero = (value != 0).any().to(torch.int32)
                    for flags in (self.flags, self.window_flags):
                        flags[index, 0].add_(1)
                        flags[index, 1].bitwise_or_(nonfinite)
                        flags[index, 2].bitwise_or_(nonzero)
                    return grad
                self.handles.append(parameter.register_hook(observe))

    def begin_microstep(self, examples, full_check=True):
        # Accelerate may call the actual DeepSpeed optimizer inside backward().
        # Clear the window only AFTER that optimizer was observed, not every
        # microstep and not according to a separate accumulation-step counter.
        if self._optimizer_step_seen:
            self.window_flags.zero_()
            self._optimizer_step_seen = False
        self.active = full_check
        self.flags.zero_()
        # Four future pixel frames share one latent. Fully padded video targets
        # legitimately produce zero gradients, while current action is always valid.
        self.flags[0, 3] = sum(all(ex["future_video_valid_mask"]) for ex in examples)
        self.flags[1, 3] = len(examples)
        self.window_flags[:, 3].add_(self.flags[:, 3])

    def before_optimizer_step(self, optimizer, _args, _kwargs):
        import torch
        # No collectives inside optimizer.step: some optimizer implementations
        # may traverse partitioned groups differently on different ranks.
        # Inspect accumulated master gradients as well as every observed raw
        # gradient. Combine GPU predicates before the single host synchronization.
        finite = [torch.isfinite(p.grad).all() for group in optimizer.param_groups
                  for p in group["params"] if p.grad is not None]
        master_finite = (torch.stack(finite).all() if finite else
                         torch.zeros((), device=self.flags.device, dtype=torch.bool))
        bad = ~master_finite
        if self.active:
            bad = bad | ((self.window_flags[:, 0] == 0) | (self.window_flags[:, 1] != 0)).any()
        if bad.item():
            raise FloatingPointError("Missing/nonfinite accumulated video/action or ZeRO master gradients")
        self._optimizer_step_seen = True

    def validate(self):
        import torch
        if not self.active:
            return None
        # All ranks call this once after each backward. The reduced valid-target
        # count controls only the video nonzero requirement, not presence/finite
        # checks; action gradients must remain nonzero even on padded video batches.
        flags = self.accelerator.reduce(torch.stack((self.flags, self.window_flags)), reduction="sum")
        require_nonzero = flags[:, :, 3] > 0
        require_nonzero[:, 1] = True
        bad = (flags[:, :, 0] == 0) | (flags[:, :, 1] != 0) | (require_nonzero & (flags[:, :, 2] == 0))
        if bad.any().item():
            raise FloatingPointError(f"Missing, nonfinite, or all-zero video/action gradients: {flags.tolist()}")
        return flags


def seed_worker(_worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)




def compare_normalization(actual, official):
    import numpy as np
    # Counts may differ; the transform/inference normalization values may not.
    for tag, expected in official.items():
        if tag == "metadata":
            continue
        if tag not in actual:
            raise ValueError(f"Missing official normalization tag {tag}")
        for modality in ("action", "state", "proprio"):
            if modality not in expected:
                continue
            candidate = actual[tag].get(modality, actual[tag].get("state" if modality == "proprio" else modality))
            if not isinstance(candidate, dict):
                raise ValueError(f"Missing official normalization modality {tag}/{modality}")
            for key, value in expected[modality].items():
                if key not in candidate:
                    raise ValueError(f"Missing official statistic {tag}/{modality}/{key}")
                x, y = np.asarray(candidate[key]), np.asarray(value)
                if x.shape != y.shape or not np.allclose(x, y, rtol=1e-6, atol=1e-7, equal_nan=False):
                    raise ValueError(f"Normalization changed at {tag}/{modality}/{key}; fix dataset adapter, do not rebase silently")


def warm_text_cache(raw, dataset, cfg):
    if not cfg.trainer.get("cache_frozen_text", False):
        return None
    extractor = raw.backbone_interface.extractor
    extractor.text_encoder.eval()
    prompts = sorted({str(text) for single in dataset.datasets for text in single.tasks["task"]})
    return extractor.prime_frozen_prompt_cache(prompts, offload_text=True)


def save_training_state(accelerator, output, context, step, epoch, batch_cursor, keep=2):
    import torch.distributed as dist
    root = Path(output) / "resume_states"
    temporary = [None]
    if accelerator.is_main_process:
        root.mkdir(parents=True, exist_ok=True)
        temporary[0] = tempfile.mkdtemp(prefix=f".step_{step:08d}-", dir=root)
    dist.broadcast_object_list(temporary, src=0)
    accelerator.save_state(temporary[0])
    accelerator.wait_for_everyone()
    destination = root / f"step_{step:08d}"
    if accelerator.is_main_process:
        metadata = {"schema_version": 1, "complete": True, "output_dir": str(Path(output).resolve()),
                    "context": context, "optimizer_steps": step, "epoch": epoch,
                    "batches_consumed_in_epoch": batch_cursor, "world_size": accelerator.num_processes}
        atomic_json(Path(temporary[0]) / "training_state.json", metadata)
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(temporary[0], destination)
        atomic_json(root / "latest.json", {"path": str(destination), "optimizer_steps": step})
        complete = sorted(root.glob("step_[0-9]*"))
        for obsolete in complete[:-keep]:
            marker = obsolete / "training_state.json"
            if marker.is_file() and json.loads(marker.read_text()).get("output_dir") == metadata["output_dir"]:
                shutil.rmtree(obsolete)
    accelerator.wait_for_everyone()
    return destination


def load_training_state(accelerator, path, output, context):
    metadata = json.loads((Path(path) / "training_state.json").read_text())
    if (not metadata.get("complete") or metadata["context"] != context
            or metadata["output_dir"] != str(Path(output).resolve()) or metadata["world_size"] != accelerator.num_processes):
        raise ValueError("Resume state belongs to a different source/config/run or is incomplete")
    accelerator.load_state(str(path), load_module_strict=True)
    return metadata


def validate_sample(sample):
    import numpy as np
    if np.asarray(sample["action"]).shape != (16, 8) or np.asarray(sample["state"]).shape != (1, 16):
        raise ValueError("Dataset must provide actions[16,8] and state[1,16]")
    mask = np.asarray(sample["action_mask"], dtype=bool)
    if mask.shape != (16, 8) or mask[:, 7].any() or not mask[8:, :7].any():
        raise ValueError("Invalid action dimension/episode validity mask")
    if len(sample["image"]) != 9 or len(sample["history_images"]) != 5 or len(sample["future_images"]) != 4:
        raise ValueError("Dataset must expose five history and four future frames")
    if not np.isfinite(sample["action"]).all() or not np.isfinite(sample["state"]).all():
        raise ValueError("Nonfinite normalized dataset sample")


def save_checkpoint(accelerator, model, destination, cfg, statistics, provenance, step):
    import torch
    from omegaconf import OmegaConf
    accelerator.wait_for_everyone()
    # Collective participation is required even when only rank zero writes files.
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint {destination}")
        temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + ".", dir=destination.parent))
        try:
            with (temporary / "pytorch_model.pt").open("wb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            atomic_bytes(temporary / "config.yaml", OmegaConf.to_yaml(cfg, resolve=True).encode())
            atomic_json(temporary / "config.json", OmegaConf.to_container(cfg, resolve=True))
            atomic_json(temporary / "dataset_statistics.json", statistics)
            atomic_json(temporary / "provenance.json", provenance)
            atomic_json(temporary / "training_metadata.json", {
                "optimizer_steps": step, "resume_supported": False,
                "format": "full_state_dict_inference_weights", "all_ranks_collected_state": True,
            })
            os.replace(temporary, destination)
            fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    del state
    accelerator.wait_for_everyone()


def train(args, cfg, context, manifest):
    import torch
    from accelerate import Accelerator, DataLoaderConfiguration, DeepSpeedPlugin
    from accelerate.utils import set_seed
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from DiT4DiT.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    from DiT4DiT.model.framework import build_framework

    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise RuntimeError("This recipe uses four ranks; launch with torchrun --nproc_per_node=4")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("Exactly four visible CUDA devices are required")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    zero = json.loads(args.zero_config.read_text())
    batch, accumulation = cfg.datasets.vla_data.per_device_batch_size, cfg.trainer.gradient_accumulation_steps
    zero.update(train_micro_batch_size_per_gpu=batch, gradient_accumulation_steps=accumulation,
                train_batch_size=batch * accumulation * 4)
    accelerator = Accelerator(
        mixed_precision="bf16", gradient_accumulation_steps=accumulation,
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=zero), log_with=None,
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False, split_batches=False, even_batches=True),
    )
    set_seed(cfg.seed)
    output = Path(cfg.output_dir)
    if accelerator.is_main_process:
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError("Output must be empty; this entrypoint does not resume or overwrite runs")
        output.mkdir(parents=True, exist_ok=True)
        lock_path = output / ".training.lock"
        if lock_path.exists():
            old_pid = int(lock_path.read_text())
            try:
                os.kill(old_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"Run lock PID {old_pid} is alive; never starting a duplicate")
        atomic_bytes(lock_path, str(os.getpid()).encode())
    accelerator.wait_for_everyone()
    checkpoint_hash = {"sha256": None, "status": "skipped_expensive_full_read"}
    if accelerator.is_main_process:
        if args.hash_checkpoint:
            checkpoint_hash = {"sha256": sha256_file(args.checkpoint), "status": "computed"}
        elif args.checkpoint_hash_cache:
            cache = json.loads(args.checkpoint_hash_cache.read_text())
            value = cache.get("sha256", "")
            if cache.get("identity") != context["checkpoint"] or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Checkpoint hash cache identity/digest is invalid")
            checkpoint_hash = {"sha256": value, "status": "cached_not_rehashed"}
        atomic_bytes(output / "config.yaml", OmegaConf.to_yaml(cfg, resolve=True).encode())
        atomic_json(output / "config.json", OmegaConf.to_container(cfg, resolve=True))
        atomic_json(output / "source_provenance.json", {"context": context, "files": manifest, "checkpoint_hash": checkpoint_hash})
        atomic_bytes(output / "original/dataset_statistics.json", args.checkpoint_stats.read_bytes())
        for parent in (args.checkpoint.parent, args.checkpoint.parent.parent):
            for filename in ("config.yaml", "config.json"):
                original = parent / filename
                if original.is_file() and not (output / "original" / filename).exists():
                    atomic_bytes(output / "original" / filename, original.read_bytes())
    accelerator.wait_for_everyone()
    provenance = json.loads((output / "source_provenance.json").read_text())
    # Do not call build_dataloader: it writes stats and assumes an initialized process group.
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, seed=cfg.seed)
    validate_sample(dataset[0])
    if accelerator.is_main_process:
        temporary_stats = output / ".dataset_statistics.json"
        dataset.save_dataset_statistics(temporary_stats)
        statistics = json.loads(temporary_stats.read_text())
        compare_normalization(statistics, json.loads(args.checkpoint_stats.read_text()))
        os.replace(temporary_stats, output / "dataset_statistics.json")
    accelerator.wait_for_everyone()
    statistics = json.loads((output / "dataset_statistics.json").read_text())
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=batch, collate_fn=collate_fn,
                        num_workers=cfg.trainer.num_workers, worker_init_fn=seed_worker,
                        generator=generator, drop_last=True, shuffle=False,
                        persistent_workers=False, pin_memory=True,
                        **({"prefetch_factor": cfg.trainer.get("prefetch_factor", 2)} if cfg.trainer.num_workers else {}))
    model = build_framework(cfg)
    load_summary = load_official_weights(model, args.checkpoint)
    groups = freeze_and_group(model, cfg)
    group_summary = {g["name"]: {"parameters": sum(p.numel() for p in g["params"]), "lr": g["lr"]} for g in groups}
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg.trainer.optimizer.betas),
                                 eps=cfg.trainer.optimizer.eps, weight_decay=cfg.trainer.optimizer.weight_decay)
    warmup = min(cfg.trainer.num_warmup_steps, max(cfg.trainer.max_train_steps - 1, 0))
    def schedule(base_lr):
        def factor(step):
            if warmup and step < warmup:
                return float(step + 1) / warmup
            progress = min(1.0, max(0.0, (step - warmup) / max(1, cfg.trainer.max_train_steps - warmup)))
            floor = cfg.trainer.min_lr / base_lr
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        return factor
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [schedule(g["lr"]) for g in groups])
    # Install hooks before ZeRO initialization; BF16 ZeRO may clear/replace .grad in backward().
    monitor = GradientMonitor(groups, accelerator)
    optimizer.register_step_pre_hook(monitor.before_optimizer_step)
    base_optimizer = optimizer
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    # Cosmos consumes CPU RGB frames and performs its own device transfer.
    # The default loader placement otherwise causes a GPU->CPU->GPU roundtrip.
    loader = accelerator.prepare_data_loader(loader, device_placement=False)
    if accelerator.state.deepspeed_plugin.zero_stage != 1:
        raise RuntimeError("Unexpected DeepSpeed stage after prepare")
    raw = accelerator.unwrap_model(model)
    for name, parameter in raw.named_parameters():
        if parameter.requires_grad and parameter.dtype != torch.bfloat16:
            raise RuntimeError(f"Unexpected trainable dtype after ZeRO initialization: {name}")
    masters = [p for g in base_optimizer.param_groups for p in g["params"]]
    if not masters or not all(p.dtype == torch.float32 for p in masters):
        raise RuntimeError("ZeRO-1 must provide FP32 optimizer master parameters for BF16 model weights")
    if source_manifest() != manifest or identity(args.checkpoint) != context["checkpoint"]:
        raise RuntimeError("Source/checkpoint changed during initialization; rerun validation")
    if accelerator.is_main_process:
        atomic_json(output / "initialization.json", {"strict_load": load_summary, "groups": group_summary,
                    "trainable_dtype": "bfloat16", "optimizer_master_dtype": "float32",
                    "effective_batch_size": batch * accumulation * accelerator.num_processes,
                    "gradient_checkpointing": cfg.trainer.enable_gradient_checkpointing,
                    "frozen_modules": ["vae", "text_encoder"]})
    if len(loader) == 0:
        raise RuntimeError("Prepared dataloader is empty")
    model.train()
    extractor = raw.backbone_interface.extractor
    extractor.vae.eval()
    extractor.text_encoder.eval()
    cache_info = warm_text_cache(raw, dataset, cfg)
    if accelerator.is_main_process and cache_info:
        atomic_json(output / "frozen_text_cache.json", cache_info)
    for module in (extractor.vae, extractor.text_encoder):
        if any(p.requires_grad for p in module.parameters()):
            raise RuntimeError("VAE/text unexpectedly unfrozen")
    set_seed(cfg.seed + accelerator.process_index)
    step, epoch, microsteps = 0, 0, 0
    batch_cursor = 0
    if args.resume:
        resumed = load_training_state(accelerator, args.resume, output, context)
        step, epoch, batch_cursor = (resumed["optimizer_steps"], resumed["epoch"], resumed["batches_consumed_in_epoch"])
    metrics_sum = torch.zeros(3, device=accelerator.device)
    started = time.monotonic()
    session_start_step = step
    timing_sum = torch.zeros(2, device=accelerator.device)
    while step < args.stop_after:
        dataset.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        active_loader = accelerator.skip_first_batches(loader, batch_cursor) if batch_cursor else loader
        previous_end = time.monotonic()
        for batch_index, examples in enumerate(active_loader, start=batch_cursor):
            data_wait = time.monotonic() - previous_end
            compute_start = time.monotonic()
            monitor.begin_microstep(examples, full_check=(step < 3 or step % cfg.trainer.get("gradient_validation_interval", 20) == 0))
            with accelerator.accumulate(model):
                with torch.enable_grad(), accelerator.autocast():
                    losses = model(examples=examples)
                    parts = [losses.get("action_loss"), losses.get("future_video_loss")]
                    valid = all(torch.is_tensor(x) and x.ndim == 0 and x.requires_grad and x.grad_fn is not None for x in parts)
                    flag = torch.tensor(int(valid), device=accelerator.device)
                    if accelerator.reduce(flag, reduction="sum").item() != accelerator.num_processes:
                        raise RuntimeError("Both scalar action/video losses must retain their autograd graph")
                    total = parts[0] + cfg.trainer.loss_scale.future_video * parts[1]
                    values = torch.stack([parts[0].detach().float(), parts[1].detach().float(), total.detach().float()])
                    finite = torch.isfinite(values).all().to(torch.int32)
                    if accelerator.reduce(finite, reduction="sum").item() != accelerator.num_processes:
                        raise FloatingPointError("Nonfinite action, video, or total loss")
                    accelerator.backward(total)
                monitor.validate()
                # In DeepSpeed these wrappers are no-ops: engine.backward/step own
                # accumulation, clipping, the optimizer, zero_grad and scheduler.
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                metrics_sum += values
                timing_sum += timing_sum.new_tensor([data_wait, time.monotonic() - compute_start])
                microsteps += 1
            batch_cursor = batch_index + 1
            previous_end = time.monotonic()
            if not accelerator.sync_gradients:
                continue
            step += 1
            if step == 1 or step % cfg.trainer.logging_frequency == 0:
                means = accelerator.reduce(metrics_sum / microsteps, reduction="mean").tolist()
                timing = accelerator.reduce(timing_sum / microsteps, reduction="mean").tolist()
                elapsed = time.monotonic() - started
                record = {"step": step, "epoch": epoch, "action_loss": means[0], "future_video_loss": means[1],
                          "loss": means[2], "seconds": elapsed,
                          "mean_data_wait_seconds": timing[0], "mean_compute_seconds": timing[1],
                          "optimizer_steps_per_second": (step - session_start_step) / elapsed,
                          "examples_per_second": (step - session_start_step) * batch * accumulation * accelerator.num_processes / elapsed,
                          "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                          "lr": [g["lr"] for g in optimizer.param_groups]}
                if accelerator.is_main_process:
                    with (output / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(record, allow_nan=False) + "\n")
                    print(json.dumps(record), flush=True)
                metrics_sum.zero_()
                timing_sum.zero_()
                microsteps = 0
            if step % cfg.trainer.save_interval == 0:
                save_checkpoint(accelerator, model, output / f"step_{step:08d}", cfg, statistics, provenance, step)
                if accelerator.is_main_process:
                    snapshots = sorted(output.glob("step_[0-9]*"))
                    for old in snapshots[:-cfg.trainer.get("keep_last_weights", 3)]:
                        old_step = int(old.name.split("_")[-1])
                        if old_step not in cfg.trainer.get("landmark_steps", [5000, 10000, 20000]):
                            shutil.rmtree(old)
                accelerator.wait_for_everyone()
                previous_end = time.monotonic()
            if step == cfg.trainer.get("first_state_save_step", 50) or step % cfg.trainer.get("state_save_interval", 1000) == 0:
                save_training_state(accelerator, output, context, step, epoch, batch_cursor,
                                    cfg.trainer.get("keep_last_training_states", 2))
                previous_end = time.monotonic()
            if step >= args.stop_after:
                break
        epoch += 1
        batch_cursor = 0
    save_checkpoint(accelerator, model, output / "final_model", cfg, statistics, provenance, step)
    if accelerator.is_main_process:
        atomic_json(output / "completion.json", {"optimizer_steps": step, "completed": True})
        (output / ".training.lock").unlink()
    accelerator.end_training()


def resolve_joint_config(args):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    for key, value in {
        "output_dir": str(args.output.resolve()),
        "trainer.pretrained_checkpoint": str(args.checkpoint.resolve()),
    }.items():
        OmegaConf.update(cfg, key, value, merge=False)
    expected = {"max_steps": 20000, "batch_size": 16, "grad_accum": 1,
                "num_workers": 4, "seed": 42}
    for name, value in expected.items():
        if getattr(args, name) not in (None, value):
            raise ValueError(f"Budget override rejected: {name}")
    if args.lr_preset != "official":
        raise ValueError("The continued Joint recipe uses the official learning-rate preset")
    OmegaConf.update(cfg, "framework.cosmos25.base_model", str(args.cosmos_path.resolve()))
    OmegaConf.update(cfg, "datasets.vla_data.data_root_dir", str(args.data_root.resolve()))
    if args.frame_cache:
        OmegaConf.update(cfg, "datasets.vla_data.streaming_frame_cache_root", str(args.frame_cache.resolve()))
    d, a, t, c = cfg.datasets.vla_data, cfg.framework.action_model, cfg.trainer, cfg.framework.cosmos25
    assert d.streaming is False and a.streaming is False
    assert (a.past_action_window_size, a.future_action_window_size, a.action_horizon) == (0, 7, 8)
    assert list(d.video_delta_indices) == list(range(9)) and d.action_video_freq_ratio == 2
    assert d.native_v3_aligned_io is True and d.lerobot_version == "v3.0"
    assert (d.max_state_dim, d.max_action_dim, a.state_dim, a.action_dim) == (16, 8, 16, 8)
    assert c.training == "joint" and c.future_loss_type == "flow_matching" and c.trainable is True
    assert c.flow_matching.time_distribution == "uniform"
    assert c.flow_matching.high_sigma_ratio is None and c.flow_matching.high_sigma_min is None
    assert t.max_train_steps == 20000 and t.num_warmup_steps == 1000 and cfg.seed == 42
    assert d.per_device_batch_size == 16 and t.gradient_accumulation_steps == 1 and t.num_workers == 4
    assert t.learning_rate.backbone_interface == 1e-5 and t.learning_rate.action_model == 1e-4
    assert t.min_lr == 5e-7 and t.optimizer.weight_decay == 1e-8
    assert tuple(t.optimizer.betas) == (.9, .95) and t.optimizer.eps == 1e-8
    assert t.gradient_clipping == 1 and t.mixed_precision == "bf16" and t.enable_gradient_checkpointing
    assert list(t.landmark_steps) == [5000, 10000, 18000, 20000]
    zero = json.loads(args.zero_config.read_text())
    assert zero["zero_optimization"]["stage"] == 1 and zero["bf16"]["enabled"] is True
    assert zero["gradient_clipping"] == t.gradient_clipping
    return cfg


def validate_joint_sample(sample):
    import numpy as np
    assert np.asarray(sample["action"]).shape == (8, 8)
    assert np.asarray(sample["state"]).shape == (1, 16)
    mask = np.asarray(sample["action_mask"], dtype=bool)
    assert mask.shape == (8, 8) and mask[:, :7].all() and not mask[:, 7].any()
    assert len(sample["image"]) == 5
    assert not any(k in sample for k in ("history_images", "future_images", "future_video_valid_mask"))
    assert all(tuple(image.shape) == (3, 224, 448) for image in sample["image"])
    assert np.isfinite(sample["action"]).all() and np.isfinite(sample["state"]).all()


class JointGradientMonitor(GradientMonitor):
    def begin_microstep(self, examples, full_check=True):
        # Native official training supervises even boundary-clamped video targets.
        monitored = [dict(ex, future_video_valid_mask=[True] * 4) for ex in examples]
        return super().begin_microstep(monitored, full_check)


def resolve_config(args):
    if args.config is None:
        args.config = ROOT / "configs" / (args.method + ".json")
    cfg = resolve_joint_config(args) if args.method == "joint" else resolve_streaming_config(args)
    if not 0 < args.stop_after <= cfg.trainer.max_train_steps:
        raise ValueError("stop-after must be positive and cannot exceed the schedule length")
    if cfg.datasets.vla_data.per_device_batch_size * cfg.trainer.gradient_accumulation_steps * 4 != 64:
        raise ValueError("The released comparison uses global batch size 64")
    return cfg


def main():
    global GradientMonitor, validate_sample
    args = parser().parse_args()
    cfg = resolve_config(args)
    context, manifest = validation_context(args, cfg)
    if not args.cosmos_path.is_dir() or not args.data_root.is_dir():
        raise FileNotFoundError("Cosmos and prepared training data directories must exist")
    if args.preflight_only:
        print(json.dumps({"context": context, "schedule_steps": cfg.trainer.max_train_steps,
                          "stop_after": args.stop_after, "training_started": False}, indent=2))
        return
    if args.method == "joint":
        GradientMonitor, validate_sample = JointGradientMonitor, validate_joint_sample
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    train(args, cfg, context, manifest)


if __name__ == "__main__":
    main()
