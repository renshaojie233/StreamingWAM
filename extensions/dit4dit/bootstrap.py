"""Select the frozen DiT4DiT runtime before any DiT4DiT imports."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
_ACTIVE = None


def activate(method):
    global _ACTIVE
    if method not in ("joint", "streaming"):
        raise ValueError(method)
    if _ACTIVE is not None and _ACTIVE != method:
        raise RuntimeError("Run each method in a separate Python process")
    if _ACTIVE is None and any(k == "DiT4DiT" or k.startswith("DiT4DiT.") for k in sys.modules):
        raise RuntimeError("Activate the release runtime before importing DiT4DiT")
    for path in (ROOT / "vendor/common", ROOT / "vendor" / method, ROOT / "evaluation"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    _ACTIVE = method


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def load_model(method, checkpoint_dir, cosmos_path, device="cuda", verify=True):
    """Strictly load one release checkpoint, using its frozen inference config."""
    import json
    activate(method)
    import torch
    from omegaconf import OmegaConf
    from DiT4DiT.model.framework import build_framework
    checkpoint_dir = Path(checkpoint_dir)
    ck = checkpoint_dir / "pytorch_model.pt"
    manifest = json.loads((ROOT / "release.json").read_text())
    info = manifest["checkpoints"][method]
    if ck.stat().st_size != info["bytes"]:
        raise ValueError("Checkpoint size mismatch: use the download script")
    if verify and sha256_file(ck) != info["sha256"]:
        raise ValueError("Checkpoint SHA256 mismatch")
    cfg = OmegaConf.load(checkpoint_dir / "inference_config.json")
    expected = OmegaConf.load(ROOT / "configs" / (method + "_inference.json"))
    if OmegaConf.to_container(cfg, resolve=False) != OmegaConf.to_container(expected, resolve=False):
        raise ValueError("Inference configuration differs from the released protocol")
    cfg.framework.cosmos25.base_model = str(Path(cosmos_path).resolve(strict=True))
    cfg.trainer.pretrained_checkpoint = None
    model = build_framework(cfg=cfg)
    payload = torch.load(ck, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(payload, strict=True)
    del payload
    statistics = json.loads((checkpoint_dir / "dataset_statistics.json").read_text())
    if statistics != json.loads((ROOT / "configs/dataset_statistics.json").read_text()):
        raise ValueError("Normalization statistics differ from the release")
    model.norm_stats = statistics
    return model.to(device, dtype=torch.bfloat16).eval()
