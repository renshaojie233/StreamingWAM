"""Download the two continued-training checkpoints and verify their identities."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import ROOT, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("joint", "streaming", "both"), default="both")
    p.add_argument("--output", type=Path, default=Path("checkpoints/dit4dit"))
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    manifest = json.loads((ROOT / "release.json").read_text())
    methods = ("joint", "streaming") if args.method == "both" else (args.method,)
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(manifest["model_repo"], revision=manifest["model_revision"],
                          local_dir=args.output,
                          allow_patterns=[f"{m}/*" for m in methods] + ["README.md", "LICENSE*", "NOTICE*", "release.json"])
    for method in methods:
        checkpoint = args.output / method / "pytorch_model.pt"
        info = manifest["checkpoints"][method]
        if checkpoint.stat().st_size != info["bytes"] or sha256_file(checkpoint) != info["sha256"]:
            raise ValueError(f"Incomplete or incorrect {method} checkpoint")
        for name, digest in info["sidecars"].items():
            if sha256_file(args.output / method / name) != digest:
                raise ValueError(f"Incorrect {method}/{name}")
        print(f"Verified {method}: {checkpoint.resolve()}")


if __name__ == "__main__":
    main()
