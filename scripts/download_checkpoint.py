#!/usr/bin/env python3
"""Download and verify the public StreamingWAM LIBERO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from check_checkpoint import check_file


REPO_ID = "rsj2003/StreamingWAM-LIBERO"
FILENAME = "streamingwam_libero.pt"
REVISION = "0b5a17cc2b81bd91ed15b3fe3e3be7a9ad13ba48"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/streamingwam-libero"),
        help="Directory that receives the checkpoint.",
    )
    parser.add_argument(
        "--revision",
        default=REVISION,
        help="Hugging Face revision to download (defaults to the verified release).",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip the full-file SHA-256 verification after download.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            revision=args.revision,
            local_dir=str(output_dir),
        )
    ).resolve()
    check_file(path, check_hash=not args.skip_sha256)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
