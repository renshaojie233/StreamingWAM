"""Fetch pinned IPEC LIBERO source snapshots for the data adapter."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path('data/ipec'))
a=p.parse_args()
for suite,source in json.loads((ROOT/'configs/data_sources.json').read_text()).items():
    folder=a.output/source['repo'].split('/')[-1]
    snapshot_download(source['repo'],repo_type='dataset',revision=source['revision'],local_dir=folder)
    info=json.loads((folder/'meta/info.json').read_text())
    if (info['codebase_version'],info['total_episodes'],info['total_frames'])!=(source['layout'],source['episodes'],source['frames']):
        raise ValueError(f'Incorrect data metadata: {suite}')
    print(f'Verified data metadata: {folder.resolve()}')
