"""Expose IPEC's LIBERO v2.1 episodes through a lossless v3-compatible file view.

Only metadata and symlinks are written. Source parquet/video bytes, timestamps,
state/action values and gripper convention are preserved. Keep the source folder.
"""
import argparse
import copy
import json
from pathlib import Path
import pandas as pd


def convert(source, target):
    source = source.resolve(strict=True)
    target = target.resolve()
    if target.exists() or source == target or source in target.parents or target in source.parents:
        raise ValueError('Choose a new, independent output directory')
    info = json.loads((source/'meta/info.json').read_text())
    if info['codebase_version'] != 'v2.1' or info['fps'] != 20:
        raise ValueError('This converter expects IPEC LIBERO v2.1 at 20 FPS')
    episodes = [json.loads(x) for x in (source/'meta/episodes.jsonl').read_text().splitlines()]
    tasks = [json.loads(x) for x in (source/'meta/tasks.jsonl').read_text().splitlines()]
    if len(episodes) != info['total_episodes'] or sum(x['length'] for x in episodes) != info['total_frames']:
        raise ValueError('Episode metadata counts disagree')
    keys = ('observation.images.image','observation.images.wrist_image')
    links=[];rows=[];offset=0
    for ep in sorted(episodes,key=lambda x:x['episode_index']):
        i=int(ep['episode_index']);chunk=i//info['chunks_size']
        row=dict(ep, **{'data/chunk_index':chunk,'data/file_index':i,'data/file_from_index':0,
                       'dataset_from_index':offset,'dataset_to_index':offset+ep['length']})
        links.append((source/info['data_path'].format(episode_chunk=chunk,episode_index=i),
                      target/f'data/chunk-{chunk:03d}/file-{i:06d}.parquet'))
        for key in keys:
            links.append((source/info['video_path'].format(episode_chunk=chunk,episode_index=i,video_key=key),
                          target/f'videos/{key}/chunk-{chunk:03d}/file-{i:06d}.mp4'))
            row.update({f'videos/{key}/chunk_index':chunk,f'videos/{key}/file_index':i,
                        f'videos/{key}/from_timestamp':0.,f'videos/{key}/to_timestamp':ep['length']/20})
        rows.append(row);offset+=ep['length']
    for src,_ in links:
        if not src.is_file():raise FileNotFoundError(src)
    meta=target/'meta';(meta/'episodes/chunk-000').mkdir(parents=True)
    for src,dst in links:
        dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src)
    out=copy.deepcopy(info)
    out.update(codebase_version='v3.0', data_path='data/chunk-{chunk_index:03d}/file-{file_index:06d}.parquet',
               video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:06d}.mp4')
    (meta/'info.json').write_text(json.dumps(out,indent=2)+'\n')
    pd.DataFrame(rows).to_parquet(meta/'episodes/chunk-000/file-000.parquet',index=False)
    pd.DataFrame(tasks).set_index('task').to_parquet(meta/'tasks.parquet')
    print(json.dumps({'output':str(target),'episodes':len(rows),'frames':offset,'videos_reencoded':False}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args()
    for suite in ('spatial','object','goal','10'):
        convert(a.source_root/f'libero_{suite}_no_noops_1.0.0_lerobot',
                a.output_root/f'libero_{suite}_no_noops_lerobot')

if __name__=='__main__':main()
