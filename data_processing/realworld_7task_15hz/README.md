# StreamingWAM real-world 7-task 15 Hz conversion

This directory contains the exact conversion and validation scripts used for
the real-world StreamingWAM/FastWAM experiment. The scripts are copied from
the formal H20 run and are intentionally kept separate from deployment code.

## Provenance

- Formal source project: `data_realworld_sdp_15hz_v1`
- Formal read-only source on H20: `/mnt/workspace/rsj/data_realworld_sdp_15hz_v1/raw/data_15hz`
- Formal LeRobot v2.1 output on H20: `/mnt/workspace/rsj/data_realworld_sdp_15hz_v1/lerobot/realworld_7task_15hz_v21`
- Formal conversion version: `fastwam_realworld_15hz_multitask_lerobot_v21_v1.0`
- Conversion script SHA-256: `067d7040e7826a84736f9ec0f351244d5361ae1ba31ec74bcddad1fc31826928`
- Validation script SHA-256: `6a1c8d236129f3a50e3a938f0810b69990ca6e2324f16aaa16b33222721afa0`

The formal training source contained 1,083 episodes, 297,008 frames, 3,249
videos, seven tasks, and no validation split. The source files are never
modified by either script.

## Public data versus formal training data

The public raw-data repository is
[rsj2003/StreamingWAM-RealWorld](https://huggingface.co/datasets/rsj2003/StreamingWAM-RealWorld).
It is the publication subset previously selected from the raw data: 590
trajectories (at most 100 per task). It is not the complete 1,083-episode
formal training source, so the fixed-count full validator below is expected to
reject that public subset. The scripts in this directory are the reproducible
code for the complete formal source; the public subset remains byte-preserved
and is not overwritten.

## Conversion contract

- Input is already synchronized 15 Hz data; no interpolation, frame insertion,
  downsampling, or re-encoding is performed.
- Camera order is `camera_1`, `camera_3`, `camera_2` (external view 1,
  external view 2, wrist view). Each camera remains a separate video.
- State is seven measured Franka joint positions in radians plus the continuous
  GELLO gripper command (`0` closed, `1` open).
- Action is `(q[t+1] - q[t]) / 0.2` for the seven joints plus the absolute
  GELLO gripper command; the final joint increment is zero.
- FastWAM resizes each camera to `224x128` and vertically concatenates the
  three views to `384x224` with `concat_multi_camera: vertical`.
- All episodes stay in the training split. The converter preserves source
  absolute timestamps and writes constant-frame-rate timestamps as `index/15`.

## Reproduce the formal conversion

Use fresh output and audit directories. Existing output/audit paths are
rejected rather than overwritten:

```bash
python -u data_processing/realworld_7task_15hz/convert_realworld_15hz_multitask_to_lerobot_v21.py \
  --source-root /path/to/data_15hz \
  --output-root /path/to/new/realworld_7task_15hz_v21 \
  --audit-root /path/to/new/realworld_7task_15hz_v21_conversion_audit
```

Then run the full video, numeric, visualization, and FastWAM dataloader
validation:

```bash
python -u data_processing/realworld_7task_15hz/validate_realworld_15hz_multitask_v21.py \
  --source-root /path/to/data_15hz \
  --dataset-root /path/to/new/realworld_7task_15hz_v21 \
  --audit-root /path/to/new/realworld_7task_15hz_v21_validation_audit \
  --fastwam-root /path/to/FastWAM \
  --workers 8
```

The formal validation passed with 3,249/3,249 videos decoded, zero missing or
bad frames, zero NaN/Inf values, exact source state/action reconstruction, and
FastWAM smoke shapes `[3, 9, 384, 224]` per sample and `[2, 3, 9, 384, 224]`
per batch. The Wan2.2 VAE/DiT spatial grid is `12x7` (84 tokens per frame).

## Files

- `convert_realworld_15hz_multitask_to_lerobot_v21.py`: formal converter.
- `validate_realworld_15hz_multitask_v21.py`: full validator and FastWAM loader
  smoke test.
