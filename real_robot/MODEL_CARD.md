---
license: mit
library_name: pytorch
tags:
  - robotics
  - streamingwam
  - real-world
tasks:
  - robotics
---

# StreamingWAM real-world checkpoint

This repository contains the StreamingWAM checkpoint used for the seven-task
Franka real-robot evaluation. It is separate from the
[LIBERO simulation checkpoint](https://huggingface.co/rsj2003/StreamingWAM-LIBERO)
and the [real-world demonstration dataset](https://huggingface.co/datasets/rsj2003/StreamingWAM-RealWorld).

## Files

- `streamingwam_realworld.pt`: complete model weights (one file);
- `streamingwam-realworld-runtime-assets.tar.gz`: normalization statistics and cached embeddings for the seven task prompts;
- `streamingwam_realworld_model.yaml`: model-only deployment configuration;
- `checkpoint_info.json` and `SHA256SUMS`: artifact metadata and integrity checks.

The runtime archive contains `train_split_zscore_stats.json` and
`text_embeds_cache/`. No training logs or optimizer artifacts are added.

The matching implementation and model-only deployment configuration are in
the public [StreamingWAM repository](https://github.com/renshaojie233/StreamingWAM/tree/main/real_robot).

## Download and use

From the [StreamingWAM code repository](https://github.com/renshaojie233/StreamingWAM):

```bash
python real_robot/download_assets.py --output checkpoints/streamingwam-realworld
```

The downloader verifies SHA-256 checksums and installs the checkpoint locally
as `step_030000.pt` for compatibility with the existing runtime. See the
[real-robot setup guide](https://github.com/renshaojie233/StreamingWAM/tree/main/real_robot)
for observation preprocessing, shadow-mode inference, and hardware integration.

## Contract

- control frequency: 15 Hz;
- cameras: `camera_1`, `camera_3`, `camera_2`;
- each camera: RGB `224 x 128`, vertically concatenated to `3 x 384 x 224`;
- image history: five consecutive frames;
- action prefix: sixteen actually executed native actions;
- output: sixteen future actions of dimension eight;
- arm action: `(q[t+1] - q[t]) / 0.2`;
- gripper action/state: continuous absolute command, `0=closed`, `1=open`;
- initial warm start: ten denoising steps;
- rolling update: one denoising substep;
- supported trained replanning horizons: `K={1,2,4,8,10}`.

The released checkpoint is the standard model, not a latency-shift fine-tune.
No transition blending or trajectory smoothing is part of the model artifact.

## Integrity

`streamingwam_realworld.pt` is 12,041,739,641 bytes and has SHA-256:

```text
ed68b39a9d7025188e6216a9336a3e265893d360b0c02b91f31629c317d1f66a
```

The Wan 2.2 TI2V 5B base model is not redistributed and must be obtained under
its original license.
