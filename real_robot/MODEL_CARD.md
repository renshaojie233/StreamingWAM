---
license: mit
frameworks:
  - pytorch
tasks:
  - robotics
---

# StreamingWAM real-world checkpoint

The `realworld-v1` GitHub release contains the final 30k StreamingWAM
checkpoint used for the seven-task Franka real-robot evaluation.

## Files

- `step_030000.pt`: final model weights;
- `train_split_zscore_stats.json`: state/action z-score statistics;
- `text_embeds_cache/`: cached embeddings for the seven task prompts;
- `SHA256SUMS`: file integrity manifest.

The matching implementation and model-only deployment configuration are in
the public [StreamingWAM repository](https://github.com/renshaojie233/StreamingWAM/tree/main/real_robot).

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

`step_030000.pt` is 12,041,739,641 bytes and has SHA-256:

```text
ed68b39a9d7025188e6216a9336a3e265893d360b0c02b91f31629c317d1f66a
```

The Wan 2.2 TI2V 5B base model is not redistributed and must be obtained under
its original license.
