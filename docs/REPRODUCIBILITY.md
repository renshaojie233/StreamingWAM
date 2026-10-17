# LIBERO-Plus reproducibility record

This file records the protocol released for StreamingWAM LIBERO-Plus. It is
kept separate from the generic defaults so that future code changes do not
silently alter the public training and evaluation recipe.

This is a four-width public recipe. The reference metrics below identify the
released paper checkpoint for provenance.

## Training

The one-command launcher first trains FastWAM-Joint for 34,720 optimization
steps with global batch 80. Both reported stage-two branches start from that
same weights-only checkpoint and receive another 40,000 optimization steps
with global batch 64. Run `python scripts/train_libero_plus_two_stage.py
--nproc-per-node 4 --second-stage both` to generate the StreamingWAM branch and
the equal-budget FastWAM-Joint continuation.

| Item | Value |
| --- | --- |
| Initialization | FastWAM-Joint `step_034720.pt`, weights only |
| GPUs | 4 |
| Precision | bf16 |
| Seed | 42 |
| Batch | 2 per rank × 8 gradient accumulation × 4 ranks = 64 global |
| Optimizer LR | `1e-4` |
| Scheduler | cosine, no linear warm-up, horizon 60,000 steps |
| Weight decay | `1e-2` |
| Reported checkpoint | stage-two step 40,000 |
| Video history | 5 dense frames at offsets `[-4,-3,-2,-1,0]` |
| Future video offsets | `[4,8,12,16]` |
| Action horizon | 32: 16 clean-prefix + 16 future steps |
| Staircase probability | 0.8 |
| Training K | `{1,2,4,8}` |
| Noise levels | 32, with random offset |

There is no 5% optimization warm-up in this stage. “Warm start” in the paper
refers to the one-time, episode-initial 10-step diffusion sampler used at
evaluation, not a learning-rate schedule.

## Main evaluation

| Item | Value |
| --- | --- |
| Manifest | `assets/libero_plus_full_10030.txt` |
| Episodes | 10,030 |
| Replanning / execution horizon | `K=4` |
| Steady-state denoising | one Euler update |
| Episode-initial warm start | 10 denoising steps |
| Legacy raw instructions | 7,960 / 10,030 (79.3619142572%) |
| Corrected clean instructions | 8,237 / 10,030 (82.1236291127%) |

These two results refer to the reference checkpoint with SHA-256 `7b321c0c…`,
now available from the public download listed below. See the
[prompt correction record](evaluation/README.md) for full checkpoint identities,
per-task results, and commands for both protocols.

The evaluator writes one result JSON per task and a worker summary. The task
manifest and normalization statistics are committed to make their identity
reviewable.

## Artifact hashes

Run `python scripts/check_release.py` to verify the committed assets. The
LIBERO-trained checkpoint is hosted in the separate
[StreamingWAM-LIBERO model repository](https://huggingface.co/rsj2003/StreamingWAM-LIBERO).
The downloader pins revision `01bd9366af1bac17c85d6f7b9af78a036372488c`,
whose checkpoint matches the reference used for both scores above:

```text
7b321c0c85c6167fda5eb21f7e1fe6abe745f9b77be20eef0144b97ae2d6aa45
```
