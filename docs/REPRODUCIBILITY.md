# LIBERO-Plus reproducibility record

This file records the protocol behind the reported StreamingWAM LIBERO-Plus
result. It is kept separate from the generic defaults so that future code
changes do not silently alter the experimental claim.

## Training

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
| Training K | `{1,2,4,8,10}` |
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
| Successes | 7,960 / 10,030 |
| Success rate | 79.3619142572% |

The evaluator writes one result JSON per task and a worker summary. The task
manifest and normalization statistics are committed to make their identity
reviewable.

## Artifact hashes

Run `python scripts/check_release.py` to verify the committed assets. The
reported checkpoint is intentionally absent from this source-only release; its
recorded SHA-256 is:

```text
7b321921e18ec4f417cebc1bfdd7aa5e9db8d81d4e0c4711aefb81d7c385aa45
```

