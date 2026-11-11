# DiT4DiT release protocol

The comparison transfers StreamingWAM to a Cosmos-Predict2.5-2B-based model. Both methods start from the same upstream DiT4DiT-LIBERO checkpoint and receive 10,000 additional optimizer updates with global batch size 64. These are the 10K checkpoints of 20K cosine schedules. Matching updates and batch size does not imply equal training FLOPs.

| Setting | Joint continued | Streaming adaptation |
| --- | --- | --- |
| Video conditioning at inference | Current paired-camera image | Five real paired-camera images, offsets −4…0 |
| Predicted future video frames | Offsets 2, 4, 6, 8 | Offsets 2, 4, 6, 8 |
| Clean executed-action prefix | None | 8 actions |
| Future action horizon H | 8 | 8 |
| Executed actions per replan K | 4 | 4 |
| Action updates per steady replan | 4 | 1 |
| Episode initialization | Ordinary sampling | 4-step warm start, then the first staircase update |
| Video flow-time distribution in training | Uniform | Logit-normal with 5% high-noise mixture (≥0.98) |
| Action training | Original joint objective | 80% staircase / 20% uniform-time objective |
| Training K | Not a streaming schedule | Fixed K=4 |

Both branches train the Cosmos transformer and ActionDiT with learning rates 1e−5 and 1e−4, respectively. They use AdamW β=(0.9, 0.95), ε=1e−8, weight decay 1e−8, gradient clipping 1, 1,000 warm-up updates, a 20K cosine schedule with minimum learning rate 5e−7, BF16 parameters and ZeRO-1 FP32 optimizer states. VAE and text encoder remain frozen; action features are detached from the video branch.

Images are paired as primary view on the left and wrist view on the right. The ordinary policy retains upstream OpenCV preprocessing; the streaming policy uses its shared training/inference tensor preprocessing. The streaming policy records observations at every control step, carries the full post-update future buffer, and feeds back the clipped/thresholded actions actually executed. It resets image history, prefix and buffer at every episode boundary.

## Evaluation and interpretation

The release uses all 10,030 LIBERO-Plus entries, raw `task.language` including benchmark suffixes, native initial-state index 0, seed 7, control rate 20 Hz, five settling steps and a 700-step limit. Virtual BDDL names are preserved because they activate perturbations. This protocol does not apply the FastWAM prompt sanitizer.

The streaming 10K/K4 point was selected using a fixed 200-entry screen. The full manifest includes those entries, so it is not a completely held-out evaluation. Joint continuation was then evaluated at the matched 10K/K4 point. The release demonstrates the combined adaptation; it is not an isolated history-only or staircase-only ablation.

A separate full-call timing measurement gave 135.526 ms for ordinary DiT4DiT and 153.685 ms for the streaming adaptation. Fewer action denoising updates do not guarantee lower end-to-end latency here because visual-history processing also costs time. This release makes no end-to-end speedup claim for DiT4DiT.

## Checkpoint and source identity

`release.json` records the two full SHA256 digests, upstream initialization digest, source revision and inference settings. The weights contain the full model state dict (1,739 tensors), without optimizer states or training logs. Configs and dataset statistics are stored beside each weight on Hugging Face. `vendor/SOURCE_SHA256.json` records the bundled runtime files.

The frozen native evaluation runtime and the training/streaming runtime derive from upstream revision `66a6f3a12e2c8740157c1e478795952c040d31dd`. Public launchers replace host-specific paths, device restrictions and internal orchestration; they preserve model computations and preprocessing. Use the bundled entrypoints rather than installing a newer DiT4DiT release over them.
