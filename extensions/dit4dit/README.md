# StreamingWAM on DiT4DiT

This extension adapts StreamingWAM to [DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT), whose video backbone is **Cosmos-Predict2.5-2B**. It includes the ordinary Joint continued-training baseline and our streaming adaptation, both initialized from the same official DiT4DiT-LIBERO checkpoint.

[🤗 Two checkpoints](https://huggingface.co/rsj2003/StreamingWAM-DiT4DiT-LIBERO) · [📋 Protocol](PROTOCOL.md) · [📊 Results](results/summary.json)

| Released model | Additional optimizer updates | Global batch | Future H / executed K | LIBERO-Plus |
| --- | ---: | ---: | --- | ---: |
| DiT4DiT Joint, continued | 10,000 | 64 | 8 / 4 | 7,109 / 10,030 (**70.88%**) |
| StreamingWAM on DiT4DiT | 10,000 | 64 | 8 / 4 | 7,660 / 10,030 (**76.37%**) |

These results use the original **raw task instructions**, including benchmark suffixes. They are separate from the FastWAM clean-prompt results in the main README. The original unmodified DiT4DiT weights remain available from the upstream authors; this model repository contains only the two continued-training checkpoints.

## 1. Environment

Use a separate environment from the FastWAM release. Commands below run from `extensions/dit4dit`.

```bash
conda create -n streamingwam-dit4dit python=3.10 -y
conda activate streamingwam-dit4dit
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The Diffusers revision is pinned in `requirements.txt`. No installation of the upstream DiT4DiT package is required: this directory selects its bundled runtime explicitly. Inference uses roughly 22 GiB of GPU memory in the reference environment. Training used four 80 GB GPUs, BF16 model parameters, and ZeRO-1 FP32 optimizer states.

## 2. Download and load the released models

Download either model, or both (about 21.3 GB each). The script checks the full checkpoint SHA256 and the accompanying configs/statistics.

```bash
python scripts/download.py --method both --output checkpoints/dit4dit
```

Download the Cosmos assets using your Hugging Face account with access to the base model:

```bash
hf download nvidia/Cosmos-Predict2.5-2B \
  --revision diffusers/base/post-trained --local-dir checkpoints/cosmos
```

The full state dicts contain the trained video and action modules, as well as the frozen modules. The Cosmos folder supplies the tokenizer, pipeline configuration and base-model construction assets. Pass its path through the CLI; no manual edits to checkpoint files are needed.

Run two replans with synthetic observations to check loading and buffer reuse without a simulator:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_inference.py \
  --method joint --checkpoint-dir checkpoints/dit4dit/joint \
  --cosmos-path checkpoints/cosmos

CUDA_VISIBLE_DEVICES=0 python scripts/smoke_inference.py \
  --method streaming --checkpoint-dir checkpoints/dit4dit/streaming \
  --cosmos-path checkpoints/cosmos
```

## 3. LIBERO-Plus evaluation

Install [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) and its assets following its README, then install this extension's evaluation dependencies. Use the LIBERO-Plus package rather than the original LIBERO package for this evaluation.

```bash
sudo apt-get install libmagickwand-dev libegl1 libgl1  # Ubuntu system libraries
pip install -r requirements-eval.txt
pip install --no-deps -e /path/to/LIBERO-plus
python scripts/setup_benchmark.py --libero-plus /path/to/LIBERO-plus \
  --output outputs/libero-config

export LIBERO_CONFIG_PATH="$PWD/outputs/libero-config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Run one episode first; omit `--limit 1` for the full 10,030-episode manifest. Use a different output directory for a full run.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --method joint --checkpoint-dir checkpoints/dit4dit/joint \
  --cosmos-path checkpoints/cosmos --output-dir outputs/joint-smoke --limit 1

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --method streaming --checkpoint-dir checkpoints/dit4dit/streaming \
  --cosmos-path checkpoints/cosmos --output-dir outputs/streaming-smoke --limit 1
```

For parallel evaluation, start one process per GPU with the same `--num-workers N` and distinct `--worker-index 0..N-1` and output directories. Each worker checks the episode manifest, BDDL contents and initial-state identity before simulation. Errors stop the run rather than count as policy failures. Completed episodes can be resumed with the same command.

```bash
python scripts/summarize.py outputs/joint-full/worker-* \
  --output outputs/joint-full/summary.json
```

## 4. Continued training

Both recipes start from the **same upstream initialization**, not from the continued Joint weight in this release. Download that initialization from its original repository:

```bash
hf download mondo-robotics/dit4dit-model --include 'dit4dit_libero/*' \
  --local-dir checkpoints/upstream
```

Prepare the four LIBERO suites in LeRobot v3 format (20 Hz, 256 × 256 RGB views, 8D state and 7D actions). The training source has 432 / 454 / 428 / 379 episodes for Spatial / Object / Goal / LIBERO-10. The source demonstrations are the [IPEC LIBERO suites](https://github.com/Mondo-Robotics/DiT4DiT/blob/66a6f3a12e2c8740157c1e478795952c040d31dd/docs/libero.md#prepare-dataset); their current public layout is v2.1. See [data preparation](DATA.md) for conversion and the exact input contract. Do not substitute the separately regenerated FastWAM dataset and expect the same training run.

```bash
python scripts/prepare_streaming_data.py \
  --source-root /path/to/libero-v3 \
  --output-root /path/to/libero-dit4dit \
  --checkpoint-stats checkpoints/upstream/dit4dit_libero/dataset_statistics.json
```

The adapter writes metadata and symlinks, validates all low-dimensional rows, and uses the official checkpoint normalization for both methods. The prepared root contains `libero_{spatial,object,goal,10}_no_noops_1.0.0_lerobot`.

```bash
# Ordinary Joint continuation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --method joint \
  --checkpoint checkpoints/upstream/dit4dit_libero/final_model/pytorch_model.pt \
  --cosmos-path checkpoints/cosmos --data-root /path/to/libero-dit4dit \
  --output runs/joint --stop-after 10000

# Streaming continuation, independently initialized from the same upstream model
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --method streaming \
  --checkpoint checkpoints/upstream/dit4dit_libero/final_model/pytorch_model.pt \
  --cosmos-path checkpoints/cosmos --data-root /path/to/libero-dit4dit \
  --output runs/streaming --stop-after 10000
```

**Keep the 20,000-update cosine schedule.** `--stop-after 10000` stops at the released point without shortening that schedule. Changing `--max-steps` to 10,000 would change the learning rates and would be a different recipe. `--preflight-only` checks paths and configuration without starting training. Inference weights and full optimizer/RNG resume states are saved separately; `--resume` accepts only the latter.

An optional lossless decoded-frame cache reduces repeated video decoding:

```bash
python scripts/build_frame_cache.py --help
```

Supply the resulting directory with `--frame-cache`. This changes I/O only. CPU contract tests are available with `python -m pytest tests -q`. See [release validation](results/release_validation.json) for the loading, simulator, data-adapter and one-update training checks performed on both variants.

## Source and licenses

- `vendor/streaming`: DiT4DiT with temporal visual context, clean action prefix, token-wise flow matching, rolling inference and aligned LeRobot v3 loading; used for both training recipes and streaming inference.
- `vendor/joint`: the frozen native inference implementation used for the ordinary Joint result.
- `vendor/common`: native policy preprocessing and transport helpers.
- `release.json`: exact weight identities, budgets and evaluation-manifest hashes.

The two inference implementations are retained to reproduce the reported comparison without changing the ordinary policy's preprocessing. They run in separate Python processes.

Built on NVIDIA Cosmos. DiT4DiT source is MIT licensed; NVIDIA/GR00T-derived data utilities retain their Apache-2.0 notices. Upstream DiT4DiT weights are Apache-2.0 licensed, subject to the underlying [NVIDIA Open Model License](licenses/NVIDIA-Open-Model-License.txt) for Cosmos components. See [NOTICE](NOTICE.md) and the included license copies.
