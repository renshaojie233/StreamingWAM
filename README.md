# StreamingWAM: LIBERO-Plus training and evaluation

This repository is the first code release for **StreamingWAM**. It contains the
training and rolling-inference implementation used for the LIBERO-Plus
experiments. Real-robot and RoboTwin code are outside the scope of this release.

StreamingWAM keeps a rolling action buffer and updates it with a token-wise
staircase noise schedule. The released configuration uses a 16-step clean
action prefix, a 16-step predicted suffix, one Euler update per replanning call,
and a 10-step sampler only for the episode-initial warm start.

The code is based on the MIT-licensed
[FastWAM repository](https://github.com/yuantianyuan01/FastWAM). See
[NOTICE](NOTICE) for attribution and [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
for the released protocol.

## Release scope

- StreamingWAM training on the four LIBERO suites used by LIBERO-Plus.
- Rolling evaluation over an arbitrary task manifest.
- The 10,030-episode LIBERO-Plus manifest and normalization statistics used in
  the reported run.
- Exact paper hyperparameters and a machine-checkable release manifest.

The model checkpoint is not included in this source-only release. Training
starts from a FastWAM-Joint checkpoint; pass its path through `resume=...`.

## Installation

The reported environment used Python 3.10, PyTorch 2.7.1, CUDA 12.8, and
MuJoCo 3.3.2.

```bash
conda create -n streamingwam python=3.10 -y
conda activate streamingwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install mujoco==3.3.2
```

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and the
LIBERO-Plus task definitions in the same environment. The preprocessed LIBERO
data expected by the default configuration follow the layout published with
[FastWAM](https://huggingface.co/datasets/yuanty/LIBERO-fastwam):

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

Set the Wan model cache and create the ActionDiT initialization before
training:

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

## Training

Precompute the instruction embeddings, then launch the exact four-GPU
fine-tuning configuration. A `.pt` path passed to `resume` loads model weights
only and starts a fresh optimizer and scheduler at step zero.

```bash
python scripts/precompute_text_embeds.py task=streamingwam_libero_plus

bash scripts/train_zero1.sh 4 \
  task=streamingwam_libero_plus \
  resume=/path/to/fastwam_joint_step_034720.pt
```

The reported checkpoint is `step_040000.pt`. Training was configured for at
most 60,000 steps; checkpoint selection used the independent 200-episode
development subset described in the paper.

## Evaluation

This command reproduces the reported `K=4` rolling-inference setting on the
included 10,030-episode manifest:

```bash
python experiments/libero/eval_libero_task_list_multi_k.py \
  task=streamingwam_libero_plus \
  ckpt=/path/to/step_040000.pt \
  EVALUATION.dataset_stats_path=assets/libero_dataset_stats.json \
  +EVALUATION.task_list_file=assets/libero_plus_full_10030.txt \
  +EVALUATION.sdp_k_values='[4]' \
  EVALUATION.num_trials=1 \
  EVALUATION.action_horizon=32 \
  EVALUATION.num_inference_steps=1 \
  +EVALUATION.sdp_num_noise_levels=32 \
  +EVALUATION.sdp_warm_start=true \
  +EVALUATION.sdp_warm_start_inference_steps=10 \
  +EVALUATION.save_rollout_video=false
```

For a robustness sweep, use
`+EVALUATION.sdp_k_values='[1,2,4,8,16]'`. The released training distribution
contains `K={1,2,4,8}`; `K=16` is the unseen execution horizon.

To shard the manifest across processes, add
`+EVALUATION.num_workers=N +EVALUATION.worker_index=i` and assign one GPU to
each process with `gpu_id=i`. Each process loads the checkpoint once and writes
its own summary.

## Validation

```bash
python scripts/check_release.py
python -m compileall -q src experiments scripts
```

The release check rejects private absolute paths, backup files, generated
bytecode, archives, and common credential patterns.

## License

The repository is released under the MIT License. External datasets, model
weights, and benchmark assets retain their own licenses and terms.
