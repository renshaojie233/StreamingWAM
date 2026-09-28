#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:-${ROOT}/checkpoints/streamingwam-libero/streamingwam_libero.pt}"
OUTPUT_DIR="${2:-${ROOT}/evaluate_results/smoke_test}"
TASK_FILE="$(mktemp "${TMPDIR:-/tmp}/streamingwam-task.XXXXXX")"
trap 'rm -f "${TASK_FILE}"' EXIT

cd "${ROOT}"
head -n 1 assets/libero_plus_full_10030.txt > "${TASK_FILE}"

python experiments/libero/eval_libero_task_list_multi_k.py \
  task=streamingwam_libero_plus \
  ckpt="${CHECKPOINT}" \
  EVALUATION.dataset_stats_path=assets/libero_dataset_stats.json \
  +EVALUATION.task_list_file="${TASK_FILE}" \
  +EVALUATION.sdp_k_values='[4]' \
  EVALUATION.num_trials=1 \
  EVALUATION.action_horizon=32 \
  EVALUATION.num_inference_steps=1 \
  +EVALUATION.sdp_num_noise_levels=32 \
  +EVALUATION.sdp_warm_start=true \
  +EVALUATION.sdp_warm_start_inference_steps=10 \
  +EVALUATION.save_rollout_video=false \
  EVALUATION.output_dir="${OUTPUT_DIR}"
