# LIBERO-Plus prompt correction

The original evaluator passed the benchmark's raw task description to the
policy. In LIBERO-Plus, this can include a suffix identifying a scene or
perturbation, such as `table 17`, `tb 12`, `light 3`, `add 2`, `level2 sample17`,
or a `view ... initstate ... noise ...` specification. These identifiers are
benchmark metadata, not part of the task instruction.

The corrected evaluator removes these trailing metadata fields and normalizes
whitespace before constructing the model prompt. It preserves the instruction,
including language perturbations. For example:

```text
Raw:   turn on the stove and put the moka pot on it table 17
Model: turn on the stove and put the moka pot on it
```

The environment, task IDs, initial states, and task manifest are unchanged.
This patch only changes evaluation instruction handling; it does not change
training, the rolling buffer, the K recipe, or the denoising schedule.

## Recorded results

| Instruction protocol | Successes / episodes | Success rate |
| --- | ---: | ---: |
| Legacy raw instruction (historical reference) | 7,960 / 10,030 | 79.36% |
| Clean instruction (2026-10-16 full run) | 8,237 / 10,030 | 82.12% |

The difference is **277 successes, or 2.76 percentage points**. These are two
evaluation protocols for the reference checkpoint, not a newly trained model.
The historical result remains available rather than being replaced in place.

The clean run used seed 42, K=4, a 32-step action buffer (16 clean-prefix and
16 future actions), one update per steady-state replan, and a 10-step initial
warm start. All 10,030 task records were checked against the committed manifest:
there are no missing or duplicate tasks, 5,344 instructions change, and the
ported sanitizer reproduces every recorded model instruction exactly.

| Suite | Successes / episodes | Success rate |
| --- | ---: | ---: |
| LIBERO-10 | 2,077 / 2,519 | 82.45% |
| LIBERO-Goal | 1,819 / 2,591 | 70.20% |
| LIBERO-Object | 2,250 / 2,518 | 89.36% |
| LIBERO-Spatial | 2,091 / 2,402 | 87.05% |

The [per-task CSV](liberoplus_clean_20261016.csv) contains success counts and
raw/model instructions. The [aggregate JSON](liberoplus_clean_20261016.json)
records the exact rates, asset hashes, and source evaluator hash. The clean
result was produced by the server evaluator; this release ports its prompt
transformation into the current public code instead of replacing the public
code with that server snapshot.

### Checkpoint identity

Both the historical 79.36% provenance and the new 82.12% run identify this
reference checkpoint SHA-256:

```text
7b321c0c85c6167fda5eb21f7e1fe6abe745f9b77be20eef0144b97ae2d6aa45
```

The existing Hugging Face download at revision
`0b5a17cc2b81bd91ed15b3fe3e3be7a9ad13ba48` has SHA-256
`35499c8b2ac7bc879c988c9af4f9e9ff9052caccd22d582b7fadd90de185d496`.
It is a different parameter checkpoint, not just a renamed file. The scores
above must not be attributed to that download. To reproduce these scores,
use the reference checkpoint identified above. The original download is
preserved for compatibility.

## Selecting a protocol

All LIBERO evaluation entry points accept `EVALUATION.prompt_protocol`:

- `clean` (default): use the corrected task instruction.
- `legacy`: pass the original task description through unchanged.

Append these overrides to the full evaluation command in the main README:

```bash
# Corrected instruction protocol
EVALUATION.prompt_protocol=clean EVALUATION.output_dir=evaluate_results/prompt_clean

# Original instruction protocol
EVALUATION.prompt_protocol=legacy EVALUATION.output_dir=evaluate_results/prompt_legacy
```

Use a distinct output directory for each protocol. The evaluator records the
protocol in each task result and worker summary, along with the raw and model
instructions. It rejects output directories from the other protocol, including
when `skip_existing=false`. Historical results without a protocol label are
treated as legacy; do not resume an externally generated clean run with
unlabelled results in the same directory.

The repository before this patch is preserved at
[`liberoplus-eval-v1`](https://github.com/renshaojie233/StreamingWAM/tree/liberoplus-eval-v1).
The corrected release is
[`liberoplus-eval-v2`](https://github.com/renshaojie233/StreamingWAM/tree/liberoplus-eval-v2).
Both modes are also available from the corrected release without switching
branches.

## Validation

```bash
python -B -m unittest -v tests/test_libero_prompts.py
```

Tests cover all 10,030 recorded prompt transformations, exact legacy
passthrough, task-runner instruction forwarding, language preservation, and
resume protection. The corrected evaluation entry point also passed an import and Hydra
configuration check in the server environment. These checks do not constitute
a new full simulator run of the public code.
