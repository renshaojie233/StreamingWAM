# StreamingWAM real-robot reference

This directory contains the model-side real-robot reference used for the
seven-task experiments.  It intentionally exposes the plain StreamingWAM
rolling policy rather than the hardware-specific candidates evaluated during
commissioning.

The released path has:

- five consecutive 15 Hz RGB observations;
- camera order `camera_1`, `camera_3`, `camera_2`;
- per-camera resize to `224 x 128`, followed by vertical concatenation to
  `3 x 384 x 224` and normalization to `[-1, 1]`;
- a 16-action clean prefix containing actions that were actually issued;
- a 16-action predicted future suffix;
- a 10-step episode-initial warm start and one denoising update thereafter;
- raw time-aligned selection from the latest rolling result;
- the recorded action conversion
  `qdot = native_action * 0.2 * 15` in radians per second.

It deliberately does **not** contain the latency-shift checkpoints, lead-tick
submission, plan-transition blending, trajectory interpolation/smoothing, or
task-specific gripper gates used in exploratory hardware studies.  The clean
reference only projects the continuous gripper command to its physical
`[0, 1]` interval and rejects non-finite or out-of-contract arm actions.

## Model assets

The final 30k real-world checkpoint and its runtime assets are hosted in the
public [realworld-v1 GitHub release](https://github.com/renshaojie233/StreamingWAM/releases/tag/realworld-v1).
The 12.04 GB checkpoint is split into seven release assets because GitHub
requires each individual release file to be smaller than 2 GiB. The download
script reassembles it and verifies both the parts and final checkpoint:

| File | SHA-256 |
| --- | --- |
| `step_030000.pt` | `ed68b39a9d7025188e6216a9336a3e265893d360b0c02b91f31629c317d1f66a` |
| `train_split_zscore_stats.json` | `e9809ddb35074157098074f3f80506775de7875c02a5e896ac66f05ddadaf997` |

The sanitized model-only configuration is versioned with the code at
`real_robot/assets/streamingwam_realworld_model.yaml`.  The exact resolved
formal-training configuration had SHA-256
`86e2f62a8ca8a26b46b3e8677aa3387466a051521b53a6a5390dfe7b1da454ad`;
private machine paths are intentionally not reproduced.

Download it with:

```bash
python real_robot/download_assets.py --output checkpoints/streamingwam-realworld
```

The Wan 2.2 TI2V 5B base assets are not redistributed.  Point
`DIFFSYNTH_MODEL_BASE_PATH` at the same pretrained asset layout used by the
training code.

## Hardware adapter

Robot installations differ, so low-level camera capture, reset, watchdog, and
servo control are kept behind the small `RobotIO` protocol in
`streamingwam_deploy/io.py`.  An adapter factory must return one object with:

```python
observe() -> Observation
reset(joint_position_rad, gripper_command)
start()
command(joint_velocity_rad_s, gripper_command)
stop()
```

`observe()` must return synchronized RGB images and measured joint positions.
The state gripper coordinate is the most recently commanded continuous value,
matching the training data (`0=closed`, `1=open`).  The adapter owns all
platform safety checks; the model loop never restarts or retries a motion
command.

Pass the factory as `package.module:create_io`.  The factory receives the
parsed command-line arguments, so a lab can add robot addresses or camera
serials without placing them in this public repository.

## Run

Shadow mode performs model inference and timing but sends no commands:

```bash
python -m real_robot.streamingwam_deploy.run \
  --assets checkpoints/streamingwam-realworld \
  --io-factory your_robot.streamingwam_io:create_io \
  --task "put the apple into the bowl" \
  --replan-k 4 \
  --shadow
```

Real execution is explicit:

```bash
python -m real_robot.streamingwam_deploy.run \
  --assets checkpoints/streamingwam-realworld \
  --io-factory your_robot.streamingwam_io:create_io \
  --task "put the apple into the bowl" \
  --replan-k 4 \
  --execute --reset --confirm-real-robot
```

The formally trained replanning horizons are `K={1,2,4,8,10}`.  This public
runner defaults to `K=4`, the main real-robot setting.  If a rolling update is
not ready at its exact K-step boundary, the reference issues zero joint
velocity while retaining the last gripper command; it never skips forward,
replays a failed command, or silently changes the trajectory.

## Validate without hardware

```bash
python -m unittest discover -s real_robot/tests -v
python -m compileall -q real_robot
```

These tests cover the camera layout, action conversion, rolling-result time
alignment, prefix padding, and the absence of shift/smoothing controls.
