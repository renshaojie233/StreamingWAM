# Training data

The two continuation runs use the same four IPEC LIBERO no-noop suites, converted to LeRobot v3, with no additional demonstrations. State order is position (3), axis-angle orientation (3), and the two gripper joint positions (2). Action order is translation (3), rotation (3), and binary gripper (1). The gripper action is already 0/1; do not invert it or normalize it again during data preparation.

| Suite | Episodes | Frames |
| --- | ---: | ---: |
| Spatial | 432 | 52,970 |
| Object | 454 | 66,984 |
| Goal | 428 | 52,042 |
| LIBERO-10 | 379 | 101,469 |

Download the upstream v2.1 layout using the dataset revisions recorded in `configs/data_sources.json`:

```bash
python scripts/download_data.py --output data/ipec
```

Create a v3-compatible file view without re-encoding videos or modifying any low-dimensional values:

```bash
python scripts/import_libero_v21.py --source-root data/ipec --output-root data/libero-v3
python scripts/prepare_streaming_data.py \
  --source-root data/libero-v3 --output-root data/libero-dit4dit \
  --checkpoint-stats checkpoints/upstream/dit4dit_libero/dataset_statistics.json
```

Keep the source folders: the prepared views use symlinks. `prepare_streaming_data.py` validates all state/action rows and requires the raw binary gripper convention. It writes a normalization reference from the official checkpoint rather than recomputing different scales for each suite. Both training commands use `--data-root data/libero-dit4dit`.

If you already have a v3 conversion, the import step can be skipped. Each camera's episode file index and `from_timestamp` must be correct; the loader applies its own offset and does not use wrist-camera timestamps for both cameras. Source paths, file grouping and video packaging may differ between conversions, so check decoded frames as well as row counts. No timestamp, action, RGB channel or camera-order transformation is performed by the v2.1 importer.

The current regenerated `yuanty/LIBERO-fastwam` dataset has different counts and image resolution. It is useful for the FastWAM release, but is not this DiT4DiT training source.
