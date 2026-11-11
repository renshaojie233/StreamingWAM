"""Load real weights and run two replans without a simulator or robot connection."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', required=True, choices=('joint', 'streaming'))
    p.add_argument('--checkpoint-dir', required=True, type=Path)
    p.add_argument('--cosmos-path', required=True, type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    import json
    import numpy as np
    import torch
    torch.manual_seed(7)
    np.random.seed(7)
    model = load_model(args.method, args.checkpoint_dir, args.cosmos_path)
    statistics = model.norm_stats['franka']['action']
    calls = []
    def predict(examples):
        with torch.inference_mode():
            if args.method == 'streaming':
                calls.append(examples[0]['action_buffer'] is not None)
                return model.predict_streaming_action(examples)
            calls.append(True)
            return model.predict_action(examples)
    if args.method == 'joint':
        from fair_policy import OfficialPolicy
        policy = OfficialPolicy(predict, statistics, 4)
    else:
        from examples.LIBERO.eval_files.streaming_policy import StreamingLiberoPolicy
        policy = StreamingLiberoPolicy(predict, statistics, 4)
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (256,256,3), dtype=np.uint8)
    actions = []
    for _ in range(8):
        actions.append(policy.act(image, image, np.zeros(8, np.float32), 'pick up the black bowl'))
    actions = np.stack(actions)
    assert actions.shape == (8,7) and np.isfinite(actions).all()
    assert len(calls) == 2
    if args.method == 'streaming':
        assert calls == [False, True]
        assert policy.buffer.shape == (8,8)
        assert np.isfinite(policy.buffer).all()
    summary = {'method': args.method, 'strict_tensors': len(model.state_dict()),
               'replans': 2, 'commands': 8, 'all_finite': True,
               'peak_cuda_gib': torch.cuda.max_memory_allocated() / 2**30,
               'scope': 'synthetic observation smoke test, not a benchmark evaluation'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
