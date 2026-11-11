"""Reproduce the released DiT4DiT LIBERO-Plus comparison (raw instructions, K=4)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import ROOT, activate, load_model, sha256_file as sha
import numpy as np
import torch


def atomic(path, value):
    temp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def state_identity(value):
    a = np.ascontiguousarray(value)
    assert a.ndim == 1 and a.size > 0 and np.isfinite(a).all()
    return {'dtype': a.dtype.str, 'shape': list(a.shape),
            'sha256': hashlib.sha256(a.tobytes()).hexdigest()}


def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    quat[3] = np.clip(quat[3], -1, 1)
    den = np.sqrt(max(1 - float(quat[3]) ** 2, 0))
    return np.zeros(3, np.float32) if den < 1e-6 else quat[:3] * (2 * np.arccos(quat[3])) / den


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('joint', 'streaming'), required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--cosmos-path', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--limit', type=int, help='Run only this many episodes per worker (smoke test)')
    p.add_argument('--smoke-cases', type=Path, help='JSON list of [suite, task_index] pairs')
    a = p.parse_args()
    a.k = 4
    a.label = a.method
    assert 0 <= a.worker_index < a.num_workers
    if a.limit is not None and a.limit <= 0:
        p.error('--limit must be positive')
    root, out = ROOT, a.output_dir
    release = json.loads((root / 'release.json').read_text())
    model_proof = release['checkpoints'][a.method]
    proof = release['evaluation']
    manifest = root / 'evaluation/episodes.jsonl'
    states_path = root / 'evaluation/initial_states.jsonl'
    assert sha(manifest) == proof['manifest_sha256']
    assert sha(states_path) == proof['state_manifest_sha256']
    rows = list(map(json.loads, manifest.read_text().splitlines()))
    keys = {(r['suite'], r['task_index'], r['init_state_index']) for r in rows}
    assert len(rows) == len(keys) == proof['expected_episodes']
    states = {(r['suite'], r['task_index'], r['init_state_index']): r['first_state']
              for r in map(json.loads, states_path.read_text().splitlines())}
    assert set(states) == keys
    protocol = {'method': a.method, 'label':a.label, 'checkpoint_sha256': model_proof['sha256'],
        'manifest_sha256': proof['manifest_sha256'], 'state_manifest_sha256': proof['state_manifest_sha256'],
        'release_sha256': sha(root / 'release.json'),
        'prompt_protocol': 'unmodified official task.language including suffixes',
        'seed': 7, 'k': a.k, 'control_hz': 20, 'settling_steps': 5, 'max_steps': 700,
        'future_actions': 8, 'action_denoising_steps': 4 if a.method == 'joint' else 1,
        'warm_start_steps': None if a.method == 'joint' else 4,
        'worker_index': a.worker_index, 'num_workers': a.num_workers,
        'limit': a.limit, 'smoke_cases': json.loads(a.smoke_cases.read_text()) if a.smoke_cases else None}
    activate(a.method)
    if a.method == 'streaming':
        from DiT4DiT.model.modules.action_model.streaming import noise_ladder
        pre,post=noise_ladder(execution_steps=a.k)
        protocol['sigma_pre'],protocol['sigma_post']=pre.tolist(),post.tolist()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        assert json.loads((out / 'protocol.json').read_text()) == protocol
    else:
        atomic(out / 'protocol.json', protocol)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    model = load_model(a.method, a.checkpoint_dir, a.cosmos_path)
    action_statistics = model.norm_stats['franka']['action']
    traces = []
    def predict(examples):
        if a.method == 'streaming':
            e = examples[0]
            assert len(e['image']) == 5 and np.asarray(e['action_prefix']).shape == (8,8)
            traces.append({'prefix': np.array(e['action_prefix'], copy=True),
                           'carried': e['action_buffer'] is not None})
            return model.predict_streaming_action(examples)
        return model.predict_action(examples)
    if a.method == 'streaming':
        from examples.LIBERO.eval_files.streaming_policy import StreamingLiberoPolicy
        policy = StreamingLiberoPolicy(predict, action_statistics, a.k)
    else:
        from fair_policy import OfficialPolicy
        policy = OfficialPolicy(predict, action_statistics, a.k)
    atomic(out / 'model_loaded.json', {'strict': True, 'tensors': len(model.state_dict()), **protocol})
    suites = {s: benchmark.get_benchmark_dict()[s]() for s in sorted({r['suite'] for r in rows})}
    selected = rows[a.worker_index::a.num_workers]
    if a.smoke_cases:
        assert a.num_workers == 1 and a.worker_index == 0
        cases = {tuple(c) for c in protocol['smoke_cases']}
        selected = [r for r in rows if (r['suite'], r['task_index']) in cases]
        assert len(selected) == len(cases)
    if a.limit is not None:
        selected = selected[:a.limit]
    for entry in selected:
        episode = out / ('%s_%05d' % (entry['suite'], entry['task_index']))
        episode.mkdir(exist_ok=True)
        target = episode / 'result.json'
        if target.exists():
            old = json.loads(target.read_text())
            assert old['error'] is None and old['checkpoint_sha256'] == model_proof['sha256']
            assert old['manifest_sha256'] == proof['manifest_sha256'] and old['method'] == a.method
            assert sha(episode / 'actions.npz') == old['actions_sha256']
            continue
        env = None
        traces.clear()
        actions, buffers = [], []
        task = suites[entry['suite']].get_task(entry['task_index'])
        assert task.name == entry['task_name']
        assert entry['prompt'] == entry['prompt_original'] == task.language
        bddl = Path(get_libero_path('bddl_files')) / entry['bddl_relative']
        assert sha(Path(get_libero_path('bddl_files')) / entry['resolved_bddl_relative']) == entry['bddl_sha256']
        np.random.seed(7)
        torch.manual_seed(7)
        started = time.monotonic()
        success = False
        try:
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, control_freq=20)
            env.seed(7)
            env.reset()
            initial = suites[entry['suite']].get_task_init_states(entry['task_index'])[entry['init_state_index']]
            identity = state_identity(initial)
            assert identity == states[(entry['suite'], entry['task_index'], entry['init_state_index'])]
            obs = env.set_init_state(initial)
            for _ in range(5):
                obs, _, _, _ = env.step([0.] * 6 + [-1.])
            policy.reset(entry['prompt'])
            for step in range(700):
                primary = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
                wrist = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
                state = np.concatenate([obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']])
                with torch.inference_mode():
                    action = policy.act(primary, wrist, state, entry['prompt'])
                assert action.shape == (7,) and np.isfinite(action).all()
                if a.method == 'streaming' and step % a.k == 0:
                    buffers.append(policy.buffer.copy())
                    assert traces[-1]['carried'] == (step != 0)
                    if step == 0:
                        assert not traces[-1]['prefix'].any()
                actions.append(action.copy())
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    success = True
                    break
            extra = ({'action_buffers': np.asarray(buffers),
                      'normalized_executed_prefixes': np.asarray([t['prefix'] for t in traces])}
                     if a.method == 'streaming' else {'predicted_chunks': np.asarray(policy.chunks)})
            with (episode / 'actions.npz.tmp').open('wb') as f:
                np.savez_compressed(f, actions=np.asarray(actions), chunk_start_steps=np.arange(0,len(actions),a.k), **extra)
            (episode / 'actions.npz.tmp').replace(episode / 'actions.npz')
            result = {**entry, 'method': a.method, 'label':a.label, 'k':a.k,'success': success, 'steps': len(actions),
                'error': None, 'initial_state_identity': identity,
                'seconds': time.monotonic()-started, 'checkpoint_sha256': model_proof['sha256'],
                'manifest_sha256': proof['manifest_sha256'],
                'actions_sha256': sha(episode / 'actions.npz')}
            atomic(target, result)
            print(json.dumps({'method': a.method,'episode': episode.name,'success': success,'steps': len(actions)}),flush=True)
        except Exception as exc:
            atomic(episode / ('error_%d.json' % time.time_ns()), {'error': repr(exc), 'traceback': traceback.format_exc(), 'entry': entry})
            raise
        finally:
            if env is not None:
                env.close()
    atomic(out / 'worker_complete.json', {'completed': True, 'episodes': len(selected), **protocol})


if __name__ == '__main__':
    main()
