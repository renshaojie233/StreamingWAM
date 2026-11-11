"""Release-level checks: checkpoint identities, matched budgets and native action bridge."""
import importlib.util
import json
from pathlib import Path
import sys
from argparse import Namespace
import numpy as np
import pytest
from bootstrap import ROOT, sha256_file


def test_frozen_runtime_integrity():
    entries = json.loads((ROOT / 'vendor/SOURCE_SHA256.json').read_text())
    for path, digest in entries.items():
        assert sha256_file(ROOT / path) == digest


def test_evaluation_manifest_and_results():
    r = json.loads((ROOT / 'release.json').read_text())
    rows = [json.loads(s) for s in (ROOT / 'evaluation/episodes.jsonl').read_text().splitlines()]
    assert len(rows) == len({(x['suite'], x['task_index'], x['init_state_index']) for x in rows}) == 10030
    assert all(x['prompt'] == x['prompt_original'] for x in rows)
    assert sha256_file(ROOT / 'evaluation/episodes.jsonl') == r['evaluation']['manifest_sha256']
    assert sha256_file(ROOT / 'evaluation/initial_states.jsonl') == r['evaluation']['state_manifest_sha256']
    assert r['checkpoints']['joint']['successful_episodes'] == 7109
    assert r['checkpoints']['streaming']['successful_episodes'] == 7660
    assert r['checkpoints']['joint']['sha256'] != r['checkpoints']['streaming']['sha256']


def test_training_recipes_share_budget_and_preserve_schedule(tmp_path):
    spec = importlib.util.spec_from_file_location('release_train', ROOT / 'scripts/train.py')
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    configs = []
    for method in ('joint', 'streaming'):
        args = train.parser().parse_args(['--method', method, '--checkpoint', str(tmp_path/'init.pt'),
                  '--cosmos-path', str(tmp_path), '--data-root', str(tmp_path), '--output', str(tmp_path/method)])
        cfg = train.resolve_config(args)
        assert cfg.trainer.max_train_steps == 20000 and args.stop_after == 10000
        assert cfg.datasets.vla_data.per_device_batch_size * cfg.trainer.gradient_accumulation_steps * 4 == 64
        assert cfg.framework.action_model.action_horizon == 8
        assert cfg.framework.cosmos25.base_model == str(tmp_path)
        configs.append(cfg)
    for field in ('learning_rate','num_warmup_steps','min_lr','optimizer','gradient_clipping','mixed_precision'):
        assert configs[0].trainer[field] == configs[1].trainer[field]
    assert configs[0].datasets.vla_data.streaming is False
    assert configs[1].datasets.vla_data.streaming is True


def test_native_policy_transforms_and_replans():
    from fair_policy import OfficialPolicy
    calls=[]
    chunk=np.zeros((1,8,8),dtype=np.float32)
    chunk[0,:,0]=np.arange(8)/10
    chunk[0,0,6]=0.5
    chunk[0,1,6]=0.49
    def predict(examples):
        calls.append(examples[0])
        return {'normalized_actions':chunk.copy()}
    stats={'min':[-2.]*7,'max':[2.]*7,'mask':[True]*6+[False]}
    policy=OfficialPolicy(predict,stats,4)
    image=np.zeros((256,256,3),np.uint8)
    state=np.arange(8,dtype=np.float32)/10
    actions=[policy.act(image,image,state,'keep raw suffix 1') for _ in range(8)]
    assert len(calls)==2
    assert all(len(x['image'])==1 and x['image'][0].shape==(224,448,3) for x in calls)
    np.testing.assert_allclose(calls[0]['state'][0],np.stack([np.sin(state),np.cos(state)],axis=-1).reshape(-1))
    np.testing.assert_allclose(np.array(actions)[:,0],[0,.2,.4,.6,0,.2,.4,.6],atol=1e-6)
    assert actions[0][6]==-1 and actions[1][6]==1
    assert all(x['lang']=='keep raw suffix 1' for x in calls)
