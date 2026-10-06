#!/usr/bin/env python3
"""Initialize the local bounded actor from Luwu's raw ONNX; no hardware access.

This is weight transfer, not PPO resume: upstream critic/optimizer and sample
count are not in ONNX. Applied-action history differs from upstream raw history
once local bounds activate; simulation adaptation is required before deployment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch


def transfer(source, template_policy, template_checkpoint, output):
    upstream, local = onnx.load(source), onnx.load(template_policy)
    meta = {p.key: p.value for p in upstream.metadata_props}
    target_meta = {p.key: p.value for p in local.metadata_props}
    for key in ('joint_names', 'observation_names', 'action_scale'):
        if meta.get(key) != target_meta.get(key):
            raise ValueError(f'incompatible {key}')
    if ([n.op_type for n in upstream.graph.node] !=
            ['Sub', 'Div', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm']):
        raise ValueError('only the audited normalized ELU actor is supported')
    if target_meta.get('action_semantics') != 'bounded_slew_home_delta_v2':
        raise ValueError('expected the existing local bounded-slew template')
    old_home = np.fromstring(meta['default_joint_pos'], sep=',', dtype=np.float32)
    home = np.fromstring(target_meta['default_joint_pos'], sep=',', dtype=np.float32)
    if old_home.shape != (14,) or home.shape != (14,):
        raise ValueError('invalid HOME dimensions')
    delta = home - old_home
    shift = np.zeros((1, 61), dtype=np.float32)
    shift[:, 6:20] = delta
    shift[:, 34:48] = delta
    shift[:, 51:55] = delta[5:9]
    arrays = {v.name: numpy_helper.to_array(v).copy() for v in upstream.graph.initializer}
    arrays['obs_normalizer._mean'] -= shift
    arrays['mlp.6.bias'] -= delta
    divisor = upstream.graph.node[1].input[1]
    denominator = arrays.pop(divisor)
    arrays[local.graph.node[1].input[1]] = denominator
    for index, value in enumerate(local.graph.initializer):
        if value.name in arrays:
            new = arrays[value.name]
            if tuple(value.dims) != new.shape or not np.isfinite(new).all():
                raise ValueError(f'invalid tensor {value.name}')
            local.graph.initializer[index].CopyFrom(numpy_helper.from_array(new, value.name))
    torch.manual_seed(2026)
    checkpoint = torch.load(template_checkpoint, map_location='cpu', weights_only=False)
    actor = checkpoint['actor_state_dict']
    for name in actor:
        if name in arrays:
            actor[name] = torch.from_numpy(arrays[name].copy())
    std = denominator - .01  # EmpiricalNormalization's epsilon, included in ONNX Div.
    if np.any(std < -1e-7):
        raise ValueError('normalizer denominator is incompatible with epsilon .01')
    actor['obs_normalizer._std'] = torch.from_numpy(np.maximum(std, 0))
    actor['obs_normalizer._var'] = actor['obs_normalizer._std'].square()
    actor['obs_normalizer.count'].fill_(100000)  # Explicit prior mass, not a recovered sample count.
    actor['distribution.std_param'].fill_(.2)
    critic = checkpoint['critic_state_dict']
    for name, value in critic.items():
        if name.endswith('weight'):
            torch.nn.init.orthogonal_(value, gain=.01 if name == 'mlp.6.weight' else 1.)
        elif name in ('obs_normalizer._var', 'obs_normalizer._std'):
            value.fill_(1.)
        else:
            value.zero_()
    checkpoint['optimizer_state_dict']['state'] = {}
    checkpoint['iter'] = 0
    checkpoint['infos'] = None
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    target_meta.update(training_recipe='luwu_actor_initialization_v1', source_actor_sha256=digest,
        initialization_only='true', hardware_tested='false', deployment_ready='false')
    onnx.helper.set_model_props(local, target_meta)
    onnx.checker.check_model(local)
    source_session = ort.InferenceSession(upstream.SerializeToString(), providers=['CPUExecutionProvider'])
    session = ort.InferenceSession(local.SerializeToString(), providers=['CPUExecutionProvider'])
    low, high = (np.asarray(json.loads(target_meta[k])) for k in ('action_delta_low', 'action_delta_high'))
    step = float(target_meta['max_action_step_rad'])
    error = 0.
    for obs in np.random.default_rng(2026).normal(0, .2, (200, 1, 61)).astype(np.float32):
        obs[:, 3:6] = [0, 0, -1]
        raw = source_session.run(None, {'obs': obs + shift})[0] - delta
        expected = np.clip(np.clip(raw, low, high), obs[:, 34:48]-step, obs[:, 34:48]+step)
        actual = session.run(None, {'obs': obs})[0]
        if not np.isfinite(raw).all() or not np.isfinite(actual).all():
            raise ValueError('nonfinite transfer output')
        error = max(error, float(abs(expected-actual).max()))
    if error > 1e-5:
        raise ValueError(f'transfer parity failed: {error}')
    output.mkdir(parents=True, exist_ok=False)
    onnx.save(local, output/'policy.onnx')
    torch.save(checkpoint, output/'model.pt')
    report = dict(source_actor_sha256=digest, samples=200, max_abs_error=error,
        home_delta_rad=delta.tolist(), optimizer_reset=True, critic_reinitialized=True,
        normalization_prior_count=100000, exploration_std=.2,
        history='local applied target; differs from upstream raw history when bounded',
        hardware_tested=False, deployment_ready=False)
    (output/'transfer.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--template-policy', type=Path, required=True)
    parser.add_argument('--template-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    transfer(args.source, args.template_policy, args.template_checkpoint, args.output)
