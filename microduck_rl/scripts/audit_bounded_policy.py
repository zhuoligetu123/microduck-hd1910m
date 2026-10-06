#!/usr/bin/env python3
"""CPU checkpoint/ONNX parity and supported-entry screening; no hardware I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import mjlab  # Finish task discovery before importing local actuator modules.
from tensordict import TensorDict
from rsl_rl.models.mlp_model import MLPModel
from mjlab_microduck.actuator.bounded_position import CONTRACT, SLEW_CONTRACT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--observations', type=Path, help='Optional archived real bench JSONL')
    p.add_argument('--native-log', type=Path, help='Local policy_offline JSONL; no hardware samples')
    args = p.parse_args()
    torch.set_num_threads(2)
    session = ort.InferenceSession(str(args.policy), providers=['CPUExecutionProvider'])
    meta = session.get_modelmeta().custom_metadata_map
    contract = meta.get('action_semantics')
    if contract not in (CONTRACT,SLEW_CONTRACT) or meta.get('previous_action_semantics') != contract:
        raise ValueError('bounded action/history contract missing')
    low = np.asarray(json.loads(meta['action_delta_low']), dtype=np.float32)
    high = np.asarray(json.loads(meta['action_delta_high']), dtype=np.float32)
    if low.shape != (14,) or high.shape != (14,) or not np.all(low < high):
        raise ValueError('invalid output limits')
    actor = MLPModel(TensorDict({'actor': torch.zeros(1,61)}, batch_size=[1]),
                     {'actor':['actor']}, 'actor', 14, hidden_dims=(512,256,128),
                     activation='elu', obs_normalization=True,
                     distribution_cfg={'class_name':'GaussianDistribution', 'init_std':1., 'std_type':'scalar'})
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
    actor.eval()
    observations = np.random.default_rng(123).normal(0,.01,(201,61)).astype(np.float32)
    observations[:,5] = -1
    observations[:,3:6] /= np.linalg.norm(observations[:,3:6],axis=1,keepdims=True)
    observations[0] = 0
    observations[0,5] = -1
    recorded = []
    if args.observations:
        for line in args.observations.read_text().splitlines():
            if line.startswith('{'):
                row = json.loads(line)
                if row.get('phase')=='rl_proposal' or row.get('event')=='policy_check':
                    recorded.append(row['observation'])
    if recorded:
        observations = np.concatenate((observations,np.asarray(recorded,dtype=np.float32)))
    native=[]
    if args.native_log:
        native=[json.loads(line) for line in args.native_log.read_text().splitlines() if line.strip()]
        if not native or any(r.get('seq')!=i or r.get('hardware_opened') is not False for i,r in enumerate(native)):
            raise ValueError('invalid native offline sequence')
        native_obs=np.asarray([r['observation'] for r in native],dtype=np.float32)
        if native_obs.shape!=(len(native),61) or not np.isfinite(native_obs).all():
            raise ValueError('invalid native observations')
        observations=np.concatenate((observations,native_obs))
    with torch.inference_mode():
        latent = actor(TensorDict({'actor':torch.from_numpy(observations)},batch_size=[len(observations)])).numpy()
    expected = np.clip(latent,low,high)
    step = float(meta['max_action_step_rad']) if contract == SLEW_CONTRACT else None
    if step is not None:
        expected = np.clip(expected, observations[:,34:48]-step, observations[:,34:48]+step)
    actual = np.concatenate([session.run(None,{'obs':row[None]})[0] for row in observations])
    error = float(np.max(np.abs(actual-expected)))
    finite = bool(np.isfinite(actual).all())
    count = int(np.sum(np.any((actual < low-1e-6) | (actual > high+1e-6), axis=1)))
    result = dict(checkpoint_iteration=checkpoint['iter'], samples=len(actual),
                  archived_real_observations=len(recorded), max_abs_parity_error=error,
                  parity_passed=finite and error<1e-5, output_bound_violations=count,
                  latent_saturated_fraction=float(np.mean((latent<low)|(latent>high))),
                  nominal_action=actual[0].tolist(), nominal_max_step_rad=float(np.max(np.abs(actual[0]))),
                  supported_entry_passed=bool(np.max(np.abs(actual[0]))<=.12),
                  hardware_tested=False, deployment_ready=False,
                  action_semantics=contract,
                  policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    native_passed=True
    if native:
        native_actions=np.asarray([r['action'] for r in native],dtype=np.float32)
        if native_actions.shape!=(len(native),14):
            raise ValueError('invalid native action shape')
        native_error=float(np.max(np.abs(actual[-len(native):]-native_actions)))
        native_passed=bool(np.isfinite(native_actions).all() and native_error<1e-5)
        native_result=dict(samples=len(native),max_abs_parity_error=native_error,parity_passed=native_passed,
                           hardware_tested=False,hardware_opened=False,policy_sha256=result['policy_sha256'],
                           fixture='fixed_home_with_action_history',action_semantics=contract)
        (args.report.parent/'native_parity.json').write_text(json.dumps(native_result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if result['parity_passed'] and count==0 and native_passed else 1


if __name__=='__main__':
    raise SystemExit(main())
