#!/usr/bin/env python3
"""Audit archived native telemetry against ONNX without hardware access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def audit(path, session):
    names = session.get_modelmeta().custom_metadata_map['joint_names'].split(',')
    output_errors, current_errors, previous_errors, heads = [], [], [], []
    last = None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get('policy') != 'walk' or not row.get('inference'):
            last = None
            continue
        inf, fb = row['inference'], row['feedback']
        indices = {joint['name']: i for i, joint in enumerate(fb['joints'])}
        velocity = np.asarray([fb['velocities'][indices[name]] for name in names])
        obs = np.asarray(inf['observation'], dtype=np.float32)
        action = np.asarray(inf['action'], dtype=np.float32)
        if (obs.shape != (61,) or action.shape != (14,)
                or not np.isfinite(obs).all() or not np.isfinite(action).all()):
            raise ValueError('invalid archived observation/action')
        predicted = session.run(None, {'obs': obs[None]})[0][0]
        output_errors.append(float(np.max(np.abs(predicted-action))))
        current_errors.append(float(np.max(np.abs(obs[20:34]-velocity))))
        if last and inf['cycle'] == last[0] + 1:
            previous_errors.append(float(np.max(np.abs(obs[20:34]-last[1]))))
        last = (inf['cycle'], velocity)
        heads.append(obs[51:55])
    if not output_errors:
        raise ValueError('no walk observations')
    head = np.asarray(heads)
    return dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        samples=len(heads), output_max_abs_error=max(output_errors),
        current_velocity_match_fraction=float(np.mean(np.asarray(current_errors)<1e-5)),
        adjacent_cycles=len(previous_errors),
        previous_velocity_match_fraction=float(np.mean(np.asarray(previous_errors)<1e-5))
            if previous_errors else None,
        head_command_min_deg=np.degrees(head.min(0)).tolist(),
        head_command_max_deg=np.degrees(head.max(0)).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True, type=Path)
    parser.add_argument('--captures', required=True, type=Path, nargs='+')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.policy), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    result = dict(policy=str(args.policy),
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        records=[audit(path, session) for path in args.captures],
        live_hardware_accessed=False,
        note='Archived telemetry; adjacent control cycles only. Not proof of fall causality.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
