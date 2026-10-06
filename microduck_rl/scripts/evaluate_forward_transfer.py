#!/usr/bin/env python3
"""Paired CPU MuJoCo evaluations; never accesses a robot or promotes a policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 123])
    p.add_argument('--seconds', type=float, default=20.)
    p.add_argument('--ground-contact', action='store_true',
                   help='Use body/head ground contacts for both arms of a paired evaluation')
    p.add_argument('--legacy-velocity-lag', action='store_true',
                   help='Diagnostic only: reproduce the old extra velocity-only cycle')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    policy = a.policy.resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root/'src'), MICRODUCK_BAM_KP='6',
        MICRODUCK_BAM_PROFILE=str(root/'src/mjlab_microduck/actuator/radxa_1910_m6.json'),
        MUJOCO_GL='egl', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    rows = []
    for seed in a.seeds:
        for name, imu, age, delay, extra in (
            ('ideal_imu', 0, 1, 4, []),
            ('measured_envelope', 10, 1, 4, []),
            ('timing_tail', 20, 2, 8, ['--command-loss-probability', '.02']),
            ('head_push_holdout', 20, 4, 8, ['--head-neck-deg','20',
                '--head-pitch-deg','-20', '--head-command-at-s','5',
                '--pitch-push-rad-s','1.2', '--head-push-n','.6'])):
            report = a.output/f'{name}_{seed}.json'
            command = [sys.executable, str(root/'scripts/replay_hd1910.py'),
                '--bam-reference', '--policy', str(policy), '--report', str(report),
                '--extended', '--seconds', str(a.seconds), '--seed', str(seed),
                '--imu-age-ms', str(imu), '--joint-age-steps', str(age),
                '--delay-steps', str(delay), *extra]
            if a.legacy_velocity_lag:
                command.append('--legacy-velocity-lag')
            if a.ground_contact:
                command.append('--ground-contact')
            with report.with_suffix('.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            data = json.loads(report.read_text())
            for case in data['cases']:
                rows.append(dict(condition=name, seed=seed, **case))
            print(name, seed, sum(c['no_fall'] for c in data['cases']), '/', len(data['cases']), flush=True)
    summary = {'policy':str(policy), 'sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),
        'cases':len(rows), 'no_fall':sum(r['no_fall'] for r in rows),
        'hardware_tested':False, 'deployment_ready':False,
        'legacy_velocity_lag':a.legacy_velocity_lag,
        'ground_contact_model':a.ground_contact,
        'timing_note':'10ms IMU, 20ms joint age are envelope scenarios, not reconstructed physical sampling timestamps',
        'results':rows}
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
