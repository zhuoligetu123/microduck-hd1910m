#!/usr/bin/env python3
"""Paired payload sensitivity, preserving explicit failures and no promotion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import mjlab  # Initialize task plugins before importing the actuator package.
from mjlab_microduck.actuator.payload_uncertainty import SCENARIOS, mass_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123])
    parser.add_argument('--seconds', type=float, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root/'src'), MICRODUCK_BAM_KP='6',
               MICRODUCK_BAM_PROFILE=str(root/'src/mjlab_microduck/actuator/radxa_1910_m6.json'),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MUJOCO_GL='egl')
    rows = []
    for seed in args.seeds:
        for scenario in SCENARIOS:
            for condition, extra in (
                ('neutral', []),
                ('head_down_push', ['--head-neck-deg', '20', '--head-pitch-deg', '-20',
                    '--head-command-at-s', '5', '--head-push-n', '.6', '--pitch-push-rad-s', '1.2'])):
                report = args.output/f'{scenario}_{condition}_{seed}.json'
                command = [sys.executable, str(root/'scripts/replay_hd1910.py'),
                    '--bam-reference', '--policy', str(args.policy.resolve()),
                    '--report', str(report), '--extended', '--seconds', str(args.seconds),
                    '--seed', str(seed), '--mass-scenario', scenario,
                    '--imu-age-ms', '20', '--joint-age-steps', '2', '--delay-steps', '6', *extra]
                with report.with_suffix('.log').open('w') as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                data = json.loads(report.read_text())
                rows.extend(dict(scenario=scenario, condition=condition, seed=seed, **case)
                            for case in data['cases'])
                summary = dict(policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                    mass_contract=mass_contract(), cases=len(rows),
                    no_fall=sum(c['no_fall'] and c['completed'] for c in rows),
                    hardware_tested=False, deployment_ready=False, results=rows)
                (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
                print(scenario, condition, seed, sum(c['no_fall'] for c in data['cases']),
                      '/', len(data['cases']), flush=True)


if __name__ == '__main__':
    main()
