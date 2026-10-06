#!/usr/bin/env python3
"""Re-evaluate collected M6 checkpoints in local CPU MuJoCo, never deploy them."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


TASKS = [('remote55', 'head_lateral_quiet', 499), ('remote55', 'step', 1199),
         ('remote51_sit', 'sitstand', 799), ('remote51_recovery', 'recovery_all', 599),
         ('remote51_roll', 'roulade', 1199)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--videos', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    scripts = Path(__file__).resolve().parent
    output = root/'local_eval'
    output.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(MUJOCO_GL='egl', MICRODUCK_BAM_KP='6', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2',
               PYTHONPATH=str(scripts.parent/'src'))
    manifest = dict(hardware_tested=False, deployment_ready=False, artifacts={},
                    evaluator_sha256={p.name: digest(p) for p in scripts.glob('replay*.py')},
                    checks={})
    policies = {}
    for folder, task, iteration in TASKS:
        archive = root/folder
        exports = list((archive/'runs'/task/'logs').rglob('*_'+task+'.onnx'))
        if len(exports) != 1:
            raise ValueError(f'{task}: expected one final export, found {len(exports)}')
        policy = exports[0]
        checkpoint = policy.parent/f'model_{iteration}.pt'
        profile = policy.parent/'motor_calibration.json'
        parity = json.loads((archive/'evaluation'/task/'parity.json').read_text())
        if digest(policy) != parity['policy_sha256']:
            raise ValueError(f'{task}: transfer digest differs from remote report')
        manifest['artifacts'][task] = dict(campaign=folder, iteration=iteration,
            files={str(p.relative_to(root)): digest(p) for p in (policy, checkpoint, profile)})
        policies[task] = policy
        env['MICRODUCK_BAM_PROFILE'] = str(profile)
        checks = [('parity', ['audit_bounded_policy.py', '--checkpoint', str(checkpoint)])]
        if task == 'head_lateral_quiet':
            for seed in (42, 7, 123):
                for delay in (4, 10):
                    checks.append((f'walk_{seed}_{delay}', ['replay_hd1910.py', '--bam-reference',
                        '--extended', '--seconds', '20', '--seed', str(seed), '--delay-steps', str(delay)]))
                checks.append((f'pitch_{seed}', ['replay_m6_recovery.py', '--suite', 'pitch', '--seed', str(seed)]))
        elif task == 'step':
            checks.append(('step', ['replay_m6_step.py']))
        elif task == 'sitstand':
            command = ['replay_hd1910_task.py', '--engine', 'cpu', '--posture-cycle', '--seconds', '10',
                       '--trace', str(output/'sitstand_trace.csv')]
            if args.videos:
                command += ['--video', str(output/'sitstand.mp4')]
            checks.append(('sit_stand', command))
        elif task == 'recovery_all':
            for seed in (42, 7, 123):
                checks.append((f'recovery_{seed}', ['replay_m6_recovery.py', '--suite', 'recovery',
                                                  '--side-cases', '--seed', str(seed)]))
        else:
            checks.append(('roll', ['replay_m6_roulade.py']))
        for label, arguments in checks:
            key = task + '_' + label
            report = output/(key+'.json')
            command = [sys.executable, str(scripts/arguments[0]), *arguments[1:],
                       '--policy', str(policy), '--report', str(report)]
            # Never accept a stale report after a failed rerun.
            report.unlink(missing_ok=True)
            with (output/(key+'.log')).open('w') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if not report.is_file() or result.returncode not in (0, 2):
                raise RuntimeError(f'{key}: replay failed, exit={result.returncode}; see log')
            data = json.loads(report.read_text())
            if data.get('policy_sha256') != digest(policy):
                raise ValueError(f'{key}: report policy mismatch')
            if label == 'parity' and not data.get('parity_passed'):
                raise ValueError(f'{key}: export parity failed')
            manifest['checks'][key] = dict(returncode=result.returncode,
                command=command, report=report.name, report_sha256=digest(report))
            save(output/'manifest.json', manifest)
            print(f'{key}: exit={result.returncode}, report={report.name}', flush=True)
    if args.videos:
        env['MICRODUCK_BAM_PROFILE'] = str(policies['head_lateral_quiet'].parent/'motor_calibration.json')
        command = [sys.executable, str(scripts/'render_m6_multitask_preview.py'),
                   '--walk', str(policies['head_lateral_quiet']),
                   '--recovery', str(policies['recovery_all']),
                   '--segment-seconds', '12', '--output', str(output/'walk_recovery_preview.mp4')]
        with (output/'preview.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest['preview_command'] = command
    manifest['evaluation_completed'] = True
    save(output/'manifest.json', manifest)


if __name__ == '__main__':
    main()
