#!/usr/bin/env python3
"""Two GPU queues, finite pilots and per-task replay; never auto-deploy candidates."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from run_reference_p6_training import run_training, save

QUEUES = {'gait': [('head_lateral_quiet', 8192, 500), ('step', 8192, 1200)],
          'posture': [('sitstand', 2048, 800)],
          'roll': [('roulade', 4096, 1200)],
          'contact': [('sitstand', 4096, 800), ('recovery_all', 4096, 600), ('roulade', 4096, 1200)]}


def evaluate(root, variant, policy, iterations, env):
    save(root/'status.json', dict(stage='evaluation', name=variant, pid=os.getpid()))
    scripts = root/'source/scripts'
    out = root/'evaluation'/variant
    out.mkdir(parents=True)
    checks = [('parity', ['audit_bounded_policy.py', '--checkpoint',
                         str(policy.parent/f'model_{iterations-1}.pt')])]
    if variant == 'head_lateral_quiet':
        for seed in (42, 7, 123):
            for delay in (4, 10):
                checks.append((f'walk_{seed}_{delay}', ['replay_hd1910.py', '--bam-reference',
                    '--extended', '--seconds', '20', '--seed', str(seed), '--delay-steps', str(delay)]))
            checks.append((f'pitch_{seed}', ['replay_m6_recovery.py', '--suite', 'pitch', '--seed', str(seed)]))
    elif variant == 'sitstand':
        checks.append(('sit_stand', ['replay_hd1910_task.py', '--engine', 'cpu', '--posture-cycle', '--seconds', '10']))
    elif variant == 'recovery_all':
        for seed in (42, 7, 123):
            checks.append((f'recovery_{seed}', ['replay_m6_recovery.py', '--suite', 'recovery', '--side-cases', '--seed', str(seed)]))
    elif variant == 'roulade':
        checks.append(('roll', ['replay_m6_roulade.py']))
    else:
        checks.append(('step', ['replay_m6_step.py']))
    results = {}
    for label, command in checks:
        with (out/f'{label}.log').open('w') as log:
            result = subprocess.run([sys.executable, str(scripts/command[0]), *command[1:],
                '--policy', str(policy), '--report', str(out/f'{label}.json')], env=env,
                stdout=log, stderr=subprocess.STDOUT)
        results[label] = dict(returncode=result.returncode, report=str(out/f'{label}.json'))
        if not (out/f'{label}.json').is_file():
            raise RuntimeError(f'replay crashed: {variant}/{label}')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--queue', choices=QUEUES, required=True)
    parser.add_argument('--resume-checkpoint', type=Path,
                        help='Single-task queue only; finishes its original total iteration budget')
    args = parser.parse_args()
    if args.resume_checkpoint and len(QUEUES[args.queue]) != 1:
        parser.error('resume requires a single-task queue')
    os.environ['PYTHONFAULTHANDLER'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    root = args.root.resolve(strict=True)
    outcomes = []
    for variant, envs, iterations in QUEUES[args.queue]:
        extra = ['--repair-variant', variant]
        remaining = iterations
        if args.resume_checkpoint:
            checkpoint = args.resume_checkpoint.resolve(strict=True)
            match = re.fullmatch(r'model_(\d+)\.pt', checkpoint.name)
            if not match or int(match[1]) + 1 >= iterations:
                parser.error('checkpoint must precede the queue iteration budget')
            remaining -= int(match[1]) + 1
            extra += ['--resume-checkpoint', str(checkpoint)]
        elif variant in ('head_lateral_quiet', 'recovery_all'):
            extra += ['--warm-start-checkpoint', str(root/'parent/model_299.pt'),
                      '--warm-start-policy', str(root/'parent/policy.onnx'),
                      '--agent.algorithm.learning-rate', '.00003', '--agent.algorithm.schedule', 'fixed']
        try:
            result, _ = run_training(root, 64, 4, 5, 'smoke_'+variant, 2026, extra)
            if result['returncode'] or result['iterations_reported'] != 5:
                raise RuntimeError(f"smoke failed: exit={result['returncode']}")
            result, env = run_training(root, envs, 4, remaining, variant, 2026, extra)
            if result['returncode'] or result['iterations_reported'] != remaining:
                raise RuntimeError(f"pilot failed: exit={result['returncode']}")
            policies = list((root/'runs'/variant/'logs').rglob('*_'+variant+'.onnx'))
            if len(policies) != 1:
                raise RuntimeError('expected one export')
            checks = evaluate(root, variant, policies[0], iterations, env)
            outcomes.append(dict(variant=variant, policy=str(policies[0]), checks=checks,
                                 stage='review_required', deployment_ready=False))
        except Exception as exc:
            outcomes.append(dict(variant=variant, stage='failed', error=str(exc)))
        save(root/'summary.json', dict(queue=args.queue, outcomes=outcomes, hardware_tested=False))
    failed = any(row['stage'] == 'failed' for row in outcomes)
    save(root/'status.json', dict(stage='failed' if failed else 'completed_review_required',
                                  queue=args.queue, outcomes=outcomes))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
