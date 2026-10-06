#!/usr/bin/env python3
"""Screen gait-transfer hypotheses before spending a full M6 training budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_luwu_p6_training import evaluate, run_training, save
from run_local_head_campaign import wait_for_predecessor

CASES = ['stand', 'forward', 'turn', 'backward', 'turn_right']
VARIANTS = {'transfer_only': '--locomotion-refine', 'transfer_regularized': '--transfer-refine'}


def screen_score(reports):
    if len(reports) != 2 or any([c.get('case') for c in r.get('cases', [])] != CASES for r in reports):
        return None
    rows = [c for r in reports for c in r['cases']]
    if not all(c.get('completed') and c.get('no_fall') for c in rows):
        return None
    if not all(r['cases'][1].get('baseline_check_passed') for r in reports):
        return None
    tracking = sum(bool(c.get('baseline_check_passed')) for c in rows)
    if tracking < 6:
        return None
    return (tracking, sum(bool(c.get('motion_quality_check_passed')) for c in rows),
            sum(bool(c.get('head_center_check_passed')) for c in rows))


def screen(root, name, iterations, env):
    out = root/'runs'/name
    policies = list((out/'logs').rglob(f'*_{name}.onnx'))
    if len(policies) != 1:
        raise RuntimeError(f'Expected one export: {name}')
    policy = policies[0]
    scripts = root/'source/scripts'
    commands = [('parity', [str(scripts/'audit_bounded_policy.py'), '--policy', str(policy),
                 '--checkpoint', str(policy.parent/f'model_{iterations-1}.pt')])]
    for engine in ('cpu', 'warp'):
        script = 'replay_hd1910.py' if engine == 'cpu' else 'replay_hd1910_warp.py'
        commands.append((engine, [str(scripts/script), '--bam-reference', '--policy', str(policy),
                         '--seconds', '20', '--extended', '--seed', '42']))
    for stage, command in commands:
        with (out/f'{stage}.log').open('w') as log:
            subprocess.run([sys.executable, *command, '--report', str(out/f'{stage}.json')],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    reports = [json.loads((out/f'{engine}.json').read_text()) for engine in ('cpu', 'warp')]
    return screen_score(reports), policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--phase', choices=('smoke', 'run'), required=True)
    parser.add_argument('--after', type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.phase == 'run' and not args.after:
        parser.error('run requires --after to avoid overlapping long jobs')
    parent = root/'parent/model_1199.pt'
    try:
        if args.phase == 'smoke':
            env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
            with (root/'parent_parity.log').open('w') as log:
                subprocess.run([sys.executable, str(root/'source/scripts/audit_bounded_policy.py'),
                    '--checkpoint', str(parent), '--policy', str(root/'parent/policy.onnx'),
                    '--report', str(root/'parent_parity.json')], env=env,
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            for name, flag in VARIANTS.items():
                result, _ = run_training(root, 64, 4, 5, 'smoke_'+name, 2026,
                                         [flag, '--warm-start-checkpoint', str(parent)])
                if result['returncode']:
                    raise RuntimeError(f'Smoke failed: {name}')
            save(root/'status.json', dict(stage='smoke_passed'))
            return
        for name in VARIANTS:
            result = json.loads((root/'runs'/('smoke_'+name)/'result.json').read_text())
            if result['returncode'] or result['iterations_reported'] != 5:
                raise RuntimeError('A completed smoke is required for both variants')
        candidates = []
        for name, flag in VARIANTS.items():
            run_name = 'pilot_'+name
            result, env = run_training(root, 2048, 4, 600, run_name, 2026,
                                      [flag, '--warm-start-checkpoint', str(parent)])
            if result['returncode']:
                raise RuntimeError(f'Pilot failed: {name}')
            save(root/'status.json', dict(stage='pilot_evaluation', name=name))
            score, policy = screen(root, run_name, 600, env)
            candidates.append(dict(name=name, score=score, policy=str(policy)))
            save(root/'pilot_results.json', dict(candidates=candidates, deployment_ready=False))
        eligible = [c for c in candidates if c['score'] is not None]
        if not eligible:
            save(root/'status.json', dict(stage='paused_no_moving_candidate', deployment_ready=False))
            return
        best = max(eligible, key=lambda c: tuple(c['score']))
        save(root/'selected.json', best)
        wait_for_predecessor(root, args.after.resolve(strict=True), 24*3600)
        policy = Path(best['policy'])
        name = 'long_'+best['name']
        result, env = run_training(root, 8192, 4, 3000, name, 2026,
            [VARIANTS[best['name']], '--warm-start-checkpoint', str(policy.parent/'model_599.pt'),
             '--warm-start-policy', str(policy)])
        if result['returncode']:
            raise RuntimeError('Long transfer training failed')
        evaluate(root, name, 3000, env)
        save(root/'status.json', dict(stage='evaluated_not_hardware_qualified'))
    except Exception as error:
        save(root/'failure.json', dict(error=str(error), phase=args.phase))
        raise


if __name__ == '__main__':
    main()
