#!/usr/bin/env python3
"""Bounded next experiments, then parity and independently seeded CPU replays."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_luwu_p6_training import run_training, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--variant', choices=('head_balance', 'delay_robust', 'pitch_retention', 'pitch_retention_delay'), required=True)
    parser.add_argument('--envs', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=300)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    extra = ['--repair-variant', args.variant, '--warm-start-checkpoint',
             str(root/'parent/model_199.pt'), '--warm-start-policy', str(root/'parent/policy.onnx')]
    if args.variant.startswith('pitch_retention'):
        extra += ['--agent.algorithm.learning-rate', '0.00003',
                  '--agent.algorithm.schedule', 'fixed']
    try:
        smoke, _ = run_training(root, 64, 4, 5, 'smoke', 2026, extra)
        if smoke['returncode']:
            raise RuntimeError('smoke failed')
        name = 'pilot_' + args.variant
        result, env = run_training(root, args.envs, 4, args.iterations, name, 2026, extra)
        if result['returncode']:
            raise RuntimeError('training failed')
        policies = list((root/'runs'/name/'logs').rglob('*_'+name+'.onnx'))
        if len(policies) != 1:
            raise RuntimeError('expected one ONNX export')
        policy = policies[0]
        out = root/'evaluation'
        out.mkdir(exist_ok=True)
        checks = [('parity', [str(root/'source/scripts/audit_bounded_policy.py'),
                  '--policy',str(policy),'--checkpoint',str(policy.parent/f'model_{args.iterations-1}.pt')])]
        for label, candidate in [('parent',root/'parent/policy.onnx'),('candidate',policy)]:
            for seed in (42, 7, 123):
                for delay in (4, 10):
                    checks.append((f'{label}_seed{seed}_delay{delay}', [str(root/'source/scripts/replay_hd1910.py'),
                        '--bam-reference','--policy',str(candidate),'--extended','--seconds','20',
                        '--seed',str(seed),'--delay-steps',str(delay)]))
                checks.append((f'{label}_pitch_seed{seed}', [str(root/'source/scripts/replay_m6_recovery.py'),
                    '--policy',str(candidate),'--suite','pitch','--seed',str(seed)]))
        save(root/'status.json',dict(stage='evaluation',variant=args.variant))
        reports = {}
        for label, command in checks:
            with (out/(label+'.log')).open('w') as log:
                subprocess.run([sys.executable,*command,'--report',str(out/(label+'.json'))],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            reports[label] = json.loads((out/(label+'.json')).read_text())
        save(root/'summary.json',dict(variant=args.variant,policy=str(policy),reports=reports,
            deployment_ready=False,hardware_tested=False))
        save(root/'status.json',dict(stage='evaluated_review_required',variant=args.variant))
    except Exception as exc:
        save(root/'status.json',dict(stage='failed',error=str(exc),automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
