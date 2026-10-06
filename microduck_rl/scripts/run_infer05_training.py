#!/usr/bin/env python3
"""Benchmark the 5090, then run independent M6 locomotion seeds. No hardware I/O."""
import argparse
import json
from pathlib import Path

from run_reference_p6_training import evaluate, run_training, save, select_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--phase', choices=('benchmark', 'train'), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 2026])
    parser.add_argument('--iterations', type=int, default=4000)
    args = parser.parse_args()
    if args.iterations <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('iterations must be positive and seeds must be unique')
    root = args.root.resolve(strict=True)
    recipe = ['--locomotion-refine']
    try:
        if args.phase == 'benchmark':
            smoke, _ = run_training(root, 64, 4, 5, 'smoke', 7, recipe)
            if smoke['returncode']:
                raise RuntimeError('Smoke failed; no long runs started')
            results = []
            for envs, threads in ((8192, 4), (12288, 4), (16384, 4), (12288, 8)):
                result, _ = run_training(root, envs, threads, 25,
                                        f'bench_{envs}_{threads}', 7, recipe)
                results.append(result)
                save(root/'benchmarks.json', results)
            selected = select_candidate(results)
            save(root/'selected.json', selected)
            save(root/'status.json', dict(stage='benchmarked', selected=selected))
            return
        selected = json.loads((root/'selected.json').read_text())
        save(root/'plan.json', dict(seeds=args.seeds, iterations=args.iterations,
             envs=selected['envs'], threads=selected['threads'], recipe='m6_locomotion_refine_v1',
             environment_steps_per_seed=selected['envs']*24*args.iterations,
             hardware_tested=False, deployment_ready=False))
        for seed in args.seeds:
            name = f'p6_locomotion{seed}'
            result, env = run_training(root, selected['envs'], selected['threads'],
                                       args.iterations, name, seed, recipe)
            if result['returncode']:
                raise RuntimeError(f'Training failed: {name}')
            evaluate(root, name, args.iterations, env)
        save(root/'status.json', dict(stage='evaluated_not_hardware_qualified'))
    except Exception as error:
        save(root/'failure.json', dict(phase=args.phase, error=str(error)))
        raise


if __name__ == '__main__':
    main()
