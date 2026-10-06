#!/usr/bin/env python3
"""Local complementary experiments. Frozen sources, no hardware deployment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_luwu_p6_training import run_training, select_candidate, save, evaluate


def predecessor_ready(root):
    if (root/'failure.json').exists():
        raise RuntimeError(f'Predecessor failed: {root}/failure.json')
    try:
        state = json.loads((root/'status.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if state.get('stage') == 'failed':
        raise RuntimeError(f'Predecessor failed: {root}')
    return state.get('stage') == 'evaluated_not_hardware_qualified'


def wait_for_predecessor(root, predecessor, timeout_s):
    save(root/'status.json', dict(stage='waiting', predecessor=str(predecessor), pid=os.getpid()))
    deadline = time.monotonic() + timeout_s
    while not predecessor_ready(predecessor):
        if time.monotonic() >= deadline:
            raise TimeoutError('Predecessor did not finish before the queue deadline')
        time.sleep(min(30, max(0, deadline-time.monotonic())))


def head_run(root, selection, iterations, name, pilot):
    save(root/'status.json', dict(stage='head_center', name=name))
    env = dict(os.environ, PYTHONPATH=str(root/'source/src'), MUJOCO_GL='egl',
               OMP_NUM_THREADS=str(selection['threads']), OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS=str(selection['threads']))
    command = [sys.executable, str(root/'source/scripts/iterate_hd1910.py'),
        '--kind', 'velocity', '--head-center', '--checkpoint', str(root/'parent/model_1199.pt'),
        '--output', str(root/'runs'/name), '--seed', '123',
        '--num-envs', str(selection['envs']), '--iterations', str(iterations),
        '--action-rate-cost', '5', '--slew-demand-cost', '20', '--yaw-square-weight', '.25',
        '--motor-delay-min-steps', '5']
    if pilot:
        command.append('--pilot')
    with (root/f'{name}.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--phase', choices=('benchmark', 'pilot', 'train', 'replica'), required=True)
    parser.add_argument('--after', type=Path)
    parser.add_argument('--wait-hours', type=float, default=24.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, default=4000)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.iterations <= 0 or not 0 < args.wait_hours <= 48:
        parser.error('iterations must be positive; wait-hours must be in (0, 48]')
    if args.after and args.phase != 'replica':
        parser.error('--after is only valid for replica')
    predecessor = args.after.resolve(strict=True) if args.after else None
    if predecessor == root:
        parser.error('a campaign cannot wait for itself')
    try:
        if args.phase == 'replica':
            if predecessor:
                wait_for_predecessor(root, predecessor, args.wait_hours*3600)
            selection = json.loads((root/'selected.json').read_text())
            recipe = ['--locomotion-refine']
            smoke, _ = run_training(root, 64, selection['threads'], 5,
                                    f'replica_smoke{args.seed}', args.seed, recipe)
            if smoke['returncode']:
                raise RuntimeError('Replica smoke failed')
            name = f'p6_locomotion{args.seed}'
            result, env = run_training(root, selection['envs'], selection['threads'],
                                      args.iterations, name, args.seed, recipe)
            if result['returncode']:
                raise RuntimeError('Replica training failed')
            evaluate(root, name, args.iterations, env)
            save(root/'status.json', dict(stage='evaluated_not_hardware_qualified'))
            return
        if args.phase == 'benchmark':
            smoke, _ = run_training(root, 64, 4, 5, 'p6_smoke', 123, ['--locomotion-refine'])
            if smoke['returncode']:
                raise RuntimeError('P6 smoke failed')
            results = []
            for envs, threads in ((4096,4), (8192,4), (4096,8)):
                result, _ = run_training(root, envs, threads, 25,
                    f'bench_{envs}_{threads}', 123, ['--locomotion-refine'])
                results.append(result)
                save(root/'benchmarks.json', results)
            save(root/'selected.json', select_candidate(results))
            save(root/'status.json', dict(stage='benchmarked'))
        else:
            selection = json.loads((root/'selected.json').read_text())
            if args.phase == 'pilot':
                head_run(root, selection, 100, 'head_pilot123', True)
                save(root/'status.json', dict(stage='pilot_evaluated'))
            else:
                head_run(root, selection, 1200, 'head_center123', False)
                result, env = run_training(root, selection['envs'], selection['threads'],
                    4000, 'p6_locomotion123', 123, ['--locomotion-refine'])
                if result['returncode']:
                    raise RuntimeError('P6 locomotion training failed')
                evaluate(root, 'p6_locomotion123', 4000, env)
                save(root/'status.json', dict(stage='evaluated_not_hardware_qualified'))
    except Exception as error:
        save(root/'failure.json', dict(phase=args.phase, error=str(error)))
        raise


if __name__ == '__main__':
    main()
