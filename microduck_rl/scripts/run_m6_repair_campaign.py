#!/usr/bin/env python3
"""Bounded, parallel B-parent ablations. Never automatically extend training or deploy."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_luwu_p6_training import run_training, save
from run_m6_transfer_campaign import CASES, screen

VARIANTS = ('control', 'reversal', 'yaw')


def assessment(reports):
    if len(reports) != 2 or any([c.get('case') for c in r.get('cases', [])] != CASES for r in reports):
        raise ValueError('Require complete CPU and Warp reports')
    rows = [c for report in reports for c in report['cases']]
    counts = {k: sum(c.get(k) is True for c in rows) for k in (
        'completed', 'no_fall', 'baseline_check_passed',
        'motion_quality_check_passed', 'head_center_check_passed')}
    return dict(counts=counts, simulation_screen_passed=all(n == 10 for n in counts.values()),
                deployment_ready=False, hardware_tested=False)


def worker(root, variant, iterations):
    root = root.resolve(strict=True)
    name = 'repair_' + variant
    parent = root/'parent'
    extra = ['--repair-variant', variant, '--warm-start-checkpoint', str(parent/'model_599.pt'),
             '--warm-start-policy', str(parent/'policy.onnx')]
    try:
        result, _ = run_training(root, 64, 4, 5, 'smoke_' + variant, 2026, extra)
        if result['returncode'] or result['iterations_reported'] != 5:
            raise RuntimeError('Smoke test failed; pilot not started')
        result, env = run_training(root, 2048, 4, iterations, name, 2026, extra)
        if result['returncode'] or result['iterations_reported'] != iterations:
            raise RuntimeError('Incomplete pilot')
        save(root/'status.json', dict(stage='evaluation', name=name))
        _, policy = screen(root, name, iterations, env)
        out = root/'runs'/name
        if not json.loads((out/'parity.json').read_text()).get('parity_passed'):
            raise RuntimeError('Export parity failed')
        reports = [json.loads((out/f'{engine}.json').read_text()) for engine in ('cpu', 'warp')]
        result = assessment(reports)
        result.update(variant=variant, policy=str(policy))
        save(root/'assessment.json', result)
        save(root/'status.json', dict(stage='evaluated_requires_review', **result))
    except Exception as error:
        save(root/'failure.json', dict(error=str(error)))
        save(root/'status.json', dict(stage='failed', error=str(error)))
        raise


def campaign(root, iterations):
    root = root.resolve(strict=True)
    if iterations < 5 or iterations > 300:
        raise ValueError('This experiment is capped at 300 iterations per variant')
    if (root/'plan.json').exists():
        raise ValueError('Use a new campaign directory; never overwrite previous trials')
    for path in ('source', 'installation.json', 'parent/model_599.pt', 'parent/policy.onnx'):
        if not (root/path).exists():
            raise FileNotFoundError(path)
    save(root/'plan.json', dict(variants=VARIANTS, iterations=iterations, envs=2048, seed=2026,
        workers=2, automatic_long_training=False, deployment_ready=False))
    pending = list(VARIANTS)
    active = {}
    finished = {}

    def stop(signum, frame):
        del frame
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while pending or active:
            while pending and len(active) < 2:
                variant = pending.pop(0)
                child = root/variant
                child.mkdir()
                for entry in ('source', 'installation.json', 'parent'):
                    (child/entry).symlink_to(root/entry)
                with (child/'worker.log').open('w') as log:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                        str(child), '--variant', variant, '--iterations', str(iterations)],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[variant] = process
            save(root/'status.json', dict(stage='pilots', active={n:p.pid for n,p in active.items()},
                 pending=pending, finished=finished))
            for name, process in list(active.items()):
                if process.poll() is not None:
                    finished[name] = process.returncode
                    del active[name]
            time.sleep(2)
        results = {}
        for name, code in finished.items():
            results[name] = (json.loads((root/name/'assessment.json').read_text())
                             if code == 0 else dict(error='worker_failed', returncode=code))
        save(root/'summary.json', dict(results=results, automatic_long_training=False,
                                      deployment_ready=False))
        save(root/'status.json', dict(stage='review_required', finished=finished))
    except BaseException:
        # Kill entire worker groups, including training/evaluation grandchildren.
        for process in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in active.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        save(root/'status.json', dict(stage='interrupted', automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args()
    if not 5 <= args.iterations <= 300:
        parser.error('iterations must be in [5, 300]')
    if args.variant:
        worker(args.root, args.variant, args.iterations)
    else:
        campaign(args.root, args.iterations)
