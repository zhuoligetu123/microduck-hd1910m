#!/usr/bin/env python3
"""Bounded, local voltage-only transfer from smoothing/B; no hardware access."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_reference_p6_training import run_training, save
from review_stable_gait import stability_metrics

ROOT = Path(__file__).resolve().parents[1]
RECIPE = 'gait_reference_curriculum_scaled_v21'
BASELINE_SHA = 'fc8b790539e22b175dce9ae41146bd253ca90d00934f64b50d9e0397f8ae2f7f'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(root):
    result = os.environ.copy()
    result.update(PYTHONPATH=str(root/'source/src'), MICRODUCK_BAM_KP='6',
        MICRODUCK_BAM_PROFILE=str(root/'source/src/mjlab_microduck/actuator/radxa_1910_m6.json'),
        MUJOCO_GL='egl', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return result


def checked(command, log, env, timeout=900):
    save(log.with_suffix('.command.json'), command)
    with log.open('w') as stream:
        subprocess.run(command, env=env, cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=timeout)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    for name in ('src', 'scripts'):
        shutil.copytree(ROOT/name, root/'source'/name, ignore=shutil.ignore_patterns('__pycache__'))
    reference_path = ROOT/'config/static_home_20261004/reference.json'
    reference = json.loads(reference_path.read_text())
    save(root/'installation.json', dict(calibration_verified=False,
        joints=[{k:j[k] for k in ('name', 'id', 'direction', 'zero_ticks')} for j in reference['joints']]))
    shutil.copy2(reference_path, root/'static_reference.json')
    parent = ROOT/'training_runs/reference_smoothing_ab_20261004/B/runs'/RECIPE/'logs/rsl_rl/microduck_hd1910_xgobam_p6'
    checkpoints = list(parent.glob('*/model_650.pt'))
    if len(checkpoints) != 1:
        raise ValueError('expected smoothing/B checkpoint 650')
    policy = ROOT/'training_runs/stability_only_final_20261004/selected/primary/policy.onnx'
    if digest(policy) != BASELINE_SHA:
        raise ValueError('baseline changed')
    (root/'parent').mkdir()
    shutil.copy2(checkpoints[0], root/'parent/model_650.pt')
    shutil.copy2(policy, root/'parent/policy.onnx')
    save(root/'plan.json', dict(parent_sha256=digest(checkpoints[0]), policy_sha256=BASELINE_SHA,
        reference_sha256=reference['source_sha256'], recipe=RECIPE,
        arms=dict(A='nominal', B='static_home'), iterations_each=100, envs=4096, seed=2026,
        unchanged=['reward', 'head course', 'M6 parameters', 'HOME', 'joint mapping', 'action semantics'],
        hypothesis='voltage-only transfer improves low-voltage stability without suppressing stepping',
        acceptance='no fall/head contact and bilateral complete swings; no speed or 25mm gate',
        voltage_model_status='telemetry-informed extrapolation, not identified', hardware_tested=False))
    checked([sys.executable, str(root/'source/scripts/audit_bounded_policy.py'),
        '--policy', str(root/'parent/policy.onnx'), '--checkpoint', str(root/'parent/model_650.pt'),
        '--report', str(root/'parent/parity.json')], root/'parent/parity.log', environment(root))


def train(root):
    for name, domain in (('A', 'nominal'), ('B', 'static_home')):
        extra = ['--repair-variant', RECIPE, '--resume-checkpoint', str(root/'parent/model_650.pt'),
            '--head-bias-course', 'frozen', '--action-rate-weight', '-0.1',
            '--agent.algorithm.desired-kl', '.005', '--agent.save-interval', '50',
            '--voltage-domain', domain]
        for suffix, envs, iterations in (('_smoke', 64, 5), ('', 4096, 100)):
            result, env = run_training(root, envs, 4, iterations, name+suffix, 2026, extra)
            if result['returncode'] or result['iterations_reported'] != iterations:
                raise RuntimeError('training incomplete: '+name+suffix)
            out = root/'runs'/(name+suffix)
            log = (out/'train.log').read_text()
            import re
            values = re.findall(r'Episode_Termination/nan_state:\s*([\d.]+)', log)
            if not values or any(float(v) != 0 for v in values):
                raise ValueError('missing or nonzero nan_state metric')
            policies = list(out.glob('logs/rsl_rl/**/*_'+name+suffix+'.onnx'))
            if len(policies) != 1:
                raise ValueError('missing/ambiguous exported policy')
            policy = policies[0]
            checked([sys.executable, str(root/'source/scripts/audit_bounded_policy.py'),
                '--policy', str(policy), '--checkpoint', str(policy.parent/f'model_{650+iterations}.pt'),
                '--report', str(out/'parity.json')], out/'parity.log', env)
            save(out/'candidate.json', dict(policy=str(policy), sha256=digest(policy),
                                           voltage_domain=domain, hardware_tested=False))
    save(root/'status.json', dict(stage='trained_not_evaluated'))


def replay_job(root, name, policy, voltage, stress, seed):
    folder = root/'evaluation'/name
    folder.mkdir(parents=True, exist_ok=True)
    report = folder/f'v{voltage}_stress{int(stress)}_seed{seed}.json'
    args = [sys.executable, str(root/'source/scripts/replay_hd1910.py'), '--bam-reference',
        '--ground-contact', '--policy', str(policy), '--report', str(report), '--extended',
        '--seconds', '20', '--seed', str(seed), '--voltage', str(voltage),
        '--joint-age-steps', '4' if stress else '1', '--imu-age-ms', '20' if stress else '10',
        '--delay-steps', '8' if stress else '4']
    if voltage < 7:
        args += ['--voltage-extrapolation']
    if stress:
        args += ['--head-pitch-deg', '-20', '--head-command-at-s', '5',
                 '--pitch-push-rad-s', '1.2', '--head-push-n', '.6']
    checked(args, report.with_suffix('.log'), environment(root))
    result = json.loads(report.read_text())
    if result['policy_sha256'] != digest(policy):
        raise ValueError('replay hash mismatch')
    return dict(name=name, voltage=voltage, stress=stress, seed=seed,
                report=str(report), cases=result['cases'])


def evaluate(root):
    models = dict(baseline=root/'parent/policy.onnx')
    for name in ('A', 'B'):
        models[name] = Path(json.loads((root/'runs'/name/'candidate.json').read_text())['policy'])
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(replay_job, root, name, policy, voltage, stress, seed)
            for name, policy in models.items() for voltage in (7.4, 6.7, 6.6)
            for stress in (False, True) for seed in (2027, 4099)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            save(root/'evaluation_progress.json', results)
            print(result['name'], result['voltage'], result['stress'], result['seed'],
                  stability_metrics(result['cases']), flush=True)
    comparison = {}
    for name, policy in models.items():
        own = [r for r in results if r['name'] == name]
        comparison[name] = dict(policy=str(policy), sha256=digest(policy),
            metrics=stability_metrics([c for r in own for c in r['cases']]),
            conditions=[dict(voltage=v, stress=s,
                **stability_metrics([c for r in own if r['voltage']==v and r['stress']==s for c in r['cases']]))
                for v in (7.4, 6.7, 6.6) for s in (False, True)])
    save(root/'comparison.json', comparison)
    save(root/'status.json', dict(stage='evaluated', hardware_tested=False, automatic_promotion=False))


def transition_job(root, name, policy):
    folder = root/'continuous'/name
    folder.mkdir(parents=True, exist_ok=False)
    report = folder/'transition.json'
    command = [sys.executable, str(root/'source/scripts/replay_hd1910.py'),
        '--bam-reference', '--ground-contact', '--policy', str(policy), '--report', str(report),
        '--transition-test', '--extended', '--seconds', '20', '--seed', '202611',
        '--voltage', '6.7', '--voltage-extrapolation', '--mass-scenario', 'back_heavy',
        '--joint-age-steps', '4', '--imu-age-ms', '20', '--delay-steps', '8',
        '--head-pitch-deg', '-20', '--head-command-at-s', '5', '--pitch-push-rad-s', '1.2',
        '--head-push-n', '.6', '--video', str(folder/'preview.mp4'), '--trace', str(folder/'trace.csv')]
    checked(command, folder/'replay.log', environment(root), timeout=1200)
    result = json.loads(report.read_text())
    return dict(name=name, sha256=digest(policy), **stability_metrics(result['cases']))


def continuous(root):
    comparison = json.loads((root/'comparison.json').read_text())
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(transition_job, root, name, Path(item['policy']))
                   for name, item in comparison.items()]
        for future in as_completed(futures):
            results.append(future.result())
            print(json.dumps(results[-1]), flush=True)
    save(root/'continuous_comparison.json', results)


def diagnose_job(root, name, changes):
    command = json.loads((root/'continuous/baseline/replay.command.json').read_text())
    for flag in ('--video', '--trace'):
        index = command.index(flag)
        del command[index:index+2]
    folder = root/'diagnosis'/name
    folder.mkdir(parents=True, exist_ok=False)
    changes = dict(changes, **{'--report': str(folder/'replay.json')})
    for flag, value in changes.items():
        command[command.index(flag)+1] = str(value)
    checked(command, folder/'replay.log', environment(root))
    report = json.loads((folder/'replay.json').read_text())
    first_failure = None
    for index, case in enumerate(report['cases']):
        times = [case.get(key) for key in ('first_fall_s', 'first_head_floor_contact_s')]
        times = [t for t in times if t is not None]
        if times:
            first_failure = index*20+min(times)
            break
    return dict(condition=name, changes=changes, first_failure_s=first_failure,
                **stability_metrics(report['cases']))


def diagnose(root):
    # All share the original failed baseline, seed and 180 s command sequence.
    conditions = dict(nominal_mass={'--mass-scenario': 'cad_nominal'},
        nominal_voltage={'--voltage': '7.4'},
        shorter_io={'--joint-age-steps': '1', '--delay-steps': '4'},
        neutral_head={'--head-pitch-deg': '0'})
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(diagnose_job, root, name, changes) for name, changes in conditions.items()]
        results = [f.result() for f in as_completed(futures)]
    save(root/'diagnosis.json', results)
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--phase', choices=('train', 'evaluate', 'continuous', 'diagnose'), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'train':
        prepare(root)
        train(root)
    elif args.phase == 'evaluate':
        evaluate(root)
    elif args.phase == 'continuous':
        continuous(root)
    else:
        diagnose(root)


if __name__ == '__main__':
    main()
