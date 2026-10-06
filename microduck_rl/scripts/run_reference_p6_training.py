#!/usr/bin/env python3
"""Benchmark and train a frozen P6 campaign. Simulation only; no hardware I/O."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def select_candidate(results):
    valid = [r for r in results if r['returncode'] == 0
             and r['median_steps_s'] > 0 and r['gpu_samples'] > 0
             and r['minimum_free_mib'] >= 600]
    if not valid:
        raise RuntimeError('No successful benchmark with 600 MiB GPU headroom')
    return max(valid, key=lambda r: r['median_steps_s'])


def run_training(root, envs, threads, iterations, name, seed, extra_args=()):
    out = root / 'runs' / name
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / 'source/src'), MUJOCO_GL='egl',
               MICRODUCK_BAM_KP='6', OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS=str(threads),
               MICRODUCK_BAM_PROFILE=str(root / 'source/src/mjlab_microduck/actuator/radxa_1910_m6.json'))
    command = [sys.executable, str(root / 'source/scripts/train_hd1910_bam.py'),
               '--installation', str(root / 'installation.json'),
               '--env.scene.num-envs', str(envs), '--agent.max-iterations', str(iterations),
               '--agent.seed', str(seed), '--agent.run-name', name,
               '--agent.logger', 'tensorboard', '--agent.upload-model', 'False']
    command += list(extra_args)
    if env.get('MICRODUCK_TRAIN_GDB') == '1':
        command = ['gdb', '-batch', '-return-child-result',
                   '-ex', 'set pagination off', '-ex', 'run',
                   '-ex', 'python if gdb.parse_and_eval("$_isvoid($_exitcode)"): '
                          'gdb.execute("thread apply all bt 12"); gdb.execute("x/12i $pc")',
                   '--args', *command]
    save(out / 'command.json', dict(command=command, environment={k:env[k] for k in
         ('PYTHONPATH', 'MICRODUCK_BAM_KP', 'MICRODUCK_BAM_PROFILE', 'OMP_NUM_THREADS',
          'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
          cpu_affinity=sorted(os.sched_getaffinity(0))))
    save(root / 'status.json', dict(stage='training', name=name, pid=os.getpid()))
    samples = []
    start = time.monotonic()
    with (out / 'train.log').open('w') as log, (out / 'gpu.csv').open('w') as metrics:
        metrics.write('elapsed_s,used_mib,total_mib,utilization_percent\n')
        process = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
        save(out / 'process.json', dict(pid=process.pid, start_unix=time.time()))
        while process.poll() is None:
            try:
                raw = subprocess.check_output(['nvidia-smi', '-i', '0',
                    '--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True, timeout=5)
                used, total, util = [int(x.strip()) for x in raw.strip().split(',')]
                samples.append((used, total, util))
                metrics.write(f'{time.monotonic()-start:.2f},{used},{total},{util}\n')
                metrics.flush()
            except (subprocess.SubprocessError, ValueError):
                pass
            time.sleep(2)
        code = process.wait()
    speeds = [int(x) for x in re.findall(r'Steps per second:\s+(\d+)', (out/'train.log').read_text())]
    steady = speeds[5:]
    result = dict(name=name, envs=envs, threads=threads, returncode=code,
                  elapsed_s=time.monotonic()-start, iterations_reported=len(speeds),
                  median_steps_s=statistics.median(steady) if steady else 0,
                  gpu_samples=len(samples),
                  minimum_free_mib=min((t-u for u,t,_ in samples), default=0),
                  peak_used_mib=max((u for u,_,_ in samples), default=0))
    save(out/'result.json', result)
    print(json.dumps(result), flush=True)
    if code:
        print((out/'train.log').read_text()[-4000:], flush=True)
    return result, env


def evaluate(root, name, iterations, env):
    out = root/'runs'/name
    save(root/'status.json', dict(stage='evaluation', name=name, pid=os.getpid()))
    policies = list((out/'logs/rsl_rl/microduck_hd1910_xgobam_p6').glob(f'**/*_{name}.onnx'))
    if len(policies) != 1:
        raise RuntimeError(f'Expected one policy for {name}, found {len(policies)}')
    policy = policies[0]
    scripts = root/'source/scripts'
    with (out/'parity.log').open('w') as log:
        subprocess.run([sys.executable, str(scripts/'audit_bounded_policy.py'),
            '--policy', str(policy), '--checkpoint', str(policy.parent/f'model_{iterations-1}.pt'),
            '--report', str(out/'parity.json')], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for seed in (42, 7, 123):
        for engine in ('cpu', 'warp'):
            script = 'replay_hd1910.py' if engine == 'cpu' else 'replay_hd1910_warp.py'
            for condition, voltage, delay, tilt in (('nominal', 7.4, 4, 0),
                    ('low_long', 7.4, 6, 5), ('high_short', 8.0, 3, 5)):
                report = out/f'{engine}_{condition}_{seed}.json'
                with report.with_suffix('.log').open('w') as log:
                    result = subprocess.run([sys.executable, str(scripts/script),
                        '--bam-reference', '--policy', str(policy), '--extended', '--seconds', '20',
                        '--seed', str(seed), '--voltage', str(voltage), '--delay-steps', str(delay),
                        '--initial-tilt-deg', str(tilt), '--report', str(report)],
                        env=env, stdout=log, stderr=subprocess.STDOUT)
                report.with_suffix('.exit').write_text(str(result.returncode)+'\n')
                if not report.is_file():
                    raise RuntimeError(f'Evaluation crashed without result: {report}')
                json.loads(report.read_text())
    (out/'stage.txt').write_text('evaluated_not_hardware_qualified\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--phase', choices=('benchmark', 'train'), required=True)
    parser.add_argument('--iterations', type=int, default=6000)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.phase == 'benchmark':
        smoke, _ = run_training(root, 64, 4, 5, 'smoke', 7)
        if smoke['returncode'] != 0:
            raise RuntimeError('Smoke failed')
        results = []
        for envs, threads in ((1024, 4), (2048, 4), (3072, 4), (4096, 4), (2048, 8)):
            result, _ = run_training(root, envs, threads, 25, f'bench_{envs}_{threads}', 7)
            results.append(result)
            save(root/'benchmarks.json', results)
        selected = select_candidate(results)
        save(root/'selected.json', selected)
        save(root/'status.json', dict(stage='benchmarked', selected=selected))
    else:
        selected = json.loads((root/'selected.json').read_text())
        for seed in (7, 42):
            name = f'p6_seed{seed}'
            result, env = run_training(root, selected['envs'], selected['threads'],
                                       args.iterations, name, seed)
            if result['returncode'] != 0:
                raise RuntimeError(f'Training failed: {name}')
            evaluate(root, name, args.iterations, env)
        save(root/'status.json', dict(stage='evaluated_not_hardware_qualified'))


if __name__ == '__main__':
    main()
