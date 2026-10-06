#!/usr/bin/env python3
"""Sequential training/export/replay inventory, bounded to one GPU process.

Default: three representative 64-env/5-iteration smoke runs. --all includes
all stock terrain/backlash/wheel variants and local step/sway tasks. Increasing
iterations trains candidates, never changes deployment_ready to true.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PILOTS = ['Mjlab-Velocity-Flat-MicroDuck', 'Mjlab-SitStand-Flat-MicroDuck',
          'Mjlab-GroundPick-Flat-MicroDuck']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--all', action='store_true')
    selection.add_argument('--tasks', nargs='+')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--replay-seconds', type=float, default=10)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 5 or args.num_envs < 1:
        parser.error('iterations >=5 and positive num-envs required')
    if not 0 <= args.shard_index < args.shard_count:
        parser.error('expected 0 <= shard-index < shard-count')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = ROOT/'src/mjlab_microduck/actuator/reference_hd1910_profile.json'
    os.environ['MICRODUCK_HD1910_REFERENCE'] = str(profile)
    os.environ['MICRODUCK_HD1910_SUITE'] = '1'
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('OMP_NUM_THREADS','4')
    import mjlab_microduck.tasks
    from mjlab.tasks.registry import list_tasks,load_rl_cfg
    stock = [t for t in list_tasks() if 'MicroDuck' in t and '-HD1910' not in t]
    tasks = stock if args.all else args.tasks or PILOTS
    if set(tasks) - set(stock):
        parser.error('unknown stock task')
    tasks = tasks[args.shard_index::args.shard_count]
    if not tasks:
        parser.error('empty task shard')
    source = hashlib.sha256()
    sources = [ROOT/'pyproject.toml', ROOT/'uv.lock']
    sources += sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'scripts').rglob('*.py'))
    for path in sources:
        source.update(str(path.relative_to(ROOT)).encode())
        source.update(path.read_bytes())
    manifest = dict(stage='smoke' if args.iterations == 5 else 'candidate_training',
                    task_count=len(tasks), iterations=args.iterations, num_envs=args.num_envs,
                    shard_index=args.shard_index, shard_count=args.shard_count,
                    source_sha256=source.hexdigest(),
                    profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                    hardware_opened=False, deployment_ready=False, results=[])
    def save():
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    for index,task in enumerate(tasks):
        name = task.removeprefix('Mjlab-')
        row = dict(task=task,status='training',requires_passive_wheels=any(w in task for w in ('Roller','Swizzle','Spin')))
        manifest['results'].append(row)
        save()
        command = [sys.executable, str(ROOT/'scripts/train_hd1910.py'), '--reference-profile',str(profile),
                   '--task',task,'--env.scene.num-envs',str(args.num_envs),'--agent.max-iterations',str(args.iterations),
                   '--agent.logger','tensorboard','--agent.upload-model','False','--enable-nan-guard','True',
                   '--agent.save-interval',str(max(5,args.iterations//4)), '--agent.run-name',output.name]
        logs = ROOT/'logs/rsl_rl'/load_rl_cfg(task+'-HD1910-Reference').experiment_name
        before = set(logs.glob('*'))
        started = time.monotonic()
        with (output/(name+'.train.log')).open('w') as stream:
            result = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        new = set(logs.glob('*')) - before
        policies = [p for run in new for p in run.glob('*.onnx')]
        row.update(training_exit_code=result.returncode, training_seconds=time.monotonic()-started)
        if result.returncode or len(policies) != 1:
            row['status'] = 'training_failed'
        else:
            policy = policies[0]
            row['policy'] = str(policy.relative_to(ROOT))
            row['policy_sha256'] = hashlib.sha256(policy.read_bytes()).hexdigest()
            with (output/(name+'.replay.log')).open('w') as stream:
                replay = subprocess.run([sys.executable,str(ROOT/'scripts/replay_hd1910_task.py'),
                    '--policy',str(policy),'--seconds',str(args.replay_seconds),
                    '--report',str(output/(name+'.replay.json'))],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            row['status'] = 'numerical_replay_passed' if replay.returncode == 0 else 'replay_failed'
            row['replay_exit_code'] = replay.returncode
        save()
        print(f'{index+1}/{len(tasks)} {task}: {row["status"]}',flush=True)
    manifest['pipeline_passed'] = all(r['status']=='numerical_replay_passed' for r in manifest['results'])
    manifest['task_mastery_evaluated'] = False
    save()
    return 0 if manifest['pipeline_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
