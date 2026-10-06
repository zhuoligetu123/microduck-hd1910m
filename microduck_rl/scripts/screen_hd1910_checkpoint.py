#!/usr/bin/env python3
"""Screen one intermediate walking checkpoint while its training keeps running.

Uses the official normalized exporter. Never stops jobs or promotes hardware.
The seed-42 screen cannot replace multi-seed/native/hardware qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def qualified_case_count(report):
    rows = report.get('cases', [])
    if [row.get('case') for row in rows] != ['stand', 'forward', 'turn', 'backward', 'turn_right']:
        return 0
    return sum(row.get('baseline_check_passed') is True
               and row.get('motion_quality_check_passed') is True for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--wait-seconds', type=int, default=0)
    args = parser.parse_args()
    if args.iteration < 0 or args.wait_seconds < 0:
        parser.error('iteration and wait-seconds must be nonnegative')
    run = args.run.resolve(strict=True)
    manifest = json.loads((run/'manifest.json').read_text())
    if manifest['kind'] != 'velocity' or manifest['recipe'] != 'refine':
        parser.error('only the unchanged Refine walking contract is supported')
    profile = ROOT/'src/mjlab_microduck/actuator/reference_hd1910_profile.json'
    if hashlib.sha256(profile.read_bytes()).hexdigest() != manifest['profile_sha256']:
        raise ValueError('training/profile hash mismatch')
    deadline = time.monotonic() + args.wait_seconds
    while True:
        candidates = [p for p in (run/'train/logs').rglob(f'model_{args.iteration}.pt')
                      if p.parent.name != 'pretrained']
        if len(candidates) == 1:
            checkpoint = candidates[0]
            stat = checkpoint.stat()
            time.sleep(2)
            if checkpoint.stat().st_size == stat.st_size and stat.st_size > 0:
                break
        current = json.loads((run/'manifest.json').read_text())
        if time.monotonic() >= deadline or current['status'] in ('failed','evaluated'):
            raise RuntimeError('checkpoint missing, ambiguous, or incomplete')
        time.sleep(min(20, max(0, deadline-time.monotonic())))
    output = run/'intermediate'/str(args.iteration)
    output.mkdir(parents=True,exist_ok=False)
    snapshot = output/checkpoint.name
    shutil.copy2(checkpoint,snapshot)
    result = dict(status='screening',iteration=args.iteration,seed=42,hardware_opened=False,
                  deployment_ready=False,qualification_complete=False,
                  checkpoint_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),stages=[])
    env = dict(os.environ, MICRODUCK_HD1910_REFERENCE=str(profile), MUJOCO_GL='egl')
    policy = output/'policy.onnx'

    def stage(name, command):
        with (output/(name+'.log')).open('w') as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        result['stages'].append(name)

    try:
        stage('export',[sys.executable,str(ROOT/'scripts/export.py'),
              'Mjlab-Velocity-Flat-MicroDuck-HD1910-Reference-Slew-Refine',
              '--checkpoint-file',str(snapshot),'--num-envs','1','--onnx-file',str(policy)])
        stage('parity',[sys.executable,str(ROOT/'scripts/audit_bounded_policy.py'),
              '--policy',str(policy),'--checkpoint',str(snapshot),'--report',str(output/'parity.json')])
        result['policy_sha256'] = hashlib.sha256(policy.read_bytes()).hexdigest()
        counts = {}
        for engine in ('cpu','warp'):
            for stress in (False,True):
                name = engine + ('_delay6' if stress else '_nominal')
                script = 'replay_hd1910.py' if engine == 'cpu' else 'replay_hd1910_warp.py'
                command = [sys.executable,str(ROOT/'scripts'/script),'--policy',str(policy),
                           '--seconds','20','--extended','--report',str(output/(name+'.json'))]
                if stress:
                    command += ['--delay-steps','6','--voltage','8.4','--initial-tilt-deg','5']
                stage(name,command)
                report = json.loads((output/(name+'.json')).read_text())
                counts[name] = qualified_case_count(report)
        result.update(status='screened',passed_cases=counts,
                      pilot_passed=all(v==5 for v in counts.values()))
    except Exception as exc:
        result.update(status='failed',error=str(exc))
        raise
    finally:
        (output/'screen.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
