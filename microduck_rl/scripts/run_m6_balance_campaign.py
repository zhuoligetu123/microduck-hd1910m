#!/usr/bin/env python3
"""One bounded warm-start comparison per host, then simulation; never deploy."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from run_reference_p6_training import run_training, save
from run_m6_transfer_campaign import screen


def replay(root, name, policy, env, options=()):
    out = root/'evaluation'
    out.mkdir(exist_ok=True)
    with (out/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable, str(root/'source/scripts/replay_hd1910.py'),
            '--bam-reference', '--policy', str(policy), '--seconds', '20', '--extended',
            '--seed', '42', '--report', str(out/(name+'.json')), *options],
            env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((out/(name+'.json')).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--variant', choices=('control','balance','balance_lift','lift_focus','lift_robust',
                                             'pitch_robust','pitch_body','head_quiet','head_sole','head_lift_progress','head_lateral','head_omni','head_lower','head_lower_joint','recovery','recovery_support','recovery_all'), required=True)
    parser.add_argument('--phase', choices=('smoke','pilot','evaluate'), required=True)
    parser.add_argument('--envs', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--parent-checkpoint', default='model_299.pt',
                        help='Checkpoint filename under root/parent; paired with parent/policy.onnx')
    args = parser.parse_args()
    if not 5 <= args.iterations <= 600 or not 64 <= args.envs <= 8192:
        parser.error('bounded pilot requires 5..600 iterations, 64..8192 environments')
    root = args.root.resolve(strict=True)
    extra = ['--repair-variant',args.variant,'--warm-start-checkpoint',str(root/'parent'/args.parent_checkpoint),
             '--warm-start-policy',str(root/'parent/policy.onnx')]
    try:
        if args.phase == 'smoke':
            result, _ = run_training(root,64,4,5,'smoke',2026,extra)
            if result['returncode'] or result['iterations_reported'] != 5:
                raise RuntimeError('smoke failed')
            save(root/'status.json',dict(stage='smoke_passed',variant=args.variant))
            return
        smoke = json.loads((root/'runs/smoke/result.json').read_text())
        if smoke['returncode'] or smoke['iterations_reported'] != 5:
            raise RuntimeError('completed smoke required')
        name = 'pilot_'+args.variant
        if args.phase == 'pilot':
            result, env = run_training(root,args.envs,4,args.iterations,name,2026,extra)
            if result['returncode'] or result['iterations_reported'] != args.iterations:
                raise RuntimeError('incomplete training')
        else:
            result = json.loads((root/'runs'/name/'result.json').read_text())
            if result['returncode'] or result['iterations_reported'] != args.iterations:
                raise RuntimeError('completed pilot required before evaluation')
            env = os.environ.copy()
            env.update(PYTHONPATH=str(root/'source/src'), MUJOCO_GL='egl',
                       MICRODUCK_BAM_KP='6', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='4',
                       MICRODUCK_BAM_PROFILE=str(root/'source/src/mjlab_microduck/actuator/radxa_1910_m6.json'))
        if args.variant in ('recovery', 'recovery_support', 'recovery_all'):
            policies = list((root/'runs'/name/'logs').glob('**/*_'+name+'.onnx'))
            if len(policies) != 1:
                raise RuntimeError('Expected exactly one recovery export')
            policy = policies[0]
            with (root/'runs'/name/'parity.log').open('w') as log:
                subprocess.run([sys.executable,str(root/'source/scripts/audit_bounded_policy.py'),
                    '--policy',str(policy),'--checkpoint',str(policy.parent/f'model_{args.iterations-1}.pt'),
                    '--report',str(root/'runs'/name/'parity.json')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        else:
            _, policy = screen(root,name,args.iterations,env)
        parity = json.loads((root/'runs'/name/'parity.json').read_text())
        if not parity['parity_passed']:
            raise RuntimeError('ONNX parity failed')
        reports = {}
        if args.variant in ('recovery', 'recovery_support', 'recovery_all', 'pitch_robust', 'pitch_body', 'head_quiet', 'head_sole', 'head_lift_progress', 'head_lateral', 'head_omni', 'head_lower', 'head_lower_joint'):
            out = root/'evaluation'
            out.mkdir(exist_ok=True)
            suite = 'pitch' if args.variant in ('pitch_robust','pitch_body','head_quiet','head_sole','head_lift_progress','head_lateral','head_omni','head_lower','head_lower_joint') else 'recovery'
            for label, path in [('parent',root/'parent/policy.onnx'),('candidate',policy)]:
                for seed in (42, 7):
                    key = f'{label}_{suite}_{seed}'
                    options = ['--video-dir',str(out/'preview')] if label == 'candidate' and seed == 42 else []
                    with (out/(key+'.log')).open('w') as log:
                        subprocess.run([sys.executable,str(root/'source/scripts/replay_m6_recovery.py'),
                            '--policy',str(path),'--suite',suite,'--seed',str(seed),
                            '--report',str(out/(key+'.json')),*options],env=env,stdout=log,
                            stderr=subprocess.STDOUT,check=True)
                    reports[key] = json.loads((out/(key+'.json')).read_text())
            if args.variant in ('recovery', 'recovery_support', 'recovery_all'):
                save(root/'summary.json',dict(variant=args.variant,policy=str(policy),reports=reports,
                    deployment_ready=False,hardware_tested=False,automatic_extension=False))
                save(root/'status.json',dict(stage='evaluated_review_required'))
                return
        for label, path in [('parent',root/'parent/policy.onnx'),('candidate',policy)]:
            for condition, options in [('nominal',[]),
                ('stress',['--voltage','7.0','--delay-steps','6','--initial-tilt-deg','10']),
                ('voltage_extrapolation',['--voltage','6.4','--voltage-extrapolation','--delay-steps','6']),
                ('transitions',['--transition-test'])]:
                key = label+'_'+condition
                reports[key] = replay(root,key,path,env,options)
        replay(root,'preview',policy,env,['--case','forward','--video',str(root/'evaluation/preview.mp4')])
        rows = reports['candidate_nominal']['cases']
        stable = all(r['completed'] and r['no_fall'] and r['max_tilt_deg'] < 45
                     and r.get('low_trunk_fraction',1) < .01 for r in rows)
        save(root/'summary.json',dict(variant=args.variant,policy=str(policy),nominal_stable=stable,
            deployment_ready=False,hardware_tested=False,automatic_extension=False,
            criteria='stability first; tracking/head centering are diagnostics, not hard gates',
            reports=reports))
        save(root/'status.json',dict(stage='evaluated_review_required',nominal_stable=stable))
    except Exception as exc:
        save(root/'status.json',dict(stage='failed',error=str(exc),automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
