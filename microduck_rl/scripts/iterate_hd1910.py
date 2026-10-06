#!/usr/bin/env python3
"""One isolated repair experiment: smoke -> train -> parity -> multi-seed replay.

Use separate --output paths on each GPU. Never promotes a model to hardware.
No SSH, host installation, hardware I/O, or automatic retry of failed training.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def velocity_stress_cases(seeds):
    """Measure the same fixed-delay disturbances on both deployment engines."""
    for seed in seeds:
        for voltage, delay in ((6.5, 6), (7.4, 3), (8.4, 6)):
            for engine in ('cpu', 'warp'):
                prefix = 'stress' if engine == 'cpu' and seed == 42 else f'stress_{engine}_seed{seed}'
                name = f'{prefix}_v{voltage}_delay{delay}'
                script = 'replay_hd1910.py' if engine == 'cpu' else 'replay_hd1910_warp.py'
                yield name, script, ['--seed', str(seed), '--voltage', str(voltage),
                                     '--delay-steps', str(delay), '--initial-tilt-deg', '5']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('velocity','posture'), required=True)
    parser.add_argument('--posture-balanced', action='store_true')
    parser.add_argument('--posture-height-weight', type=float,
                        help='Posture-only ablation: L1 height cost in reward units per metre')
    parser.add_argument('--posture-progress-weight', type=float, default=0.,
                        help='Signed bidirectional goal progress during posture discovery')
    parser.add_argument('--motor-delay-min-steps', type=int, choices=(3,4,5,6),
                        help='Velocity-only stress curriculum within the existing 15..30 ms envelope')
    parser.add_argument('--yaw-square-weight', type=float, default=0.)
    parser.add_argument('--action-rate-cost', type=float,
                        help='Velocity-only positive action-rate penalty magnitude; default recipe is unchanged')
    parser.add_argument('--slew-demand-cost', type=float, default=0.,
                        help='Velocity-only penalty for range-bounded demand beyond the applied slew target')
    parser.add_argument('--delay-hold-steps', type=int, default=0,
                        help='Hold each sampled motor delay for N physics steps; zero retains per-step jitter')
    parser.add_argument('--yaw-push-rad-s', type=float, default=0.,
                        help='Symmetric training-only yaw velocity disturbances at existing push intervals')
    parser.add_argument('--posture-transition-prob', type=float, default=0.)
    parser.add_argument('--posture-transition-clearance', type=float, default=0.)
    parser.add_argument('--pilot', action='store_true',
                        help='One-seed screen before allocating a full multi-seed qualification')
    parser.add_argument('--head-center', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--iterations', type=int, default=600)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.head_center and args.kind != 'velocity':
        parser.error('head-center requires velocity')
    if args.iterations < 5 or args.num_envs < 1:
        parser.error('positive environments and >=5 iterations required')
    if args.posture_balanced and args.kind != 'posture':
        parser.error('posture-balanced requires kind=posture')
    if args.posture_height_weight is not None and (args.kind != 'posture'
            or not math.isfinite(args.posture_height_weight) or args.posture_height_weight <= 0):
        parser.error('posture-height-weight requires posture and a finite positive value')
    if not math.isfinite(args.posture_progress_weight) or args.posture_progress_weight < 0 or (args.posture_progress_weight and args.kind != 'posture'):
        parser.error('posture-progress-weight requires posture and a finite nonnegative value')
    if args.motor_delay_min_steps is not None and args.kind != 'velocity':
        parser.error('motor-delay-min-steps requires velocity')
    if not math.isfinite(args.yaw_square_weight) or args.yaw_square_weight < 0 or (args.yaw_square_weight and args.kind != 'velocity'):
        parser.error('yaw-square-weight requires velocity and a finite nonnegative value')
    if args.action_rate_cost is not None and (args.kind != 'velocity' or not math.isfinite(args.action_rate_cost) or args.action_rate_cost <= 0):
        parser.error('action-rate-cost requires velocity and a finite positive value')
    if not math.isfinite(args.slew_demand_cost) or args.slew_demand_cost < 0 or (args.slew_demand_cost and args.kind != 'velocity'):
        parser.error('slew-demand-cost requires velocity and a finite nonnegative value')
    if args.delay_hold_steps < 0 or (args.delay_hold_steps and args.kind != 'velocity'):
        parser.error('delay-hold-steps requires velocity and a nonnegative integer')
    if not math.isfinite(args.yaw_push_rad_s) or args.yaw_push_rad_s < 0 or (args.yaw_push_rad_s and args.kind != 'velocity'):
        parser.error('yaw-push-rad-s requires velocity and a finite nonnegative value')
    if not math.isfinite(args.posture_transition_prob) or not 0 <= args.posture_transition_prob <= 1 or (args.posture_transition_prob and args.kind != 'posture'):
        parser.error('posture-transition-prob requires posture and a value in 0..1')
    if not math.isfinite(args.posture_transition_clearance) or not 0 <= args.posture_transition_clearance <= .05 or (args.posture_transition_clearance and not args.posture_transition_prob):
        parser.error('posture-transition-clearance requires transition spawns and 0..0.05 m')
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    profile = ROOT/'src/mjlab_microduck/actuator/reference_hd1910_profile.json'
    source = hashlib.sha256()
    for path in sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'scripts').glob('*.py')):
        source.update(str(path.relative_to(ROOT)).encode())
        source.update(path.read_bytes())
    manifest = dict(kind=args.kind, seed=args.seed, num_envs=args.num_envs, iterations=args.iterations,
                    head_center=args.head_center,
                    recipe='balanced' if args.posture_balanced else 'refine',
                    posture_height_weight=args.posture_height_weight,
                    posture_progress_weight=args.posture_progress_weight,
                    motor_delay_min_steps=args.motor_delay_min_steps,
                    yaw_square_weight=args.yaw_square_weight, posture_transition_prob=args.posture_transition_prob,
                    action_rate_cost=args.action_rate_cost,
                    slew_demand_cost=args.slew_demand_cost,
                    delay_hold_steps=args.delay_hold_steps,yaw_push_rad_s=args.yaw_push_rad_s,
                    posture_transition_clearance=args.posture_transition_clearance,
                    pilot=args.pilot,
                    source_sha256=source.hexdigest(), profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                    status='starting', deployment_ready=False, hardware_opened=False, stages=[])

    def save():
        temp = output/'manifest.tmp'
        temp.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        temp.replace(output/'manifest.json')

    def run(name, command, cwd, accepted=(0,)):
        row = dict(name=name,command=command,status='running')
        manifest['stages'].append(row)
        manifest['status'] = name
        save()
        begin = time.monotonic()
        with (output/(name+'.log')).open('w') as stream:
            code = subprocess.call(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT)
        row.update(exit_code=code,seconds=time.monotonic()-begin,status='done' if code in accepted else 'failed')
        save()
        if code not in accepted:
            raise RuntimeError(f'{name} exited {code}; inspect {output/name}.log')

    base = [sys.executable,str(ROOT/'scripts/train_hd1910.py'), '--reference-profile',str(profile),
            '--slew-targets','--refine']
    if args.head_center:
        base += ['--head-center']
    if args.kind == 'posture':
        base += ['--task','Mjlab-SitStand-Flat-MicroDuck']
    if args.posture_balanced:
        base += ['--posture-balanced']
    if args.posture_height_weight is not None:
        base += ['--env.rewards.posture-height-l1.weight',str(args.posture_height_weight)]
    if args.posture_progress_weight:
        base += ['--env.rewards.hd-posture-progress.weight',str(args.posture_progress_weight),
                 '--env.rewards.rise-bootstrap.weight','0.0']
    if args.motor_delay_min_steps is not None:
        base += ['--env.scene.entities.robot.articulation.actuators.0.delay-min-lag',str(args.motor_delay_min_steps),
                 '--env.scene.entities.robot.articulation.actuators.0.delay-max-lag','6']
    if args.yaw_square_weight:
        base += ['--env.rewards.hd-velocity-error.params.yaw-square-weight', str(args.yaw_square_weight)]
    if args.action_rate_cost is not None:
        base += ['--env.rewards.action-rate-l2.weight',str(-args.action_rate_cost)]
    if args.slew_demand_cost:
        base += ['--env.rewards.hd-slew-demand.weight',str(-args.slew_demand_cost)]
    if args.delay_hold_steps:
        base += ['--env.scene.entities.robot.articulation.actuators.0.delay-update-period',str(args.delay_hold_steps),
                 '--env.scene.entities.robot.articulation.actuators.0.delay-per-env-phase','False']
    if args.yaw_push_rad_s:
        base += ['--env.events.push-robot.params.velocity-range.yaw',
                 str((-args.yaw_push_rad_s,args.yaw_push_rad_s))]
    if args.posture_transition_prob:
        base += ['--env.events.set-ground-state.params.transition-prob', str(args.posture_transition_prob)]
        base += ['--env.events.set-ground-state.params.transition-clearance', str(args.posture_transition_clearance)]
    common = ['--agent.logger','tensorboard','--agent.upload-model','False',
              '--enable-nan-guard','True','--agent.seed',str(args.seed)]
    try:
        os.environ.setdefault('MUJOCO_GL','egl')
        os.environ.setdefault('OMP_NUM_THREADS','4')
        for name, num_envs, iterations in (('smoke',64,5),('train',args.num_envs,args.iterations)):
            cwd = output/name
            cwd.mkdir()
            cmd = base + common + ['--env.scene.num-envs',str(num_envs),'--agent.max-iterations',str(iterations),
                  '--agent.save-interval',str(min(200,iterations)), '--agent.run-name',output.name]
            if name == 'train' and args.checkpoint:
                checkpoint = args.checkpoint.resolve(strict=True)
                parent_profile = checkpoint.parent/'motor_calibration.json'
                if parent_profile.read_bytes() != profile.read_bytes():
                    raise ValueError('warm-start physics profile mismatch')
                parent = json.loads((checkpoint.parent/'hardware_provenance.json').read_text())
                prefix = ('Mjlab-Velocity-Flat-MicroDuck-HD1910-Reference-Slew'
                          if args.kind == 'velocity' else 'Mjlab-SitStand-Flat-MicroDuck-HD1910-Reference-Slew')
                if not parent.get('task_id','').startswith(prefix):
                    raise ValueError('warm-start task/command/action semantics mismatch')
                experiment = ('microduck_hd1910_refine' if args.kind == 'velocity'
                              else 'hd_suite_sitstand_flat_microduck_slew_refine')
                if args.posture_balanced:
                    experiment = 'hd_suite_sitstand_flat_microduck_slew_balanced'
                imported = cwd/'logs/rsl_rl'/experiment/'pretrained'
                imported.mkdir(parents=True)
                shutil.copy2(checkpoint,imported/checkpoint.name)
                manifest['parent_checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                cmd += ['--warm-start','--agent.resume','True','--agent.load-run','pretrained',
                        '--agent.load-checkpoint',checkpoint.name]
            run(name,cmd,cwd)
        policies = list((output/'train/logs').rglob('*.onnx'))
        if len(policies) != 1:
            raise ValueError(f'expected one final ONNX, found {len(policies)}')
        policy = policies[0]
        checkpoint = policy.parent/f'model_{args.iterations-1}.pt'
        manifest.update(policy=str(policy),policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest())
        run('onnx_parity',[sys.executable,str(ROOT/'scripts/audit_bounded_policy.py'),
            '--policy',str(policy),'--checkpoint',str(checkpoint),'--report',str(output/'onnx_parity.json')],ROOT)
        passed = True
        for seed in ((42,) if args.pilot else (42,7,123)):
            engines = ('cpu','warp') if args.kind == 'velocity' else ('posture_cpu','posture')
            for engine in engines:
                name = f'{engine}_seed{seed}'
                script = {'cpu':'replay_hd1910.py','warp':'replay_hd1910_warp.py',
                          'posture':'replay_hd1910_task.py','posture_cpu':'replay_hd1910_task.py'}[engine]
                cmd = [sys.executable,str(ROOT/'scripts'/script),'--policy',str(policy),
                       '--seed',str(seed),'--report',str(output/(name+'.json')),
                       '--seconds','20' if args.kind == 'velocity' else '10']
                cmd += ['--extended'] if args.kind == 'velocity' else ['--posture-cycle']
                if engine == 'posture_cpu':
                    cmd += ['--engine','cpu']
                run(name,cmd,ROOT,accepted=(0,2) if args.kind == 'posture' else (0,))
                result = json.loads((output/(name+'.json')).read_text())
                passed &= result.get('baseline_checks_passed',False) if args.kind == 'velocity' else result.get('task_success',False)
                if args.head_center:
                    cases = result.get('cases', [])
                    passed &= bool(cases) and all(c.get('head_center_check_passed', False) for c in cases)
        if args.kind == 'velocity':
            run('transitions',[sys.executable,str(ROOT/'scripts/replay_hd1910.py'),
                '--policy',str(policy),'--seconds','20','--transition-test','--report',str(output/'transitions.json')],ROOT)
            passed &= json.loads((output/'transitions.json').read_text())['baseline_checks_passed']
        manifest['stress_evaluated'] = False
        if args.kind == 'velocity':
            # A failed nominal case is not a reason to hide stress diagnostics.
            for name, script, options in velocity_stress_cases((42,) if args.pilot else (42,7,123)):
                run(name,[sys.executable,str(ROOT/'scripts'/script),
                    '--policy',str(policy),'--seconds','20','--extended',
                    *options,'--report',str(output/(name+'.json'))],ROOT)
                result = json.loads((output/(name+'.json')).read_text())
                passed &= result['baseline_checks_passed']
                if args.head_center:
                    cases = result.get('cases', [])
                    passed &= bool(cases) and all(c.get('head_center_check_passed', False) for c in cases)
            manifest['stress_evaluated'] = True
        manifest.update(status='evaluated',simulation_screen_passed=bool(passed),
                        qualification_complete=False,
                        note='Pilot/recipe screen only. Full qualification and hardware acceptance are separate gates.')
        save()
        print(json.dumps(manifest,indent=2))
        return 0
    except Exception as exc:
        manifest.update(status='failed',error=str(exc))
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
