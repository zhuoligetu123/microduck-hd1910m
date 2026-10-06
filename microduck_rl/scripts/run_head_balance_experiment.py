#!/usr/bin/env python3
"""Local M6 head/locomotion ablation with independent CPU MuJoCo evaluation.

Never connects to hardware or overwrites the deployed policy. Training budget
is bounded; survival, actual motion and head tracking are evaluated separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from run_reference_p6_training import run_training, save

SOURCE = Path(__file__).resolve().parents[1]


def training_start(root, variant, resume=False):
    """Stage a verified parent without conflating resume with a new recipe."""
    extra = ['--repair-variant', variant, '--agent.save-interval', '50']
    if not resume:
        return extra + ['--warm-start-checkpoint', str(root/'parent/model.pt'),
                        '--warm-start-policy', str(root/'parent/policy.onnx')], 0
    import torch
    payload = torch.load(root/'parent/model.pt', map_location='cpu', weights_only=False)
    completed = int(payload['iter'])
    if completed < 0:
        raise ValueError('resume requires a completed training iteration')
    checkpoint = root/'parent'/f'model_{completed}.pt'
    shutil.copy2(root/'parent/model.pt', checkpoint)
    # train_hd1910_bam verifies the companion ONNX task/physics/recipe contract.
    return extra + ['--resume-checkpoint', str(checkpoint)], completed + 1


def environment(root):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root/'source/src'), MICRODUCK_BAM_KP='6',
               MICRODUCK_BAM_PROFILE=str(root/'source/src/mjlab_microduck/actuator/radxa_1910_m6.json'),
               MUJOCO_GL='egl', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    return env


def evaluate(root, policy, name, seeds, stress=False, seconds=20):
    out = root/'evaluations'/name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    heads = [('head_force', 20, -20)] if stress else [
        ('neutral', 0, 0), ('look_down', 0, -20), ('neck_forward', 20, 0), ('combined', 20, -20)]
    for seed in seeds:
        for head_name, neck, head in heads:
            for age, delay in (((4, 8),) if stress else ((0, 4), (4, 8))):
                report = out/f'{head_name}_age{age}_seed{seed}.json'
                cmd = [sys.executable, str(root/'source/scripts/replay_hd1910.py'),
                       '--bam-reference', '--policy', str(policy), '--report', str(report),
                       '--extended', '--curve-cases', '--seconds', str(seconds), '--seed', str(seed),
                       '--head-neck-deg', str(neck), '--head-pitch-deg', str(head),
                       '--joint-age-steps', str(age), '--delay-steps', str(delay)]
                if stress:
                    cmd += ['--head-command-at-s', '5', '--pitch-push-rad-s', '1.2', '--head-push-n', '.6']
                with report.with_suffix('.log').open('w') as log:
                    subprocess.run(cmd, env=environment(root), stdout=log, stderr=subprocess.STDOUT, check=True)
                data = json.loads(report.read_text())
                for case in data['cases']:
                    rows.append(dict(head=head_name, age=age, seed=seed, **case))
                print(name, head_name, age, seed, 'survived',
                      sum(c['no_fall'] for c in data['cases']), '/', len(data['cases']), flush=True)
    summary = dict(policy=str(policy), sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                   cases=len(rows), no_fall=sum(r['no_fall'] for r in rows),
                   movement_pass=sum(r['baseline_check_passed'] for r in rows),
                   worst_tilt_deg=max(r['max_tilt_deg'] for r in rows),
                   head_error_mean_abs_deg=sum(sum(abs(x) for x in r['head_mean_error_deg'][:2])/2
                                              for r in rows)/len(rows),
                   failures=[{k:r[k] for k in ('head','age','seed','case','first_fall_s','max_tilt_deg')}
                             for r in rows if not r['no_fall']],
                   deployment_ready=False, hardware_tested=False)
    save(out/'summary.json', summary)
    print(json.dumps(summary), flush=True)


def compare(root):
    """Aggregate paired replay evidence without automatically promoting a policy."""
    comparison = {}
    for directory in sorted((root/'evaluations').iterdir()):
        reports = [json.loads(p.read_text()) for p in sorted(directory.glob('*_seed*.json'))]
        if not reports:
            continue
        hashes = {d['policy_sha256'] for d in reports}
        if len(hashes) != 1:
            raise ValueError(f'mixed policy hashes in {directory}')
        cases = [c for d in reports for c in d['cases']]
        survived = [c for c in cases if c['no_fall'] and c['completed']]
        head, neck = [], []
        for report in reports:
            for case in report['cases']:
                error = case.get('settled_prefall_head_metrics', {}).get('head_mean_error_deg')
                if not case['no_fall'] or error is None:
                    continue
                if report['head_command_offset_deg'][1] < 0:
                    head.append(abs(error[1]))
                if report['head_command_offset_deg'][0] > 0:
                    neck.append(abs(error[0]))
        moving = [c for c in survived if any(c['command'])]
        forward = [c for c in survived if c['command'][0] > 0]
        turning = [c for c in survived if c['command'][2] != 0]
        means = {
            'head_down_abs_dc_error_deg': head, 'neck_forward_abs_dc_error_deg': neck,
            'forward_actual_m_s': [c['mean_body_velocity_after_1s'][0] for c in forward],
            'turn_signed_progress_rad_s': [c['mean_body_velocity_after_1s'][2] *
                (1 if c['command'][2] > 0 else -1) for c in turning],
            'moving_sole_p95_mean_mm': [x*1000 for c in moving for x in c['foot_clearance_p95_m']],
        }
        comparison[directory.name] = dict(policy_sha256=hashes.pop(),
            cases=len(cases), no_fall=sum(c['no_fall'] for c in cases),
            movement_pass=sum(c['baseline_check_passed'] for c in cases),
            max_tilt_deg=max(c['max_tilt_deg'] for c in cases),
            survived_only_means={k: statistics.mean(v) if v else None for k,v in means.items()},
            metric_sample_counts={k: len(v) for k,v in means.items()})
    result = dict(comparison=comparison, hardware_tested=False, deployment_ready=False,
                  note='Means exclude fallen cases; compare survival and sample counts first. No automatic selection.')
    save(root/'metric_comparison.json', result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--phase', choices=('prepare', 'train', 'evaluate', 'stress', 'summarize', 'compare'), required=True)
    p.add_argument('--parent', type=Path)
    p.add_argument('--parent-policy', type=Path, help='Export matching the selected checkpoint, not necessarily the final export')
    p.add_argument('--variant', choices=('gait_head_commands', 'gait_head_balance', 'gait_head_follow_v2', 'gait_head_stride_v2', 'gait_head_force_v3', 'gait_head_force_sole_v3', 'gait_head_dc_v4', 'gait_head_dc_stride_v4', 'gait_head_lift_v5', 'gait_timing_v6', 'gait_forward_balance_v6', 'gait_forward_tail_v7', 'gait_payload_v8', 'gait_sole_support_v9', 'gait_sole_demand_v10', 'gait_bilateral_v11', 'gait_bilateral_mirror_v12', 'gait_bilateral_lift_v13', 'gait_bilateral_stage20_v14', 'gait_cycle_yaw_v15', 'gait_lift_release_v16', 'gait_weak_quality_v17', 'gait_reference_recipe_v18', 'gait_reference_scaled_v19', 'gait_reference_curriculum_v20', 'gait_reference_curriculum_scaled_v21', 'gait_reference_linear_only_v22'))
    p.add_argument('--learning-rate', type=float, default=None)
    p.add_argument('--desired-kl', type=float, default=None)
    p.add_argument('--policy', type=Path)
    p.add_argument('--name', default='baseline')
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--resume', action='store_true',
                   help='Continue the prepared identical recipe; iterations are additional updates')
    p.add_argument('--head-bias-course', choices=('scheduled', 'frozen'),
                   help='Explicit head-bias curriculum ablation; all other recipe terms are retained')
    p.add_argument('--num-envs', type=int, default=2048)
    p.add_argument('--action-rate-weight', type=float,
                   help='Freeze only the action-rate penalty, preserving the other curriculum terms')
    p.add_argument('--action-rate-domain', choices=('applied', 'latent'))
    p.add_argument('--tracking-axes', choices=('coupled', 'separate', 'separate_yaw'))
    p.add_argument('--swing-reference', choices=('ray', 'collision'))
    p.add_argument('--standing-fraction', type=float)
    p.add_argument('--mirror-loss-weight', type=float)
    p.add_argument('--tracking-mean-seconds', type=float)
    p.add_argument('--straight-yaw-std', type=float)
    p.add_argument('--airtime-height-gate', choices=('off', 'gentle', 'on'))
    p.add_argument('--feedback-age-max-steps', type=int, choices=range(9))
    p.add_argument('--bilateral-clearance-weight', type=float)
    p.add_argument('--bilateral-clearance-target', type=float)
    p.add_argument('--resume-exploration-std-cap', type=float)
    p.add_argument('--fixed-exploration-std', type=float)
    p.add_argument('--walking-hip-roll-std', type=float)
    p.add_argument('--airtime-window-shift', type=float)
    p.add_argument('--terrain-course', choices=('flat', 'microblocks', 'microblocks12'))
    p.add_argument('--slew-demand-weight', type=float)
    p.add_argument('--forward-probability', type=float)
    p.add_argument('--walking-flexion-scale', type=float)
    p.add_argument('--seconds', type=float, default=20)
    a = p.parse_args()
    root = a.root.resolve()
    if a.phase == 'prepare':
        root.mkdir(parents=True, exist_ok=False)
        for directory in ('src', 'scripts'):
            shutil.copytree(SOURCE/directory, root/'source'/directory,
                            ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(SOURCE.parent/'radxa/reports/native_zero_stand_rl_20260928/bench_installation.json',
                     root/'installation.json')
        parent = a.parent.resolve(strict=True)
        (root/'parent').mkdir()
        shutil.copy2(parent, root/'parent/model.pt')
        policies = [a.parent_policy.resolve(strict=True)] if a.parent_policy else list(parent.parent.glob('*.onnx'))
        if len(policies) != 1:
            raise ValueError('parent requires one companion exported policy')
        shutil.copy2(policies[0], root/'parent/policy.onnx')
        save(root/'plan.json', dict(parent=str(parent), seed=2026,
             variants=([a.variant] if a.variant else ['gait_head_commands','gait_head_balance']),
             head_range_rad=.35, actor_shape=61, actions=14, hardware_tested=False,
             acceptance='Compare survival, movement and actual head tracking separately; no automatic deployment'))
    elif a.phase == 'train':
        if not a.variant:
            p.error('--variant required')
        plan = json.loads((root/'plan.json').read_text())
        if a.variant not in plan['variants']:
            plan['variants'].append(a.variant)
            save(root/'plan.json', plan)
        extra, start_iteration = training_start(root, a.variant, a.resume)
        if a.head_bias_course is not None:
            extra += ['--head-bias-course', a.head_bias_course]
            plan['head_bias_course'] = a.head_bias_course
            save(root/'plan.json', plan)
        if a.action_rate_weight is not None:
            extra += ['--action-rate-weight', str(a.action_rate_weight)]
            plan['fixed_action_rate_weight'] = a.action_rate_weight
            save(root/'plan.json', plan)
        if a.walking_hip_roll_std is not None:
            extra += ['--walking-hip-roll-std', str(a.walking_hip_roll_std)]
            plan['walking_hip_roll_std'] = a.walking_hip_roll_std
            save(root/'plan.json', plan)
        if a.airtime_window_shift is not None:
            extra += ['--airtime-window-shift', str(a.airtime_window_shift)]
            plan['airtime_window_shift'] = a.airtime_window_shift
            save(root/'plan.json', plan)
        if a.terrain_course is not None:
            extra += ['--terrain-course', a.terrain_course]
            plan['terrain_course'] = a.terrain_course
        if a.slew_demand_weight is not None:
            extra += ['--slew-demand-weight', str(a.slew_demand_weight)]
            plan['slew_demand_weight'] = a.slew_demand_weight
            save(root/'plan.json', plan)
        if a.forward_probability is not None:
            extra += ['--forward-probability', str(a.forward_probability)]
            plan['forward_probability'] = a.forward_probability
        if a.walking_flexion_scale is not None:
            extra += ['--walking-flexion-scale', str(a.walking_flexion_scale)]
            plan['walking_flexion_scale'] = a.walking_flexion_scale
        if a.tracking_axes is not None:
            extra += ['--tracking-axes', a.tracking_axes]
            plan['tracking_axes'] = a.tracking_axes
            save(root/'plan.json', plan)
        if a.swing_reference is not None:
            extra += ['--swing-reference', a.swing_reference]
            plan['swing_height_reference'] = a.swing_reference
            save(root/'plan.json', plan)
        for key, value in (('standing_fraction', a.standing_fraction),
                           ('action_rate_domain', a.action_rate_domain),
                           ('mirror_loss_weight', a.mirror_loss_weight),
                           ('tracking_mean_seconds', a.tracking_mean_seconds),
                           ('straight_yaw_std', a.straight_yaw_std),
                           ('feedback_age_max_steps', a.feedback_age_max_steps),
                           ('bilateral_clearance_weight', a.bilateral_clearance_weight),
                           ('bilateral_clearance_target', a.bilateral_clearance_target),
                           ('resume_exploration_std_cap', a.resume_exploration_std_cap),
                           ('fixed_exploration_std', a.fixed_exploration_std)):
            if value is not None:
                extra += ['--' + key.replace('_', '-'), str(value)]
                plan[key] = value
        save(root/'plan.json', plan)
        if a.airtime_height_gate is not None:
            extra += ['--airtime-height-gate', a.airtime_height_gate]
            plan['airtime_height_gate'] = a.airtime_height_gate
            save(root/'plan.json', plan)
        final_iteration = start_iteration + a.iterations - 1
        if a.learning_rate is not None:
            extra += ['--agent.algorithm.learning-rate', str(a.learning_rate)]
        if a.desired_kl is not None:
            extra += ['--agent.algorithm.desired-kl', str(a.desired_kl)]
        smoke, _ = run_training(root, 64, 4, 5, a.variant+'_smoke', 2026, extra)
        if smoke['returncode'] or smoke['iterations_reported'] != 5:
            raise RuntimeError('smoke failed')
        result, env = run_training(root, a.num_envs, 4, a.iterations, a.variant, 2026, extra)
        if result['returncode'] or result['iterations_reported'] != a.iterations:
            raise RuntimeError('training incomplete')
        out = root/'runs'/a.variant
        policies = list(out.glob('logs/rsl_rl/**/*_'+a.variant+'.onnx'))
        if len(policies) != 1:
            raise RuntimeError('missing/ambiguous export')
        policy = policies[0]
        from mjlab_microduck.tasks.hd1910_bam import restore_joint_snapshot_metadata
        restore_joint_snapshot_metadata(policy)
        subprocess.run([sys.executable, str(root/'source/scripts/audit_bounded_policy.py'),
                        '--policy', str(policy), '--checkpoint', str(policy.parent/f'model_{final_iteration}.pt'),
                        '--report', str(out/'parity.json')], env=env, check=True)
        save(out/'candidate.json', dict(policy=str(policy), deployment_ready=False))
        save(root/'status.json', dict(stage='trained_not_evaluated', name=a.variant,
                                     iterations=a.iterations, start_iteration=start_iteration,
                                     final_iteration=final_iteration, resumed=a.resume,
                                     head_bias_course=a.head_bias_course,
                                     fixed_action_rate_weight=a.action_rate_weight,
                                     action_rate_domain=a.action_rate_domain,
                                     tracking_axes=a.tracking_axes,
                                     swing_height_reference=a.swing_reference,
                                     standing_fraction=a.standing_fraction,
                                     mirror_loss_weight=a.mirror_loss_weight,
                                     tracking_mean_seconds=a.tracking_mean_seconds,
                                     straight_yaw_std=a.straight_yaw_std,
                                     airtime_height_gate=a.airtime_height_gate,
                                     feedback_age_max_steps=a.feedback_age_max_steps,
                                     bilateral_clearance_weight=a.bilateral_clearance_weight,
                                     bilateral_clearance_target=a.bilateral_clearance_target,
                                     resume_exploration_std_cap=a.resume_exploration_std_cap,
                                     fixed_exploration_std=a.fixed_exploration_std,
                                     walking_hip_roll_std=a.walking_hip_roll_std,
                                     airtime_window_shift=a.airtime_window_shift,
                                     terrain_course=a.terrain_course,
                                     slew_demand_weight=a.slew_demand_weight,
                                     forward_probability=a.forward_probability,
                                     walking_flexion_scale=a.walking_flexion_scale,
                                     policy=str(policy)))
    elif a.phase in ('evaluate', 'stress'):
        evaluate(root, a.policy.resolve(strict=True), a.name, a.seeds,
                 stress=a.phase == 'stress', seconds=a.seconds)
    elif a.phase == 'compare':
        compare(root)
    else:
        comparison = {}
        for name in ('baseline', 'commands', 'balance'):
            reports = [json.loads((root/'evaluations'/directory/'summary.json').read_text())
                       for directory in (name, name+'_holdout')]
            comparison[name] = {key: sum(r[key] for r in reports)
                                for key in ('cases', 'no_fall', 'movement_pass')}
        candidate = json.loads((root/'runs/gait_head_balance/candidate.json').read_text())
        policy = Path(candidate['policy'])
        parity = json.loads((root/'runs/gait_head_balance/parity_final.json').read_text())
        if not parity['parity_passed'] or parity['policy_sha256'] != hashlib.sha256(policy.read_bytes()).hexdigest():
            raise ValueError('final policy parity/hash mismatch')
        for filename in ('dynamic_push.json', 'look_down_long.json'):
            data = json.loads((root/'evaluations/balance'/filename).read_text())
            if data['policy_sha256'] != parity['policy_sha256']:
                raise ValueError('evaluation policy mismatch')
            comparison[filename] = dict(cases=len(data['cases']),
                no_fall=sum(c['no_fall'] for c in data['cases']),
                max_tilt_deg=max(c['max_tilt_deg'] for c in data['cases']))
        save(root/'comparison.json', dict(comparison=comparison,
             selected_candidate=str(policy), policy_sha256=parity['policy_sha256'],
             selection='local stability candidate, not automatic deployment',
             source_files_sha256={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (SOURCE/'src/mjlab_microduck/tasks/hd1910_bam.py',
                           SOURCE/'src/mjlab_microduck/tasks/mdp.py',
                           SOURCE/'scripts/replay_hd1910.py')},
             remaining=['head command tracking', 'turning underspeed', 'sole clearance below 25mm'],
             deployment_ready=False, hardware_tested=False))
        print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
