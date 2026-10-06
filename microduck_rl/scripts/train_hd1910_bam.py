#!/usr/bin/env python3
"""Train the external M6 reference without replacing the default HD1910 model.

First run: --env.scene.num-envs 64 --agent.max-iterations 5.
No hardware I/O or firmware parameter writes. All policies remain unqualified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import mjlab
from mjlab_microduck.tasks.hd1910_bam import register_task, TASK_ID, SITSTAND_TASK_ID, ROULADE_TASK_ID, STEP_TASK_ID

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--warm-start-checkpoint', type=Path)
    parser.add_argument('--resume-checkpoint', type=Path,
                        help='Continue the identical task, retaining optimizer and curriculum')
    parser.add_argument('--warm-start-policy', type=Path,
                        help='Matching exported M6 parent policy; validate physics/action contract')
    parser.add_argument('--motion-refine', action='store_true')
    parser.add_argument('--locomotion-refine', action='store_true',
                        help='Use the established low-speed tracking reward with M6 physics')
    parser.add_argument('--transfer-refine', action='store_true',
                        help='Gradual smoothing and DC head/yaw repair for gait transfer')
    parser.add_argument('--repair-variant', choices=('head_lateral_quiet', 'gait_delivery', 'gait_yaw_only', 'gait_yaw_hold', 'gait_yaw_hold_head', 'gait_joint_age', 'gait_age_only', 'gait_age_mixed', 'gait_joint_age_stress', 'gait_coherent_age', 'gait_head_limit', 'step', 'step_balanced', 'step_right_lift', 'step_anchored', 'step_mild_right', 'step_hold_robust', 'step_delay_curriculum', 'step_joint_age', 'step_age_drift', 'step_lift35_age', 'step_lift35', 'step_coherent_age', 'pitch_retention', 'pitch_retention_delay', 'head_balance', 'delay_robust', 'control', 'reversal', 'yaw', 'balance', 'balance_lift',
                                                   'lift_focus', 'lift_robust', 'pitch_robust', 'pitch_body', 'head_quiet', 'head_sole', 'head_lift_progress', 'head_lateral', 'head_omni', 'head_lower', 'head_lower_joint', 'recovery', 'recovery_support', 'recovery_all', 'sitstand', 'roulade', 'gait_head_commands', 'gait_head_balance', 'gait_head_follow_v2', 'gait_head_stride_v2', 'gait_head_force_v3', 'gait_head_force_sole_v3', 'gait_head_dc_v4', 'gait_head_dc_stride_v4', 'gait_head_lift_v5', 'gait_timing_v6', 'gait_forward_balance_v6', 'gait_forward_tail_v7', 'gait_payload_v8', 'gait_sole_support_v9', 'gait_sole_demand_v10', 'gait_bilateral_v11', 'gait_bilateral_mirror_v12', 'gait_bilateral_lift_v13', 'gait_bilateral_stage20_v14', 'gait_cycle_yaw_v15', 'gait_lift_release_v16', 'gait_weak_quality_v17', 'gait_reference_recipe_v18', 'gait_reference_scaled_v19', 'gait_reference_curriculum_v20', 'gait_reference_curriculum_scaled_v21', 'gait_reference_linear_only_v22'),
                        help='Fixed-budget M6 B-parent ablations; no runtime changes')
    parser.add_argument('--installation', type=Path,
                        help='Read-only native joint-zero snapshot; never writes hardware')
    parser.add_argument('--voltage-domain', choices=('nominal', 'static_home'),
                        help='Explicit voltage-only transfer; static_home is unvalidated M6 extrapolation')
    parser.add_argument('--head-bias-course', choices=('scheduled', 'frozen'),
                        help='Explicit single-term curriculum ablation, recorded in the exported policy')
    parser.add_argument('--action-rate-weight', type=float,
                        help='Freeze only the action-rate penalty for a paired curriculum ablation')
    parser.add_argument('--action-rate-domain', choices=('applied', 'latent'))
    parser.add_argument('--tracking-axes', choices=('coupled', 'separate', 'separate_yaw'),
                        help='Separate command tracking tolerance from upstream stability tolerance')
    parser.add_argument('--swing-reference', choices=('ray', 'collision'),
                        help='Landing swing-height metric; reward weight and target remain unchanged')
    parser.add_argument('--standing-fraction', type=float)
    parser.add_argument('--mirror-loss-weight', type=float)
    parser.add_argument('--tracking-mean-seconds', type=float)
    parser.add_argument('--straight-yaw-std', type=float,
                        help='Straight-only yaw EMA reward; preserve turn/idle tracking')
    parser.add_argument('--airtime-height-gate', choices=('off', 'gentle', 'on'))
    parser.add_argument('--feedback-age-max-steps', type=int, choices=range(9))
    parser.add_argument('--bilateral-clearance-weight', type=float)
    parser.add_argument('--bilateral-clearance-target', type=float)
    parser.add_argument('--clearance-course-mm', type=float, choices=(12., 15., 20., 25.),
                        help='Unify all active height rewards on lowest collision-sole clearance')
    parser.add_argument('--resume-exploration-std-cap', type=float)
    parser.add_argument('--fixed-exploration-std', type=float,
                        help='Keep training noise fixed; deterministic inference is unchanged')
    parser.add_argument('--walking-hip-roll-std', type=float)
    parser.add_argument('--airtime-window-shift', type=float)
    parser.add_argument('--terrain-course', choices=('flat', 'microblocks', 'microblocks12'))
    parser.add_argument('--slew-demand-weight', type=float)
    parser.add_argument('--forward-probability', type=float)
    parser.add_argument('--walking-flexion-scale', type=float)
    args, rest = parser.parse_known_args()
    if args.resume_exploration_std_cap is not None and not args.resume_checkpoint:
        parser.error('--resume-exploration-std-cap requires --resume-checkpoint')
    if args.fixed_exploration_std is not None and args.resume_exploration_std_cap is not None:
        parser.error('choose fixed exploration or one-time exploration cap')
    if args.resume_checkpoint and (args.warm_start_checkpoint or args.warm_start_policy):
        parser.error('resume and warm start are mutually exclusive')
    if sum((args.motion_refine, args.locomotion_refine, args.transfer_refine, bool(args.repair_variant))) > 1:
        parser.error('choose one refinement recipe')
    if args.warm_start_policy and not args.warm_start_checkpoint:
        parser.error('--warm-start-policy requires --warm-start-checkpoint')
    if args.repair_variant and args.repair_variant not in ('step', 'sitstand', 'roulade') and not (args.warm_start_policy or args.resume_checkpoint):
        parser.error('--repair-variant requires a verified M6 parent policy and checkpoint')
    if args.warm_start_checkpoint:
        checkpoint = args.warm_start_checkpoint.resolve(strict=True)
        if args.warm_start_policy:
            import onnx
            from mjlab_microduck.actuator.cpu_hd1910_bam import PROFILE_PATH, KP_FW
            provenance = {x.key:x.value for x in onnx.load(args.warm_start_policy).metadata_props}
            required = dict(task_id=STEP_TASK_ID if args.repair_variant in ('step_balanced', 'step_right_lift', 'step_anchored', 'step_mild_right', 'step_hold_robust', 'step_delay_curriculum', 'step_joint_age', 'step_age_drift', 'step_lift35_age', 'step_lift35', 'step_coherent_age') else TASK_ID,
                actuator_backend='hd1910_bam_m6', kp_fw=str(KP_FW),
                calibration_sha256=hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
                action_semantics='bounded_slew_home_delta_v2',
                previous_action_semantics='bounded_slew_home_delta_v2')
            if any(provenance.get(k) != v for k,v in required.items()):
                parser.error('M6 parent policy physics/action contract mismatch')
        else:
            provenance = json.loads((checkpoint.parent / 'hardware_provenance.json').read_text())
            if provenance.get('task_id') != 'Mjlab-Velocity-Flat-MicroDuck-HD1910-Reference-Slew-Refine':
                parser.error('warm start requires the existing velocity v2 reference contract')
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        from mjlab_microduck.actuator.cpu_hd1910_bam import KP_FW
        experiment = 'microduck_hd1910_xgobam' + ('_p6' if KP_FW == 6 else '')
        target = Path('logs/rsl_rl') / experiment / ('parent_' + digest[:12])
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, target / checkpoint.name)
        if args.warm_start_policy:
            subprocess.run([sys.executable, str(Path(__file__).with_name('audit_bounded_policy.py')),
                '--checkpoint', str(checkpoint), '--policy', str(args.warm_start_policy.resolve()),
                '--report', str(target/'parent_parity.json')], check=True)
        (target / 'warm_start.json').write_text(json.dumps(dict(
            source=str(checkpoint), sha256=digest, source_provenance=provenance,
            physics_changed=not bool(args.warm_start_policy), hardware_tested=False), indent=2))
        os.environ['MICRODUCK_HD1910_WARM_START'] = '1'
        rest += ['--agent.resume', 'True', '--agent.load-run', target.name,
                 '--agent.load-checkpoint', checkpoint.name]
    else:
        os.environ.pop('MICRODUCK_HD1910_WARM_START', None)
    register_task(args.installation, motion_refine=args.motion_refine,
                  locomotion_refine=args.locomotion_refine, transfer_refine=args.transfer_refine,
                  repair_variant=args.repair_variant, head_bias_course=args.head_bias_course,
                  action_rate_weight=args.action_rate_weight, tracking_axes=args.tracking_axes,
                  action_rate_domain=args.action_rate_domain,
                  swing_reference=args.swing_reference, standing_fraction=args.standing_fraction,
                  mirror_loss_weight=args.mirror_loss_weight,
                  tracking_mean_seconds=args.tracking_mean_seconds,
                  straight_yaw_std=args.straight_yaw_std,
                  airtime_height_gate=args.airtime_height_gate,
                  feedback_age_max_steps=args.feedback_age_max_steps,
                  bilateral_clearance_weight=args.bilateral_clearance_weight,
                  bilateral_clearance_target=args.bilateral_clearance_target,
                  clearance_course_mm=args.clearance_course_mm,
                  resume_exploration_std_cap=args.resume_exploration_std_cap,
                  fixed_exploration_std=args.fixed_exploration_std,
                  walking_hip_roll_std=args.walking_hip_roll_std,
                  airtime_window_shift=args.airtime_window_shift,
                  terrain_course=args.terrain_course, slew_demand_weight=args.slew_demand_weight,
                  forward_probability=args.forward_probability,
                  walking_flexion_scale=args.walking_flexion_scale,
                  voltage_domain=args.voltage_domain)
    task_id = (STEP_TASK_ID if args.repair_variant in ('step', 'step_balanced', 'step_right_lift', 'step_anchored', 'step_mild_right', 'step_hold_robust', 'step_delay_curriculum', 'step_joint_age', 'step_age_drift', 'step_lift35_age', 'step_lift35', 'step_coherent_age') else
               SITSTAND_TASK_ID if args.repair_variant == 'sitstand' else
               ROULADE_TASK_ID if args.repair_variant == 'roulade' else TASK_ID)
    if args.resume_checkpoint:
        import onnx
        import torch
        from mjlab_microduck.actuator.cpu_hd1910_bam import KP_FW, PROFILE_PATH
        checkpoint = args.resume_checkpoint.resolve(strict=True)
        policies = list(checkpoint.parent.glob('*.onnx'))
        if len(policies) != 1:
            parser.error('resume requires one companion ONNX physics/task contract')
        metadata = {p.key:p.value for p in onnx.load(policies[0]).metadata_props}
        if metadata.get('voltage_domain') == 'static_home' and args.voltage_domain is None:
            parser.error('voltage-adapted parent requires explicit --voltage-domain')
        if metadata.get('head_bias_course') == 'frozen' and args.head_bias_course is None:
            parser.error('frozen parent requires an explicit --head-bias-course; do not silently restore its curriculum')
        if 'fixed_action_rate_weight' in metadata and args.action_rate_weight is None:
            parser.error('fixed action-rate parent requires explicit --action-rate-weight')
        if metadata.get('tracking_axes') in ('separate', 'separate_yaw') and args.tracking_axes is None:
            parser.error('separate-axis parent requires explicit --tracking-axes')
        if metadata.get('swing_height_reference') == 'collision' and args.swing_reference is None:
            parser.error('collision swing parent requires explicit --swing-reference')
        if metadata.get('airtime_height_gate') in ('gentle', 'on') and args.airtime_height_gate is None:
            parser.error('height-gated parent requires explicit --airtime-height-gate')
        if 'feedback_age_max_steps' in metadata and args.feedback_age_max_steps is None:
            parser.error('feedback-age parent requires explicit --feedback-age-max-steps')
        if 'bilateral_clearance_weight' in metadata and args.bilateral_clearance_weight is None:
            parser.error('bilateral parent requires explicit --bilateral-clearance-weight')
        if 'bilateral_clearance_target' in metadata and args.bilateral_clearance_target is None:
            parser.error('height-course parent requires explicit --bilateral-clearance-target')
        if 'clearance_course_mm' in metadata and args.clearance_course_mm is None:
            parser.error('unified-course parent requires explicit --clearance-course-mm')
        if 'walking_hip_roll_std' in metadata and args.walking_hip_roll_std is None:
            parser.error('posture parent requires explicit --walking-hip-roll-std')
        if 'airtime_window_shift' in metadata and args.airtime_window_shift is None:
            parser.error('swing timing parent requires explicit --airtime-window-shift')
        if 'terrain_course' in metadata and args.terrain_course is None:
            parser.error('terrain parent requires explicit --terrain-course')
        if 'slew_demand_weight' in metadata and args.slew_demand_weight is None:
            parser.error('slew demand parent requires explicit --slew-demand-weight')
        if 'forward_probability' in metadata and args.forward_probability is None:
            parser.error('forward curriculum parent requires explicit --forward-probability')
        if 'walking_flexion_scale' in metadata and args.walking_flexion_scale is None:
            parser.error('flexion parent requires explicit --walking-flexion-scale')
        if 'fixed_exploration_std' in metadata and args.fixed_exploration_std is None:
            parser.error('fixed-noise parent requires explicit --fixed-exploration-std')
        if metadata.get('action_rate_domain') == 'latent' and args.action_rate_domain is None:
            parser.error('latent-rate parent requires explicit --action-rate-domain')
        for key, value in (('fixed_standing_fraction', args.standing_fraction),
                           ('straight_yaw_std', args.straight_yaw_std),
                           ('mirror_loss_weight', args.mirror_loss_weight),
                           ('tracking_mean_seconds', args.tracking_mean_seconds)):
            if key in metadata and value is None:
                parser.error(f'{key} parent requires an explicit matching curriculum choice')
        required = dict(task_id=task_id, kp_fw=str(KP_FW), actuator_backend='hd1910_bam_m6',
                        calibration_sha256=hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
                        action_semantics='bounded_slew_home_delta_v2',
                        training_recipe='m6_repair_' + str(args.repair_variant) + '_v1')
        if any(metadata.get(k) != v for k, v in required.items()):
            parser.error('resume task/physics/action contract mismatch; use a deliberate warm start')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if checkpoint.name != f'model_{payload["iter"]}.pt':
            parser.error('checkpoint filename and saved iteration disagree')
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        from mjlab.tasks.registry import load_rl_cfg
        target = Path('logs/rsl_rl') / load_rl_cfg(task_id).experiment_name / ('resume_' + digest[:12])
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, target / checkpoint.name)
        (target / 'resume.json').write_text(json.dumps(dict(source=str(checkpoint), sha256=digest,
            completed_iteration=payload['iter'], contract=required,
            parent_head_bias_course=metadata.get('head_bias_course', 'scheduled'),
            requested_head_bias_course=args.head_bias_course,
            parent_action_rate_weight=metadata.get('fixed_action_rate_weight'),
            requested_action_rate_weight=args.action_rate_weight,
            requested_action_rate_domain=args.action_rate_domain,
            parent_voltage_domain=metadata.get('voltage_domain', 'nominal'),
            requested_voltage_domain=args.voltage_domain,
            parent_tracking_axes=metadata.get('tracking_axes', 'coupled'),
            requested_tracking_axes=args.tracking_axes,
            parent_swing_reference=metadata.get('swing_height_reference', 'ray'),
            requested_swing_reference=args.swing_reference,
            requested_standing_fraction=args.standing_fraction,
            requested_mirror_loss_weight=args.mirror_loss_weight,
            requested_tracking_mean_seconds=args.tracking_mean_seconds,
            requested_straight_yaw_std=args.straight_yaw_std,
            requested_airtime_height_gate=args.airtime_height_gate,
            requested_feedback_age_max_steps=args.feedback_age_max_steps,
            requested_bilateral_clearance_weight=args.bilateral_clearance_weight,
            requested_bilateral_clearance_target=args.bilateral_clearance_target,
            requested_clearance_course_mm=args.clearance_course_mm,
            requested_resume_exploration_std_cap=args.resume_exploration_std_cap,
            requested_fixed_exploration_std=args.fixed_exploration_std,
            requested_walking_hip_roll_std=args.walking_hip_roll_std,
            requested_airtime_window_shift=args.airtime_window_shift,
            requested_terrain_course=args.terrain_course,
            requested_slew_demand_weight=args.slew_demand_weight,
            requested_forward_probability=args.forward_probability,
            requested_walking_flexion_scale=args.walking_flexion_scale), indent=2) + '\n')
        rest += ['--agent.resume', 'True', '--agent.load-run', target.name,
                 '--agent.load-checkpoint', checkpoint.name]
    sys.argv = [sys.argv[0], task_id, *rest]
    from mjlab.scripts.train import main
    main()
