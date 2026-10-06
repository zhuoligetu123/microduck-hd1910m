#!/usr/bin/env python3
"""CPU M6 phase-conditioned stepping; report actual sole clearance and drift."""
import argparse
import hashlib
import json
import math
import mujoco
import numpy as np
from pathlib import Path
from replay_hd1910 import DEFAULT_POSE, ReplayPolicy, head_optical_pitch_deg, joint_age_lags_from_capture, load_replay_model, step_control_period, validate_metadata


def lift_events(heights):
    """A lift needs ground contact followed by >=60 ms above 3 mm, not sign jitter."""
    grounded = [False, False]
    airborne = [False, False]
    above = [0, 0]
    events = []
    for row in heights:
        for foot, height in enumerate(row):
            if height <= .001:
                grounded[foot] = True
                airborne[foot] = False
                above[foot] = 0
            elif height >= .003 and grounded[foot] and not airborne[foot]:
                above[foot] += 1
                if above[foot] == 3:
                    airborne[foot] = True
                    events.append(foot)
            else:
                above[foot] = 0
    counts = [events.count(foot) for foot in (0, 1)]
    alternations = sum(a != b for a, b in zip(events, events[1:]))
    return counts, alternations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=20)
    parser.add_argument('--delay-steps', type=int, default=4, choices=range(3, 11))
    parser.add_argument('--command-loss-probability', type=float, default=0.)
    parser.add_argument('--command-hold-max-steps', type=int, default=3)
    parser.add_argument('--joint-age-steps', type=int, default=0, choices=range(5))
    parser.add_argument('--joint-age-capture', type=Path)
    args = parser.parse_args()
    if not 0 <= args.command_loss_probability < .5 or not 1 <= args.command_hold_max_steps <= 5:
        parser.error('invalid command delivery stress')
    if not 20 <= args.seconds <= 180:
        parser.error('seconds must be between 20 and 180')
    if args.joint_age_capture and args.joint_age_steps:
        parser.error('choose a fixed joint age or a captured age sequence')
    captured_ages = joint_age_lags_from_capture(args.joint_age_capture) if args.joint_age_capture else None
    frames = args.seconds * 50
    model, data, motor = load_replay_model(7.4, bam_reference=True, repair_variant='step')
    policy = ReplayPolicy(model, data, bam_ctrl=motor, walking_onnx_path=str(args.policy),
                          new_cmd_obs=True, use_projected_gravity=True)
    policy.set_joint_observation_delay(4 if captured_ages is not None else args.joint_age_steps)
    validate_metadata(policy.ort_session.get_modelmeta().custom_metadata_map,
        [model.joint(int(i)).name for i in motor.joint_ids], bam_reference=True, step=True)
    feet = []
    joint_limits = model.jnt_range[model.actuator_trnid[:, 0]]
    camera_id = model.camera('head_camera').id
    for name in ('left_foot_collision', 'right_foot_collision'):
        gid = model.geom(name).id
        mesh = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        feet.append((gid, model.mesh_vert[start:start+count].copy()))
    rows = []
    for seed_index, seed in enumerate((42, 7, 123)):
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = [0, 0, .125, 1, 0, 0, 0]
        data.qpos[motor.qids] = DEFAULT_POSE + np.random.default_rng(seed).uniform(-.005, .005, 14)
        motor.reset(data.qpos)
        motor.delay = args.delay_steps
        mujoco.mj_forward(model, data)
        policy.last_action[:] = 0
        policy.reset_joint_observation_history()
        previous = DEFAULT_POSE.copy()
        rng = np.random.default_rng(seed + 1000)
        hold_left = held_targets = 0
        samples = []
        xy_samples = []
        near_limit = np.zeros(14, dtype=int)
        limit_violations = 0
        head_pitches = []
        for frame in range(frames):
            if captured_ages is not None:
                policy.current_joint_age_steps = int(captured_ages[(frame + 1000 * seed_index) % len(captured_ages)])
            phase = 2*math.pi*(frame+1)/50
            policy.command[:] = 0
            policy.command[:2] = [math.cos(phase), math.sin(phase)]
            action = policy.infer()
            target = policy.default_pose + action * policy.action_scale
            if hold_left == 0 and rng.random() < args.command_loss_probability:
                hold_left = int(rng.integers(1, args.command_hold_max_steps + 1))
            if hold_left:
                hold_left -= 1
                held_targets += 1
                target = previous.copy()
                policy.last_action = ((target - policy.default_pose) / policy.action_scale).astype(np.float32)
            previous = target.copy()
            margins = np.minimum(target - joint_limits[:, 0], joint_limits[:, 1] - target)
            near_limit += margins < .03
            limit_violations += int(np.any(margins < -1e-6))
            policy.set_position_targets(target)
            step_control_period(model, data, motor)
            head_pitches.append(head_optical_pitch_deg(data.cam_xmat[camera_id]))
            height = [float((vertices @ data.geom_xmat[gid].reshape(3,3).T + data.geom_xpos[gid])[:,2].min())
                      for gid, vertices in feet]
            tilt = math.degrees(math.acos(float(np.clip(-policy.get_projected_gravity()[2], -1, 1))))
            samples.append([*height, float(np.linalg.norm(data.qpos[:2])), tilt])
            xy_samples.append(data.qpos[:2].copy())
        values = np.asarray(samples)
        xy = np.asarray(xy_samples)
        peaks = np.quantile(values[50:, :2], .95, axis=0)
        counts, alternations = lift_events(values[50:, :2])
        no_fall = bool(np.isfinite(values).all() and values[:,3].max() < 45)
        rows.append(dict(seed=seed, no_fall=no_fall, sole_p95_m=peaks.tolist(),
            drift_max_m=float(values[:,2].max()), alternations=alternations,
            displacement_xy_final_m=xy[-1].tolist(),
            max_axis_excursion_m=np.abs(xy).max(axis=0).tolist(),
            lift_event_counts=counts, lift_threshold_m=.003, minimum_lift_duration_s=.06,
            simulated_target_hold_fraction=held_targets/frames,
            target_near_limit_fraction_by_joint=(near_limit/frames).tolist(),
            target_limit_violations=limit_violations,
            head_optical_pitch_p95_deg=float(np.quantile(head_pitches[50:], .95)),
            passed=bool(no_fall and min(peaks) >= .015 and values[:,2].max() < .15
                        and alternations >= args.seconds)))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(cases=rows, hardware_tested=False,
        seconds_per_case=args.seconds,
        actuator_delay_ms=args.delay_steps*5,
        command_loss_probability=args.command_loss_probability,
        command_hold_max_steps=args.command_hold_max_steps,
        joint_observation_age_ms=args.joint_age_steps*20 if captured_ages is None else None,
        joint_age_capture_sha256=hashlib.sha256(args.joint_age_capture.read_bytes()).hexdigest() if args.joint_age_capture else None,
        joint_age_capture_lag_counts=np.bincount(captured_ages, minlength=5).tolist() if captured_ages is not None else None,
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest()), indent=2)+'\n')


if __name__ == '__main__':
    main()
