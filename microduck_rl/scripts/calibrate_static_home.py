#!/usr/bin/env python3
"""Offline HOME reference calibration; never connects to or commands hardware.

Loaded position errors are validation residuals, NOT encoder zero corrections.
Current is uncalibrated servo telemetry, NOT measured joint torque. This one
pose cannot identify PD, friction, inertia, dynamic delay or mechanical offsets.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from mjlab_microduck.actuator.radxa_alignment import JOINTS, RAD_PER_TICK


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    array = np.asarray(values, dtype=float)
    return dict(mean=float(array.mean()), min=float(array.min()), max=float(array.max()),
                p05=float(np.percentile(array, 5)), p50=float(np.median(array)),
                p95=float(np.percentile(array, 95)))


def extract_reference(path):
    frames, seen, mapping = [], set(), None
    lines = 0
    with Path(path).open() as stream:
        for line in stream:
            state = json.loads(line)
            lines += 1
            f = state['feedback']
            joints, motors = f['joints'], f['states']
            if (state['policy'] != 'held' or f['policy_enabled'] or not f['homed']
                    or not f['control_valid'] or not f['imu_valid'] or f['error']
                    or not f['servo_gains_verified'] or f['servo_gain_profile'] != 'luwu_runtime'
                    or not 0 <= f['joint_age_s'] < .1 or not 0 <= f['imu_age_s'] < .05):
                raise ValueError('capture must contain fresh, alarm-free, homed HOLD only')
            if ([j['name'] for j in joints] != list(JOINTS)
                    or {j['id'] for j in joints} != set(range(1, 16))
                    or len(motors) != 15 or len(f['positions']) != 15 or len(state['targets']) != 15):
                raise ValueError('joint mapping/shape mismatch')
            if mapping is None:
                mapping = joints
            if joints != mapping:
                raise ValueError('installation mapping changed during capture')
            values = [*f['positions'], *state['targets'], *f['imu']['gravity'], *f['imu']['gyro']]
            if not np.isfinite(values).all():
                raise ValueError('nonfinite state')
            gravity = np.asarray(f['imu']['gravity'])
            if not .98 <= np.linalg.norm(gravity) <= 1.02:
                raise ValueError('invalid gravity norm')
            tilt = math.degrees(math.acos(float(np.clip(-gravity[2], -1, 1))))
            if tilt > 10 or np.linalg.norm(f['imu']['gyro']) > .15:
                raise ValueError('reference is not upright and stationary')
            for joint, motor, q in zip(joints, motors, f['positions']):
                if (joint['direction'] not in (-1, 1) or not 0 <= joint['zero_ticks'] < 4096
                        or motor['servo_id'] != joint['id'] or not motor['valid']
                        or motor['torque_enabled'] != 1 or motor['status'] != 0):
                    raise ValueError('invalid motor state or installation')
                expected = joint['direction'] * (motor['position_ticks'] - joint['zero_ticks']) * RAD_PER_TICK
                if abs(expected - q) > 1e-6:
                    raise ValueError('encoder/model coordinate mismatch')
                if (not np.isfinite([motor['current_a'], motor['voltage_v']]).all()
                        or abs(motor['current_a']) > 2.5 or not 0 < motor['voltage_v'] < 12):
                    raise ValueError('implausible telemetry; do not use for calibration')
            if f['sequence'] in seen:
                continue
            if frames and state['t'] <= frames[-1]['t']:
                raise ValueError('capture crosses a restart or reverses time')
            seen.add(f['sequence'])
            frames.append(state)
    if len(frames) < 100 or frames[-1]['t'] - frames[0]['t'] < 10:
        raise ValueError('need at least 10 seconds and 100 independent snapshots')
    q = np.array([s['feedback']['positions'] for s in frames])
    targets = np.array([s['targets'] for s in frames])
    if np.ptp(q, axis=0).max() > math.radians(1) or np.ptp(targets, axis=0).max() > 1e-6:
        raise ValueError('joint positions or targets are not static')
    rows = []
    for i, joint in enumerate(mapping):
        motor_samples = [s['feedback']['states'][i] for s in frames]
        rows.append(dict(**joint, target_rad=float(targets[0, i]),
            measured_rad=float(np.median(q[:, i])), measured_span_rad=float(np.ptp(q[:, i])),
            measured_minus_target_rad=float(np.median(q[:, i]) - targets[0, i]),
            current_abs_a=stats([abs(m['current_a']) for m in motor_samples]),
            current_signed_a=stats([m['current_a'] for m in motor_samples]),
            voltage_v=stats([m['voltage_v'] for m in motor_samples])))
    return dict(schema=1, source=str(Path(path).resolve()), source_sha256=sha256(path),
        frames=lines, unique_snapshots=len(frames), seconds=frames[-1]['t']-frames[0]['t'],
        joints=rows, policy_joint_names=[j for j in JOINTS if j != 'mouth'],
        tilt_deg=stats([math.degrees(math.acos(float(np.clip(-s['feedback']['imu']['gravity'][2], -1, 1)))) for s in frames]),
        frame_median_voltage_v=stats([np.median([m['voltage_v'] for m in s['feedback']['states']]) for s in frames]),
        applied_encoder_offset_rad=[0.]*15, applied_bam_parameter_changes={},
        external_support_measured=False, current_to_torque_identified=False,
        role='static validation and initialization reference, not a new policy or mechanical zero',
        limits=['No measured external support/contact forces',
                'Voltage telemetry is not independently calibrated',
                'One loaded pose cannot identify zero, PD, friction, inertia or delay'])


def simulate(reference, policy_path, output):
    import mujoco
    from replay_hd1910 import ReplayPolicy, load_replay_model, validate_metadata
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME
    from mjlab_microduck.actuator.cpu_xgoduck_bam import PROFILE_PATH, KP_FW

    measured_v = reference['frame_median_voltage_v']['p50']
    if not 6 <= measured_v <= 8:
        raise ValueError('measured voltage outside supported simulation extrapolation')
    rows = [r for r in reference['joints'] if r['name'] != 'mouth']
    names = [r['name'] for r in rows]
    target = np.array([r['target_rad'] for r in rows])
    measured = np.array([r['measured_rad'] for r in rows])
    training_home = []
    for name in names:
        matches = [v for pattern, v in HOME_FRAME.joint_pos.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError('ambiguous training HOME')
        training_home.append(matches[0])
    np.testing.assert_allclose(target, training_home, atol=1e-7, rtol=0)
    cases = []
    for voltage in sorted({7.4, measured_v}):
        model, data, motor = load_replay_model(voltage, bam_reference=True,
            voltage_extrapolation=voltage < 7., ground_contact=True,
            repair_variant='gait_luwu_curriculum_scaled_v21')
        if [model.joint(int(j)).name for j in motor.joint_ids] != names:
            raise ValueError('simulator joint order mismatch')
        policy = ReplayPolicy(model, data, walking_onnx_path=str(policy_path),
                              bam_ctrl=motor, new_cmd_obs=True, use_projected_gravity=True)
        validate_metadata(policy.ort_session.get_modelmeta().custom_metadata_map, names, bam_reference=True)
        np.testing.assert_allclose(policy.default_pose, target, atol=1e-7, rtol=0)
        feet = []
        for name in ('left_foot_collision', 'right_foot_collision'):
            gid = model.geom(name).id
            mesh = model.geom_dataid[gid]
            start = model.mesh_vertadr[mesh]
            feet.append((gid, model.mesh_vert[start:start+model.mesh_vertnum[mesh]].copy()))
        for initial_name, initial in (('home', target), ('observed', measured)):
            for seed in (0, 42, 123):
                mujoco.mj_resetData(model, data)
                data.qpos[:7] = [0, 0, .125, 1, 0, 0, 0]
                noise = np.random.default_rng(seed).uniform(-.005, .005, 14) if seed else np.zeros(14)
                data.qpos[motor.qids] = initial + noise
                mujoco.mj_forward(model, data)
                lowest = min(float((vertices @ data.geom_xmat[gid].reshape(3, 3)[2]
                                   + data.geom_xpos[gid, 2]).min()) for gid, vertices in feet)
                data.qpos[2] += .0005 - lowest
                mujoco.mj_forward(model, data)
                motor.reset(data.qpos)
                motor.q_target = target.copy()
                history, failure = [], None
                for step in range(round(10/model.opt.timestep)):
                    motor.update()
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    if not np.isfinite(data.qpos).all():
                        raise ValueError('nonfinite physics')
                    w, x, y, z = data.qpos[3:7]
                    tilt = math.degrees(math.acos(float(np.clip(1-2*(x*x+y*y), -1, 1))))
                    if tilt > 60 or data.qpos[2] < .06:
                        failure = float(data.time)
                        break
                    history.append([float(data.time), tilt, *data.qpos[motor.qids],
                                    float(np.max(np.abs(data.qvel[motor.vids])))])
                settled = np.array([r for r in history if r[0] >= 7])
                case = dict(voltage_v=voltage, voltage_is_extrapolation=voltage < 7.,
                    initial=initial_name, seed=seed, first_failure_s=failure,
                    max_tilt_deg=max((r[1] for r in history), default=0.))
                if failure is None and len(settled):
                    q_mean = settled[:, 2:16].mean(axis=0)
                    case.update(mean_position_rad=q_mean.tolist(),
                        sim_minus_real_deg=np.degrees(q_mean-measured).tolist(),
                        position_rmse_deg=float(np.sqrt(np.mean(np.degrees(q_mean-measured)**2))),
                        final_tilt_deg=float(settled[-1, 1]),
                        final_max_joint_speed_rad_s=float(settled[-1, -1]))
                cases.append(case)
    report = dict(policy_sha256=sha256(policy_path), reference_sha256=reference['source_sha256'],
        policy_joint_names=names, home_contract_matches=True, rl_inference_executed=False,
        script_sha256=sha256(__file__), bam_profile_sha256=sha256(PROFILE_PATH),
        kp_fw=KP_FW, physics_dt_s=model.opt.timestep, total_mass_kg=float(model.body_mass.sum()),
        geometry='groundcontact model with nominal replica masses; no payload randomization',
        initial_base_orientation='upright identity quaternion, free base thereafter; not IMU compensation',
        voltage_model='one scalar median for all joints, not a reproduced electrical network',
        simulation='free base, floor contacts, M6/P6, 10 s HOME hold, last 3 s evaluated',
        cases=cases, limitations=reference['limits'], hardware_parameters_identified=False)
    (output/'simulation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(cases=len(cases), no_fall=sum(c['first_failure_s'] is None for c in cases),
                          measured_voltage_v=measured_v, policy_sha256=report['policy_sha256'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = extract_reference(args.capture)
    args.output.mkdir(parents=True, exist_ok=False)
    reference['baseline_policy_sha256'] = sha256(args.policy)
    (args.output/'reference.json').write_text(json.dumps(reference, indent=2, allow_nan=False)+'\n')
    simulate(reference, args.policy, args.output)


if __name__ == '__main__':
    main()
