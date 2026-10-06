#!/usr/bin/env python3
"""Continuous M6 pitch/get-up tests. No auto-reset, policy switching or hardware I/O."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from replay_hd1910 import ReplayPolicy, DEFAULT_POSE, head_optical_pitch_deg, load_replay_model, step_control_period, validate_metadata


def recovery_success(rows):
    """Require the final two seconds upright and actually supported by the feet."""
    tail = np.asarray(rows, dtype=float)[-100:]
    if len(tail) != 100 or not np.isfinite(tail).all():
        return False
    return bool(np.all(tail[:, 1] < 20) and np.all(tail[:, 2] > .095)
                and np.mean(tail[:, 3] == 2) >= .8 and np.all(tail[:, 4] == 0))


def floor_clearance(model, data):
    lowest = float('inf')
    for gid in range(model.ngeom):
        if not model.geom_contype[gid] & 1 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[gid]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start:start+model.mesh_vertnum[mesh]]
        lowest = min(lowest, float(np.min(vertices @ data.geom_xmat[gid].reshape(3, 3)[2]
                                          + data.geom_xpos[gid, 2])))
    if not math.isfinite(lowest):
        raise ValueError('no floor-contact mesh found')
    return lowest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--suite', choices=('pitch', 'recovery'), required=True)
    p.add_argument('--axis', choices=('pitch', 'roll'), default='pitch',
                   help='Body-frame disturbance axis for the walking suite')
    p.add_argument('--side-cases', action='store_true',
                   help='Also evaluate left/right side-lying starts in the recovery suite')
    p.add_argument('--seconds', type=float, default=16.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--voltage', type=float, default=7.4)
    p.add_argument('--video-dir', type=Path)
    args = p.parse_args()
    if args.suite == 'recovery' and args.axis != 'pitch':
        p.error('get-up suite only defines front/back spawns')
    if not math.isfinite(args.seconds) or not 12 <= args.seconds <= 120:
        p.error('duration must be within 12..120 seconds')
    metadata = ort.InferenceSession(str(args.policy)).get_modelmeta().custom_metadata_map
    is_recovery = metadata.get('policy_role') == 'recovery'
    model, data, motor = load_replay_model(args.voltage, bam_reference=True,
                                         repair_variant='recovery' if is_recovery else None)
    policy = ReplayPolicy(model, data, walking_onnx_path=str(args.policy), bam_ctrl=motor,
                          new_cmd_obs=True, use_projected_gravity=True)
    validate_metadata(metadata, [model.joint(int(j)).name for j in motor.joint_ids], bam_reference=True)
    if policy.ort_session.get_inputs()[0].shape != [1,61] or policy.ort_session.get_outputs()[0].shape != [1,14]:
        raise ValueError('61 -> 14 required')
    if args.suite == 'recovery' and not is_recovery:
        # Evaluate the walking parent on the SAME contact model as the specialist.
        model, data, motor = load_replay_model(args.voltage, bam_reference=True, repair_variant='recovery')
        policy = ReplayPolicy(model, data, walking_onnx_path=str(args.policy), bam_ctrl=motor,
                              new_cmd_obs=True, use_projected_gravity=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.video_dir:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    cases = [('front', 90, 0.), ('back', -90, 0.), ('front_partial', 50, 0.),
             ('back_partial', -50, 0.), ('stand', 0, 0.)] if args.suite == 'recovery' else [
             ('forward_pitch', 12, .1), ('backward_pitch', -12, -.1),
             ('forward_push', 0, .1), ('backward_push', 0, -.1), ('stand_push', 0, 0.)]
    if args.suite == 'recovery' and args.side_cases:
        cases += [('left_side', 0, 0.), ('right_side', 0, 0.)]
    floor = model.geom('floor').id
    camera_id = model.camera('head_camera').id
    feet = {model.geom('left_foot_collision').id, model.geom('right_foot_collision').id}
    rng = np.random.default_rng(args.seed)
    results = []
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = .4
    for name, pitch, vx in cases:
        if args.axis == 'roll':
            name = name.replace('_pitch', '_roll').replace('_push', '_side_push')
        mujoco.mj_resetData(model, data)
        angle = math.radians(pitch)
        side_roll = 90 if name == 'left_side' else -90 if name == 'right_side' else 0
        if side_roll:
            half = math.radians(side_roll) / 2
            data.qpos[:7] = [0, 0, .125, math.cos(half), math.sin(half), 0, 0]
        else:
            data.qpos[:7] = [0, 0, .125, math.cos(angle/2),
                             math.sin(angle/2) if args.axis == 'roll' else 0,
                             math.sin(angle/2) if args.axis == 'pitch' else 0, 0]
        data.qpos[policy.joint_qpos_indices] = DEFAULT_POSE + rng.uniform(-.005, .005, 14)
        mujoco.mj_forward(model, data)
        data.qpos[2] += .002 - floor_clearance(model, data)
        mujoco.mj_forward(model, data)
        motor.reset(data.qpos)
        motor.delay = 6
        policy.last_action[:] = 0
        policy.previous_velocity = None
        policy.set_vel_cmd(vx, 0., 0.)
        rows = []
        with contextlib.ExitStack() as stack:
            renderer = writer = None
            if args.video_dir and name in ('front', 'back', 'forward_push'):
                import imageio.v2 as imageio
                renderer = stack.enter_context(mujoco.Renderer(model, height=480, width=640))
                writer = stack.enter_context(imageio.get_writer(str(args.video_dir/(name+'.mp4')), fps=25))
            camera = mujoco.MjvCamera()
            camera.distance, camera.elevation, camera.azimuth = .65, -15, 90
            trace = csv.writer(stack.enter_context(args.report.with_name(args.report.stem+'_'+name+'.csv').open('w')))
            trace.writerow(['time_s','tilt_deg','trunk_z_m','feet_contacts','nonfeet_floor_contacts','pitch_deg','head_optical_pitch_deg'])
            for step in range(round(args.seconds*50)):
                if args.suite == 'pitch' and step in (100, 300, 500):
                    sign = 1 if step != 300 else -1
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
                    linear_axis = 0 if args.axis == 'pitch' else 1
                    data.qvel[:2] += sign * .25 * rotation.reshape(3,3)[:2,linear_axis]
                    # MuJoCo free-joint angular qvel is body-local.
                    # A lateral +Y push corresponds to negative body roll.
                    data.qvel[4 if args.axis == 'pitch' else 3] += sign * (1.2 if args.axis == 'pitch' else -1.2)
                    mujoco.mj_forward(model, data)
                action = policy.infer()
                target = policy.default_pose + action*policy.action_scale
                if not np.isfinite(target).all():
                    raise ValueError('nonfinite target')
                policy.set_position_targets(target)
                step_control_period(model, data, motor)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError('nonfinite simulation')
                gravity = policy.get_projected_gravity()
                tilt = math.degrees(math.acos(float(np.clip(-gravity[2], -1, 1))))
                touching, other = set(), 0
                for contact in data.contact:
                    pair = set(map(int, contact.geom))
                    if floor in pair and contact.dist <= .001:
                        gids = pair - {floor}
                        touching.update(gids & feet)
                        other += bool(gids - feet)
                row = [(step+1)/50, tilt, float(data.qpos[2]), len(touching), other,
                       math.degrees(math.asin(float(np.clip(gravity[0], -1, 1)))),
                       head_optical_pitch_deg(data.cam_xmat[camera_id])]
                rows.append(row)
                trace.writerow(row)
                if renderer and step % 2 == 0:
                    camera.lookat[:] = data.qpos[:3]
                    renderer.update_scene(data, camera=camera)
                    writer.append_data(renderer.render())
        values = np.asarray(rows)
        results.append(dict(case=name, initial_pitch_deg=pitch if args.axis == 'pitch' else 0,
            initial_roll_deg=side_roll or (pitch if args.axis == 'roll' else 0), completed=True,
            standing_at_end=recovery_success(rows), max_tilt_deg=float(values[:,1].max()),
            tilt_over_40_fraction=float(np.mean(values[:,1] > 40)),
            head_optical_pitch_p95_deg=float(np.quantile(values[:,6], .95)),
            head_optical_pitch_max_deg=float(values[:,6].max()),
            final_tilt_deg=float(values[-1,1]), final_height_m=float(values[-1,2]),
            fell_after_start=bool(np.any(values[50:,1] > 60)),
            recovered_from_ground=bool((abs(pitch) == 90 or side_roll) and recovery_success(rows))))
    report = dict(policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        suite=args.suite, seed=args.seed, seconds=args.seconds, cases=results,
        automatic_resets=0, hardware_tested=False, deployment_ready=False,
        voltage_v=args.voltage, actuator_delay_ms=30,
        impulse_frame='body_sagittal_planar' if args.axis == 'pitch' else 'body_lateral_planar',
        collision_model='groundcontact' if is_recovery or args.suite == 'recovery' else 'walk',
        success_definition='final 2s: tilt <20deg, trunk >95mm, both feet contact >=80%, no other floor contacts')
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
