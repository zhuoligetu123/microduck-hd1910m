#!/usr/bin/env python3
"""Render independent, real-time MuJoCo scenarios; no hardware or automatic recovery switch."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from replay_hd1910 import DEFAULT_POSE, ReplayPolicy, load_replay_model, step_control_period, validate_metadata
from replay_m6_recovery import floor_clearance, recovery_success


SCENARIOS = [
    ('stand', 'Standing', (0., 0., 0.), 0, False),
    ('forward', 'Walk forward', (.1, 0., 0.), 0, False),
    ('backward', 'Walk backward', (-.1, 0., 0.), 0, False),
    ('left', 'Turn left', (0., 0., .4), 0, False),
    ('right', 'Turn right', (0., 0., -.4), 0, False),
    ('forward_push', 'Forward + sagittal disturbances', (.1, 0., 0.), 0, True),
    ('backward_push', 'Backward + sagittal disturbances', (-.1, 0., 0.), 0, True),
    ('front', 'Get up from front-down', (0., 0., 0.), 90, False),
    ('back', 'Get up from back-down', (0., 0., 0.), -90, False),
]


def add_ground_grid(scene, position):
    """World-fixed visual reference only; does not participate in contacts."""
    center = np.floor(position[:2] * 10) / 10
    for offset in range(-12, 13):
        for axis in (0, 1):
            start = np.array([center[0]-1.2, center[1]-1.2, .0002])
            end = np.array([center[0]+1.2, center[1]+1.2, .0002])
            start[axis] = end[axis] = center[axis] + offset * .1
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                np.zeros(3), np.eye(3).ravel(), np.array([.58, .67, .69, .4]))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1., start, end)
            scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--walk', type=Path, required=True)
    parser.add_argument('--recovery', type=Path,
                        help='Include the separate get-up policy; omit for walk-only disturbance preview')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--segment-seconds', type=int, default=20)
    parser.add_argument('--five-gait', action='store_true',
                        help='Render stand, forward, left, backward, right only')
    args = parser.parse_args()
    if not 1 <= args.segment_seconds <= 120:
        parser.error('segment-seconds must be 1..120')
    scenarios = SCENARIOS if args.recovery else SCENARIOS[:7] + [
        ('forward_side_push', 'Forward + lateral disturbances', (.1, 0., 0.), 0, True),
        ('backward_side_push', 'Backward + lateral disturbances', (-.1, 0., 0.), 0, True),
    ]
    if args.five_gait:
        if args.recovery:
            parser.error('--five-gait cannot be combined with --recovery')
        scenarios = [next(scene for scene in SCENARIOS if scene[0] == name)
                     for name in ('stand', 'forward', 'left', 'backward', 'right')]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps = 1280, 720, 25
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title_font = ImageFont.truetype(font_path, 28)
    font = ImageFont.truetype(font_path, 18)
    steps = args.segment_seconds * 50
    total_frames = 0
    results = []
    models = {}
    with contextlib.ExitStack() as stack:
        policies = [('locomotion', args.walk)]
        if args.recovery:
            policies.append(('recovery', args.recovery))
        for role, path in policies:
            model, data, motor = load_replay_model(7.4, bam_reference=True,
                repair_variant='recovery' if role == 'recovery' else None)
            policy = ReplayPolicy(model, data, walking_onnx_path=str(path), bam_ctrl=motor,
                                  new_cmd_obs=True, use_projected_gravity=True)
            metadata = policy.ort_session.get_modelmeta().custom_metadata_map
            validate_metadata(metadata, [model.joint(int(j)).name for j in motor.joint_ids], bam_reference=True)
            if metadata.get('policy_role') != role:
                raise ValueError(f'{path}: expected policy_role={role}')
            model.vis.global_.offwidth, model.vis.global_.offheight = width, height
            model.vis.headlight.active = 1
            model.vis.headlight.ambient[:] = .4
            model.vis.headlight.diffuse[:] = .7
            renderer = stack.enter_context(mujoco.Renderer(model, height=height, width=width))
            models[role] = model, data, motor, policy, renderer
        writer = stack.enter_context(imageio.get_writer(str(args.output), fps=fps,
            codec='libx264', quality=8, macro_block_size=1,
            ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart']))
        trace = csv.writer(stack.enter_context(args.output.with_suffix('.csv').open('w')))
        trace.writerow(['scene', 'time_s', 'tilt_deg', 'trunk_z_m', 'feet_contacts', 'nonfeet_contacts'])
        for index, (name, title, command, pitch, push) in enumerate(scenarios):
            role = 'recovery' if pitch else 'locomotion'
            model, data, motor, policy, renderer = models[role]
            mujoco.mj_resetData(model, data)
            angle = math.radians(pitch)
            data.qpos[:7] = [0, 0, .125, math.cos(angle/2), 0, math.sin(angle/2), 0]
            data.qpos[policy.joint_qpos_indices] = DEFAULT_POSE + np.random.default_rng(42).uniform(-.005, .005, 14)
            mujoco.mj_forward(model, data)
            data.qpos[2] += .002 - floor_clearance(model, data)
            mujoco.mj_forward(model, data)
            motor.reset(data.qpos)
            motor.delay = 6
            policy.last_action[:] = 0
            policy.previous_velocity = None
            policy.set_vel_cmd(*command)
            floor = model.geom('floor').id
            feet = {model.geom('left_foot_collision').id, model.geom('right_foot_collision').id}
            camera = mujoco.MjvCamera()
            camera.distance, camera.elevation, camera.azimuth = .65, -18, 130
            rows = []
            for step in range(steps):
                if push and step in (100, 300, 500):
                    sign = -1 if step == 300 else 1
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
                    lateral = name.endswith('_side_push')
                    data.qvel[:2] += sign * .25 * rotation.reshape(3, 3)[:2, int(lateral)]
                    data.qvel[3 if lateral else 4] += sign * (-1.2 if lateral else 1.2)
                    mujoco.mj_forward(model, data)
                target = policy.default_pose + policy.infer() * policy.action_scale
                if not np.isfinite(target).all():
                    raise ValueError('nonfinite policy target')
                policy.set_position_targets(target)
                step_control_period(model, data, motor)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError('nonfinite simulation state')
                tilt = math.degrees(math.acos(float(np.clip(-policy.get_projected_gravity()[2], -1, 1))))
                touching, other = set(), 0
                for contact in data.contact:
                    pair = set(map(int, contact.geom))
                    if floor in pair and contact.dist <= .001:
                        gids = pair - {floor}
                        touching.update(gids & feet)
                        other += bool(gids - feet)
                rows.append([(step+1)/50, tilt, float(data.qpos[2]), len(touching), other])
                trace.writerow([name] + rows[-1])
                if step % 2 == 0:
                    camera.lookat[:] = [data.qpos[0], data.qpos[1], .11]
                    renderer.update_scene(data, camera=camera)
                    add_ground_grid(renderer.scene, data.qpos)
                    pixels = renderer.render()
                    if step in (0, steps//2, steps-2):
                        imageio.imwrite(args.output.with_name(f'{name}_{step:04d}.png'), pixels)
                    frame = Image.fromarray(pixels)
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, width, 88), fill=(14, 20, 28))
                    draw.text((24, 10), f'{index+1:02d} / {len(scenarios):02d}  {title}', font=title_font, fill=(101, 216, 240))
                    draw.text((24, 51), f'MuJoCo | HD1910M / BAM M6 | {role} | 1x simulation speed', font=font, fill='white')
                    if push and any(impulse <= step < impulse+25 for impulse in (100, 300, 500)):
                        draw.text((24, 104), 'LATERAL DISTURBANCE' if name.endswith('_side_push')
                                  else 'SAGITTAL DISTURBANCE', font=title_font, fill=(255, 180, 70))
                    draw.rectangle((0, height-66, width, height), fill=(14, 20, 28))
                    t = total_frames / fps
                    draw.text((24, height-58), f'{int(t)//60:02d}:{int(t)%60:02d} / {len(scenarios)*args.segment_seconds//60:02d}:{len(scenarios)*args.segment_seconds%60:02d}'
                              f'   Tilt {tilt:5.1f} deg   Height {data.qpos[2]*1000:5.0f} mm'
                              f'   vx {command[0]:+.2f} m/s   wz {command[2]:+.2f} rad/s', font=font, fill='white')
                    draw.text((24, height-32), 'Independent scene reset | 7.4 V | 30 ms actuator delay | No hardware test', font=font, fill=(164, 180, 191))
                    writer.append_data(np.asarray(frame))
                    total_frames += 1
            values = np.asarray(rows)
            result = dict(scene=name, role=role, seconds=args.segment_seconds,
                initial_pitch_deg=pitch, command=command, max_tilt_deg=float(values[:, 1].max()),
                final_tilt_deg=float(values[-1, 1]), standing_at_end=recovery_success(rows),
                locomotion_no_fall=None if pitch else bool(np.all(values[:, 1] <= 60) and np.all(values[:, 2] >= .06)))
            results.append(result)
            print(json.dumps(result), flush=True)
    assert total_frames == len(scenarios) * args.segment_seconds * fps
    report = dict(video=str(args.output.resolve()), frames=total_frames, fps=fps,
        seconds=total_frames/fps, resolution=[width, height], automatic_resets_within_scenes=0,
        independent_scene_count=len(scenarios), hardware_tested=False,
        policy_sha256={role: hashlib.sha256(path.read_bytes()).hexdigest()
                       for role, path in policies}, scenes=results)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Finished: {args.output} ({total_frames} frames)', flush=True)


if __name__ == '__main__':
    main()
