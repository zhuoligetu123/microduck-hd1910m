#!/usr/bin/env python3
"""Unmodified upstream ONNX on upstream XgoDuck geometry. Simulation only.

50 Hz inference / 200 Hz CPU MuJoCo, official BAM M6 equations. No local
bounded-action wrapper, target slew, reward changes, or hardware connection.
The runtime23 profile is a HOME/P/EMA sensitivity test, not an emulation of
the Arduino IMU estimator or serial transport. Foot height is the lowest
collision-mesh vertex above the flat floor, not the foot site height.
"""
import argparse
from collections import deque
from functools import partial
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import onnxruntime as ort
import mjlab
from bam.mjlab import BamActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.sim.sim import MujocoCfg
from mjlab_microduck.actuator.cpu_xgoduck_bam import XgoBamCpuController


PROFILES = {
    'metadata_p5': (5., 0., False),
    'ema_p6': (6., .45, False),
    'runtime23_p6': (6., .45, True),
}


def scenarios():
    result = [('stand', (0., 0., 0.), 0.)]
    for speed in (.1, .2, .3, -.1, -.2, -.3):
        result.append((f'vx_{speed:+.1f}', (speed, 0., 0.), 0.))
    for yaw in (.4, .8, -.4, -.8):
        result.append((f'wz_{yaw:+.1f}', (0., 0., yaw), 0.))
    result.extend([('curve', (.2, 0., .4), 0.),
                   ('push_1N', (.3, 0., 0.), 1.),
                   ('push_2N', (.3, 0., 0.), 2.)])
    return result


def command_at(t, requested, seconds):
    return np.asarray(requested if 2. <= t < seconds - 4. else (0., 0., 0.))


def build_observation(gyro, gravity, q, dq, home, previous, command):
    return np.concatenate((gyro, gravity, q-home, dq, previous, command,
                           np.zeros(10))).astype(np.float32)


def summarize_swings(samples):
    """Contact -> flight -> contact, ignoring initial air time and final flight."""
    seen = False
    flight = []
    swings = []
    for t, height, contact in samples:
        if contact:
            if seen and flight:
                duration = t-flight[0][0]
                if duration >= .04-1e-8:
                    swings.append((duration, max(x[1] for x in flight), t))
            seen = True
            flight = []
        elif seen:
            flight.append((t, height))
    peaks = [s[1]*1000 for s in swings]
    return dict(complete_swings=len(swings),
                peak_mm_median=float(np.median(peaks)) if peaks else None,
                peak_mm_p10_p90=np.quantile(peaks, [.1, .9]).tolist() if peaks else None,
                swings_over_3mm=int(sum(p >= 3 for p in peaks)),
                landing_times_s=[s[2] for s in swings])


def recovery_delay(times, values, start, stop, threshold, dwell=.5):
    first = None
    for t, value in zip(times, values):
        if not start <= t < stop:
            continue
        if value < threshold:
            if first is None:
                first = t
            if t-first >= dwell-1e-8:
                return float(first-start)
        else:
            first = None
    return None


def source_spec(robot):
    spec = mujoco.MjSpec.from_file(str(robot/'robot_walk.xml'))
    for geom in spec.geoms:
        geom.contype, geom.conaffinity = 0, 0
        if geom.name.endswith('_collision'):
            geom.contype = 1
            geom.condim = 1
        if geom.name in ('left_foot_collision', 'right_foot_collision'):
            geom.condim, geom.priority = 3, 1
            geom.friction[0] = 1.
    return spec


def make_model(source, kp, voltage, delay):
    robot = source/'src/mjlab_microduck/robot/xgoduck'
    # Use BAM's actual edit_spec: the XML's viewer-only position actuators
    # must become torque motors before writing BAM-computed torques to ctrl.
    cfg = EntityCfg(spec_fn=partial(source_spec, robot),
                    articulation=EntityArticulationInfoCfg(actuators=(BamActuatorCfg(
                        json_path=str(robot/'params/1910_m6.json'),
                        target_names_expr=(r'^(?!passive_).*',), kp_fw=kp,
                        vin_range=(7.4, 8.0), delay_min_lag=3, delay_max_lag=6),)))
    spec = cfg.build().spec
    spec.add_texture(name='replay_grid', type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[.64, .79, .87], rgb2=[.54, .69, .77], width=256, height=256)
    material = spec.add_material(name='replay_floor', texuniform=True, texrepeat=[5, 5])
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'replay_grid'
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                           size=[0, 0, .1], material='replay_floor')
    model = spec.compile()
    MujocoCfg(timestep=.005, iterations=10, ls_iterations=20).apply(model)
    if (not np.allclose(model.actuator_gainprm[:, 0], 1)
            or np.any(model.actuator_biastype != mujoco.mjtBias.mjBIAS_NONE)):
        raise ValueError('BAM ctrl must be torque, not XML companion-PD position')
    data = mujoco.MjData(model)
    motor = XgoBamCpuController(model, data, voltage, delay=delay, kp_fw=kp,
                               profile_path=robot/'params/1910_m6.json')
    return model, data, motor


def foot_vertices(model):
    result = []
    for name in ('left_foot_collision', 'right_foot_collision'):
        gid = model.geom(name).id
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError('Foot clearance requires mesh geoms')
        mesh = model.geom_dataid[gid]
        first = model.mesh_vertadr[mesh]
        result.append((gid, model.mesh_vert[first:first+model.mesh_vertnum[mesh]]))
    return result


def clearance(data, feet):
    return [float(np.min(v @ data.geom_xmat[g].reshape(3, 3)[2]
                         + data.geom_xpos[g, 2])) for g, v in feet]


def run_case(args, session, profile, scenario, seed, writer=None):
    kp, alpha, runtime_home = PROFILES[profile]
    name, requested, push = scenario
    model, data, motor = make_model(args.source, kp, args.voltage, args.delay_steps)
    metadata = session.get_modelmeta().custom_metadata_map
    names = metadata['joint_names'].split(',')
    if names != [model.joint(int(i)).name for i in motor.joint_ids]:
        raise ValueError('Upstream actuator order differs from ONNX metadata')
    home = np.asarray([float(x) for x in metadata['default_joint_pos'].split(',')], dtype=np.float32)
    if runtime_home:
        home = np.radians([0, -5, -23, 0, 23, 20, 20, 0, 0, 0, 5, 23, 0, -23]).astype(np.float32)
    feet = foot_vertices(model)
    rng = np.random.default_rng(seed)
    data.qpos[:7] = [0, 0, .15, 1, 0, 0, 0]
    data.qpos[motor.qids] = home + rng.uniform(-.003, .003, 14)
    mujoco.mj_forward(model, data)
    # Match actual HOME geometry; never spawn the feet penetrating the floor.
    data.qpos[2] += .001-min(clearance(data, feet))
    initial_z = float(data.qpos[2])
    mujoco.mj_forward(model, data)
    motor.reset(data.qpos)
    previous = np.zeros(14, dtype=np.float32)
    filtered = previous.copy()
    head_id, root_id = model.body('jaw_soft').id, model.body('trunk_base').id
    floor = model.geom('floor').id
    gyro_adr = model.sensor_adr[model.sensor('imu_ang_vel').id]
    history = deque(maxlen=30)
    rows = []
    first_fall = None
    path = args.output/f'{profile}_{name}_seed{seed}.csv'
    renderer = mujoco.Renderer(model, height=480, width=640) if writer else None
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = .65, -18, 130
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = [.5]*3
    model.vis.headlight.diffuse[:] = [.7]*3
    try:
        with path.open('w', newline='') as handle:
            trace = csv.writer(handle)
            trace.writerow(['t','vx_cmd','wz_cmd','vx','vy','wz','tilt_deg','z',
                            'x','y','yaw','left_mm','right_mm','left_contact','right_contact',
                            'head_force_N',*[f'q_{n}' for n in names],*[f'target_{n}' for n in names]])
            for step in range(round(args.seconds*50)):
                t = step*.02
                cmd = command_at(t, requested, args.seconds)
                rot = data.xmat[root_id].reshape(3, 3)
                history.append((data.sensordata[gyro_adr:gyro_adr+3].copy(),
                                -rot[2].copy(), data.qpos[motor.qids].copy(),
                                data.qvel[motor.vids].copy()))
                state = history[max(0, len(history)-1-args.observation_delay_steps)]
                obs = build_observation(*state, home, previous, cmd)
                raw = session.run(None, {session.get_inputs()[0].name: obs[None]})[0][0]
                if not np.isfinite(raw).all():
                    raise ValueError('Nonfinite ONNX action')
                previous = raw.copy()
                filtered = alpha*filtered+(1-alpha)*raw
                target = home+filtered
                motor.q_target = target.copy()
                force = push if 8. <= t < 8.2 else -push if 12. <= t < 12.2 else 0.
                for _ in range(4):
                    data.xfrc_applied[head_id] = 0
                    data.xfrc_applied[head_id, :3] = data.xmat[root_id].reshape(3, 3)[:, 0]*force
                    motor.update()
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                if not np.isfinite(data.qpos).all():
                    raise ValueError('Nonfinite physics')
                rot = data.xmat[root_id].reshape(3, 3)
                velocity = rot.T @ data.qvel[:3]
                wz = float(data.sensordata[gyro_adr+2])
                tilt = math.degrees(math.acos(float(np.clip(rot[2, 2], -1, 1))))
                if first_fall is None and (tilt > 60 or data.qpos[2] < .06):
                    first_fall = t+.02
                contacts = set()
                for contact in data.contact:
                    if floor in contact.geom and contact.dist <= .001:
                        contacts.update(set(map(int, contact.geom))-{floor})
                heights = np.asarray(clearance(data, feet))*1000
                row = [t+.02, cmd[0], cmd[2], *velocity[:2], wz, tilt,
                       data.qpos[2], *data.qpos[:2], math.atan2(rot[1, 0], rot[0, 0]),
                       *heights, *[int(g in contacts) for g, _ in feet], force]
                trace.writerow([*row, *data.qpos[motor.qids], *target])
                rows.append(row)
                if renderer and step % 2 == 0:
                    from PIL import Image, ImageDraw, ImageFont
                    camera.lookat[:] = data.qpos[:3]
                    camera.lookat[2] += .05
                    renderer.update_scene(data, camera=camera)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame, 'RGBA')
                    draw.rectangle((0, 0, 640, 88), fill=(12, 25, 35, 210))
                    text = (f'Luwu original ONNX | {profile} | {name} | t={t:.1f}s\n'
                            f'vx {cmd[0]:+.2f}/{velocity[0]:+.2f} m/s  wz {cmd[2]:+.2f}/{wz:+.2f} rad/s\n'
                            f'sole L/R {heights[0]:.1f}/{heights[1]:.1f} mm  tilt {tilt:.1f} deg\n'
                            f'head push {force:+.1f} N  | '+('FALL' if first_fall else 'SIMULATION'))
                    draw.multiline_text((8, 5), text, font=ImageFont.load_default(size=14), fill='white')
                    writer.append_data(np.asarray(frame))
    finally:
        if renderer:
            renderer.close()
    a = np.asarray(rows)
    valid = a[:, 0] < (first_fall if first_fall else np.inf)
    active = valid & (a[:, 0] >= 4) & (a[:, 0] < args.seconds-4)
    active_rows = a[active]
    results = dict(profile=profile, case=name, seed=seed, command=list(requested),
                   first_fall_s=first_fall, no_fall=first_fall is None,
                   home_rad=home.tolist(), initial_root_z_m=initial_z,
                   max_tilt_deg=float(a[:, 6].max()),
                   mass_kg=float(model.body_mass.sum()), active_prefall_seconds=len(active_rows)*.02,
                   trace=str(path), head_push_N=push,
                   evaluation_window_s=[4, args.seconds-4], kp_fw=kp, action_alpha=alpha)
    results['feet'] = [summarize_swings([(r[0], r[11+i]/1000, bool(r[13+i]))
                                       for r in active_rows]) for i in range(2)]
    if len(active_rows):
        results.update(mean_vx_m_s=float(active_rows[:, 3].mean()),
                       rms_vx_error_m_s=float(np.sqrt(np.mean((active_rows[:, 3]-requested[0])**2))),
                       mean_wz_rad_s=float(active_rows[:, 5].mean()),
                       rms_wz_error_rad_s=float(np.sqrt(np.mean((active_rows[:, 5]-requested[2])**2))))
    else:
        results.update(mean_vx_m_s=None, rms_vx_error_m_s=None, mean_wz_rad_s=None, rms_wz_error_rad_s=None)
    moving = a[valid & (a[:, 0] >= 2) & (a[:, 0] < args.seconds-4)]
    results['heading_change_deg'] = float(np.degrees(np.unwrap(moving[:, 10])[-1]-moving[0, 10])) if len(moving) else None
    results['cross_track_abs_max_m'] = float(np.max(np.abs(moving[:, 9]-moving[0, 9]))) if len(moving) else None
    tracking_error = np.maximum(np.abs(a[:, 3]-requested[0])/max(.02, abs(requested[0])*.25),
                                np.abs(a[:, 5]-requested[2])/max(.15, abs(requested[2])*.25))
    results['start_tracking_delay_s'] = recovery_delay(a[valid, 0], tracking_error[valid],
                                                       2, 8, 1.) if np.any(requested) else None
    stopped = np.maximum(np.linalg.norm(a[:, 3:5], axis=1)/.02, np.abs(a[:, 5])/.15)
    results['stop_settle_delay_s'] = recovery_delay(a[valid, 0], stopped[valid],
                                                    args.seconds-4, args.seconds, 1.)
    results['push_upright_recovery_s'] = [recovery_delay(a[valid, 0], a[valid, 6], t, t+3.8, 15)
                                         for t in (8.2, 12.2)] if push else []
    recovered = np.maximum(tracking_error, a[:, 6]/15.)
    results['push_tracking_recovery_s'] = [recovery_delay(a[valid, 0], recovered[valid], t, t+3.8, 1.)
                                          for t in (8.2, 12.2)] if push else []
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    p.add_argument('--cases', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--seconds', type=float, default=22.)
    p.add_argument('--voltage', type=float, default=7.4)
    p.add_argument('--delay-steps', type=int, default=4, choices=range(3, 11))
    p.add_argument('--observation-delay-steps', type=int, default=0, choices=range(5))
    p.add_argument('--video', type=Path)
    args = p.parse_args()
    if not 18 <= args.seconds <= 180:
        p.error('seconds must be 18..180 to include startup, both pushes, and stop')
    cases = [c for c in scenarios() if args.cases is None or c[0] in args.cases]
    if not cases or args.cases and set(args.cases)-{c[0] for c in cases}:
        p.error('unknown/empty case selection')
    args.output.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(args.policy), sess_options=options, providers=['CPUExecutionProvider'])
    meta = session.get_modelmeta().custom_metadata_map
    if session.get_inputs()[0].shape != [1, 61] or session.get_outputs()[0].shape != [1, 14]:
        raise ValueError('Expected 61->14 ONNX')
    if meta.get('action_scale') != '1.0' or meta.get('action_semantics', 'raw_home_delta') != 'raw_home_delta':
        raise ValueError('Not an upstream raw-action model')
    robot = args.source/'src/mjlab_microduck/robot/xgoduck'
    report = dict(policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                  source_commit=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
                  geometry_sha256=hashlib.sha256((robot/'robot_walk.xml').read_bytes()).hexdigest(),
                  bam_sha256=hashlib.sha256((robot/'params/1910_m6.json').read_bytes()).hexdigest(),
                  metadata=meta, voltage_V=args.voltage, motor_delay_ms=args.delay_steps*5,
                  observation_delay_ms=args.observation_delay_steps*20,
                  simulation_only=True, policy_modified=False, previous_action='raw_onnx_output',
                  policy_hz=50, physics_hz=200, seconds_per_case=args.seconds,
                  measurement='minimum collision mesh vertex height; complete swings >=40 ms',
                  limitations=['CPU MuJoCo is not training MJWarp', 'No Arduino IMU estimator or serial emulation',
                               'Upstream walk collision model has feet only; no physical head impact validation',
                               'Single fixed voltage/friction per run; not hardware qualification'], cases=[])
    writer = None
    if args.video:
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.video), fps=25)
    try:
        for profile in args.profiles:
            for seed in args.seeds:
                for case in cases:
                    row = run_case(args, session, profile, case, seed, writer)
                    report['cases'].append(row)
                    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                    print(json.dumps({k: row[k] for k in ('profile','case','seed','no_fall','mean_vx_m_s','mean_wz_rad_s')}), flush=True)
    finally:
        if writer:
            writer.close()


if __name__ == '__main__':
    main()
