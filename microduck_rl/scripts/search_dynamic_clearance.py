#!/usr/bin/env python3
"""Simulation-only guided reference search, NOT a modified deployment policy.

Search symmetric target shaping around a frozen feedback policy. Free base,
collision contacts, BAM, observation age and target slew remain active. A
failed finite search does not establish dynamic impossibility.
"""
import argparse
import csv
import hashlib
import json
import math
import multiprocessing
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from replay_hd1910 import (ReplayPolicy, load_replay_model, step_control_period,
                          validate_metadata, planar_pose, straight_path_metrics)
from action_request_probe import ActionRequestProbe


IDENTITY = np.array([1., 1., 1., 1., 0., 0., 0., 0.])
BOUNDS = [(.4, 2.), (.4, 2.), (.4, 2.5), (.4, 2.),
          (-.2, .2), (-.35, .35), (-.2, .2), (0., .75)]
ENGINE = None


def shape_target(action, params, previous, home, limits):
    gains = np.ones(14)
    offset = np.zeros(14)
    for left, right, gain in zip((1, 2, 3, 4), (10, 11, 12, 13), params[:4]):
        gains[left] = gains[right] = gain
    for left, right, bias in zip((2, 3, 4), (11, 12, 13), params[4:7]):
        offset[left], offset[right] = bias, -bias
    desired = home + action * gains + offset
    desired = np.clip(desired, limits[:, 0], limits[:, 1])
    desired = params[7] * previous + (1 - params[7]) * desired
    return previous + np.clip(desired - previous, -.1, .1)


class DynamicProbe:
    def __init__(self, policy_path, seconds, target_mm):
        self.path = Path(policy_path)
        self.seconds, self.target_mm = seconds, target_mm
        self.model, self.data, self.motor = load_replay_model(7.4, bam_reference=True, ground_contact=True)
        self.policy = ReplayPolicy(self.model, self.data, walking_onnx_path=str(self.path),
                                   bam_ctrl=self.motor, new_cmd_obs=True, use_projected_gravity=True)
        metadata = self.policy.ort_session.get_modelmeta().custom_metadata_map
        self.profile_hash = validate_metadata(metadata,
            [self.model.joint(int(j)).name for j in self.motor.joint_ids], bam_reference=True)
        if metadata.get('joint_snapshot_training') != 'coherent_pos_vel_delay_v1':
            raise ValueError('reference search requires coherent joint snapshots')
        self.policy.ort_session = ActionRequestProbe(self.policy.ort_session, self.path)
        self.policy.coherent_joint_snapshot = True
        self.policy.set_joint_observation_delay(1)
        self.policy.set_imu_observation_delay(10)
        self.policy.set_vel_cmd(.1, 0., 0.)
        self.floor = self.model.geom('floor').id
        self.head = self.model.body('jaw_soft').id
        self.feet = []
        for name in ('left_foot_collision', 'right_foot_collision'):
            gid = self.model.geom(name).id
            mid = self.model.geom_dataid[gid]
            begin = self.model.mesh_vertadr[mid]
            self.feet.append((gid, self.model.mesh_vert[begin:begin+self.model.mesh_vertnum[mid]].copy()))
        self.limits = np.column_stack((self.policy.default_pose + np.asarray(json.loads(metadata['action_delta_low'])),
                                       self.policy.default_pose + np.asarray(json.loads(metadata['action_delta_high']))))

    def run(self, params, seconds=None, trace_path=None, video_path=None, seed=42):
        model, data, motor, policy = self.model, self.data, self.motor, self.policy
        duration = self.seconds if seconds is None else seconds
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = [0., 0., .125, 1., 0., 0., 0.]
        data.qpos[motor.qids] = policy.default_pose + np.random.default_rng(seed).uniform(-.005, .005, 14)
        motor.reset(data.qpos)
        policy.last_action[:] = 0.
        policy.reset_joint_observation_history()
        policy.set_head_command(0.)
        mujoco.mj_forward(model, data)
        previous = policy.default_pose.copy()
        peaks, active, seen, peak, supported = [[], []], [False]*2, [False]*2, np.zeros(2), [False]*2
        landings, poses, velocities, errors, tilts = [], [planar_pose(data.qpos)], [], [], []
        first_failure = None
        failure_reasons = []
        max_step, max_torque = 0., 0.
        rows = []
        renderer = writer = None
        if video_path:
            import imageio.v2 as imageio
            model.vis.headlight.active = 1
            model.vis.headlight.ambient[:] = [.35]*3
            model.vis.headlight.diffuse[:] = [.7]*3
            model.geom_rgba[self.floor] = [.64, .79, .87, 1.]
            renderer = mujoco.Renderer(model, height=480, width=640)
            writer = imageio.get_writer(str(video_path), fps=25)
            camera = mujoco.MjvCamera()
            camera.distance, camera.elevation, camera.azimuth = .75, -20, 135
        try:
            for step in range(round(duration*50)):
                bounded = policy.infer()
                latent, _ = policy.ort_session.samples.pop()
                # Shape before the one range/slew projection. Shaping the
                # already-slewed output feeds scaled history back recursively.
                target = shape_target(latent[0], params, previous, policy.default_pose, self.limits)
                if np.array_equal(params, IDENTITY):
                    np.testing.assert_allclose(target-policy.default_pose, bounded, atol=1e-6, rtol=1e-6)
                max_step = max(max_step, float(np.max(np.abs(target-previous))))
                previous = target.copy()
                # The next observation must describe the target actually sent.
                policy.last_action = (target-policy.default_pose).astype(np.float32)
                policy.set_position_targets(target)
                step_control_period(model, data, motor, policy)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError('nonfinite physics')
                max_torque = max(max_torque, float(np.abs(data.qfrc_actuator).max()))
                gravity = policy.get_projected_gravity()
                tilt = math.degrees(math.acos(float(np.clip(-gravity[2], -1, 1))))
                touching = set()
                head_contact = False
                for contact in data.contact:
                    if self.floor in contact.geom and contact.dist <= .001:
                        other = int(contact.geom[1] if contact.geom[0] == self.floor else contact.geom[0])
                        touching.add(other)
                        head_contact |= model.geom_bodyid[other] == self.head and contact.dist <= 0.
                clearances = [float((vertices @ data.geom_xmat[gid].reshape(3, 3)[2]
                                     + data.geom_xpos[gid, 2]).min()) for gid, vertices in self.feet]
                contact = [gid in touching for gid, _ in self.feet]
                if tilt > 60 or data.qpos[2] < .06 or head_contact:
                    first_failure = (step+1)/50
                    failure_reasons = [name for name, failed in (
                        ('tilt_over_60_deg', tilt > 60), ('trunk_below_60_mm', data.qpos[2] < .06),
                        ('head_floor_contact', head_contact)) if failed]
                    break
                poses.append(planar_pose(data.qpos))
                if step >= 50:
                    velocities.append(policy.quat_rotate_inverse(data.qpos[3:7], data.qvel[:3]).copy())
                    errors.append(float(np.mean((target-data.qpos[motor.qids])**2)))
                    tilts.append(tilt)
                    landed_now = []
                    for foot in range(2):
                        if not contact[foot]:
                            active[foot] = True
                            if contact[1-foot] and tilt < 30:
                                peak[foot] = max(peak[foot], clearances[foot])
                                supported[foot] = True
                        elif active[foot]:
                            if seen[foot] and supported[foot]:
                                peaks[foot].append(peak[foot]*1000)
                                landed_now.append(foot)
                            active[foot], supported[foot], peak[foot] = False, False, 0.
                        seen[foot] |= contact[foot]
                    # Simultaneous landings are not alternating steps.
                    if landed_now:
                        landings.append(landed_now[0] if len(landed_now) == 1 else -1)
                if trace_path:
                    rows.append([(step+1)/50, *data.qpos[:3], tilt, *clearances, *map(int, contact),
                                 *data.qpos[motor.qids], *target, *data.qpos[3:7],
                                 *data.qvel[:6], *data.qvel[motor.vids], *data.qfrc_actuator[motor.vids]])
                if writer and step % 2 == 0:
                    camera.lookat[:] = data.qpos[:3]
                    renderer.update_scene(data, camera)
                    writer.append_data(renderer.render())
        finally:
            if writer:
                writer.close()
                renderer.close()
        medians = [float(np.median(p)) if p else 0. for p in peaks]
        fraction = [float(np.mean(np.asarray(p) >= self.target_mm)) if p else 0. for p in peaks]
        alternation = (sum(a >= 0 and b >= 0 and a != b for a, b in zip(landings, landings[1:]))
                       / (len(landings)-1)) if len(landings) > 1 else 0.
        path = straight_path_metrics(poses)
        mean_v = np.mean(velocities, axis=0).tolist() if velocities else [0.]*3
        survival = (first_failure or duration)/duration
        quality = min(min(medians)/self.target_mm, 1.)
        score = (10*survival + 3*quality + min(min(map(len, peaks))/5, 1.)
                 + alternation - 2*abs(mean_v[0]-.1)/.1
                 - min(path['max_heading_error_deg']/45, 3.) if path else -100.)
        if first_failure is not None:
            score -= 20.
        result = dict(params=np.asarray(params).tolist(), score=score, seed=seed,
                      seconds=duration, first_failure_s=first_failure, failure_reasons=failure_reasons,
                      supported_swing_count=[len(p) for p in peaks],
                      supported_swing_peak_median_mm=medians,
                      supported_swing_over_target_fraction=fraction,
                      supported_swing_peaks_mm=peaks, alternation_fraction=alternation,
                      mean_body_velocity=mean_v, path=path, max_target_step_rad=max_step,
                      max_actuator_torque_nm=max_torque,
                      mean_tilt_deg=float(np.mean(tilts)) if tilts else None,
                      joint_tracking_rms_rad=float(np.sqrt(np.mean(errors))) if errors else None,
                      dynamic_stage_passed=bool(first_failure is None and duration >= 20
                          and min(map(len, peaks)) >= 10 and min(medians) >= self.target_mm
                          and alternation >= .9 and abs(mean_v[0]-.1) < .05
                          and path['max_heading_error_deg'] <= 15 and path['max_cross_track_m'] <= .15))
        if trace_path:
            with Path(trace_path).open('w') as stream:
                csv.writer(stream).writerows([['time', 'x', 'y', 'z', 'tilt', 'left_clearance', 'right_clearance',
                    'left_contact', 'right_contact', *['q'+str(i) for i in range(14)],
                    *['target'+str(i) for i in range(14)], 'qw', 'qx', 'qy', 'qz',
                    *['root_vel'+str(i) for i in range(6)], *['dq'+str(i) for i in range(14)],
                    *['torque'+str(i) for i in range(14)]], *rows])
        return result


def initialize(policy, seconds, target):
    global ENGINE
    ENGINE = DynamicProbe(policy, seconds, target)


def objective(params):
    return -ENGINE.run(params)['score']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-mm', type=float, choices=(12., 15., 20., 25.), default=12.)
    parser.add_argument('--generations', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=6.)
    args = parser.parse_args()
    if not 1 <= args.generations <= 20 or not 1 <= args.workers <= 8 or not 4 <= args.seconds <= 20:
        parser.error('bounded search: 1..20 generations, 1..8 workers, 4..20 seconds')
    args.output.mkdir(parents=True, exist_ok=False)
    probe = DynamicProbe(args.policy, args.seconds, args.target_mm)
    baseline = probe.run(IDENTITY, seconds=20)
    (args.output/'baseline.json').write_text(json.dumps(baseline, indent=2)+'\n')
    rng = np.random.default_rng(2026)
    population = np.clip(IDENTITY+rng.normal(size=(32, 8))*[.2, .2, .3, .2, .05, .1, .05, .1],
                         np.array(BOUNDS)[:, 0], np.array(BOUNDS)[:, 1])
    population[0] = IDENTITY
    with multiprocessing.get_context('spawn').Pool(args.workers, initialize,
            (str(args.policy.resolve()), args.seconds, args.target_mm)) as pool:
        result = differential_evolution(objective, BOUNDS, maxiter=args.generations,
            init=population, seed=2026, workers=pool.map, updating='deferred', polish=False, disp=True)
    validations = [probe.run(result.x, seconds=20, seed=seed,
        trace_path=args.output/f'candidate_{seed}.csv',
        video_path=args.output/'reference_preview.mp4' if seed == 42 else None) for seed in (42, 123)]
    report = dict(method='symmetric target-shaping reference search, closed-loop frozen actor',
        reference_version=2, shaping_domain='latent_before_range_and_slew',
        target_mm=args.target_mm, policy=str(args.policy.resolve()),
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        bam_sha256=probe.profile_hash, kp=probe.motor.kp_fw, voltage=7.4, physics_hz=200,
        joint_age_ms=20, imu_age_ms=10, actuator_delay_ms=20, target_step_limit_rad=.1,
        evaluations=result.nfev, baseline=baseline, candidates=validations,
        dynamic_stage_passed=all(r['dynamic_stage_passed'] for r in validations),
        reference_only=True, original_policy_changed=False, deployment_ready=False, hardware_tested=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('baseline', 'candidates')}, indent=2))


if __name__ == '__main__':
    main()
