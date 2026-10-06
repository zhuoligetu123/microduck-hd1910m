#!/usr/bin/env python3
"""CPU MuJoCo/ONNX step/sway acceptance and viewer. No hardware/UDP access.

Contract: normalized ONNX 61 -> 14, action scale 1; phase period from metadata.
Command slot 48:51 is phase [cos,sin,0], not a commanded walking velocity.
Uses the same stock BAM XL330 physics as training, not an HD1910 model.
"""
import argparse
import contextlib
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from infer_policy import (PolicyInference, load_bam_model, load_mujoco_with_bam,
                          BAM_KP_FW, DEFAULT_POSE)

ROOT = Path(__file__).resolve().parents[1]
FALL_ANGLE = math.radians(70.)  # Original velocity task bad_orientation threshold.


def mouth_target(elapsed, period=2.):
    """Independent mouth channel: zero closed, positive opening up to 30 degrees."""
    opening = .5 * (1. - math.cos(2*math.pi*elapsed/period))
    return math.radians(30.*opening)


class MouthPreview:
    """Visual-only jaw hinge: the stock training mesh has no mouth DOF.

    Approximate hinge at the mouth motor; no mass, contact, policy or joint changes.
    This animation is not a simulated mouth servo or hardware motion evidence.
    """

    def __init__(self, model):
        mesh = model.mesh('jaw').id
        self.ids = np.flatnonzero((model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
                                 & (model.geom_dataid == mesh))
        if not len(self.ids) or np.any(model.geom_contype[self.ids]) or np.any(model.geom_conaffinity[self.ids]):
            raise ValueError('mouth preview requires visual-only jaw geometry')
        self.model = model
        self.positions = model.geom_pos[self.ids].copy()
        self.quaternions = model.geom_quat[self.ids].copy()
        self.pivot = np.array([.003, 0., -.018])

    def update(self, angle):
        # The exported mesh depicts the calibrated zero (closed) mouth.
        delta = angle
        rotation = np.array([math.cos(delta/2), 0., math.sin(delta/2), 0.])
        for i, gid in enumerate(self.ids):
            offset = np.empty(3)
            mujoco.mju_rotVecQuat(offset, self.positions[i]-self.pivot, rotation)
            self.model.geom_pos[gid] = self.pivot+offset
            mujoco.mju_mulQuat(self.model.geom_quat[gid], rotation, self.quaternions[i])


def transition_action(action, elapsed, duration):
    """One-time HOME-to-policy blend, not a per-cycle rate limiter."""
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('blend duration must be finite and nonnegative')
    if duration == 0 or elapsed >= duration:
        return action.copy()
    u = min(1., max(0., elapsed / duration))
    weight = u**3 * (10. + u * (-15. + 6.*u))
    return action * weight


class StepPolicy(PolicyInference):
    """Match the training actor's one-control-step joint velocity delay."""

    previous_velocity = None

    def get_observations(self):
        self.last_observation = super().get_observations()
        return self.last_observation

    def get_joint_vel(self):
        current = super().get_joint_vel()
        previous = current if self.previous_velocity is None else self.previous_velocity
        self.previous_velocity = current
        return previous


def run(args):
    model_sha = hashlib.sha256(args.policy.read_bytes()).hexdigest()
    scene = ROOT/'src/mjlab_microduck/robot/microduck/scene_walk.xml'
    motor_profile = 'xl330_bam_m6'
    calibration = getattr(args,'hd1910_calibration',None)
    if calibration:
        from mjlab_microduck.robot.hd1910 import MotorFit, load_cpu_model
        fit = MotorFit.load(calibration)
        model,data,bam,_ = load_cpu_model(scene,fit)
        args.actuator_lag = fit.delay_steps
        motor_profile = 'hd1910_mode4_pd_dc_approximation'
    else:
        model, data, bam, _ = load_mujoco_with_bam(str(scene),
            load_bam_model(BAM_KP_FW, 7.4, 0), .005, .1, 6.)
    policy = StepPolicy(model, data, walking_onnx_path=str(args.policy),
                             bam_ctrl=bam, new_cmd_obs=True, use_projected_gravity=True)
    if hashlib.sha256(args.policy.read_bytes()).hexdigest() != model_sha:
        raise RuntimeError('policy changed while loading; evaluate an immutable export')
    if policy.ort_session.get_inputs()[0].shape != [1,61] or policy.ort_session.get_outputs()[0].shape != [1,14]:
        raise ValueError('expected normalized ONNX [1,61] -> [1,14]')
    metadata = policy.ort_session.get_modelmeta().custom_metadata_map
    sway = metadata.get('task_id') in ('Mjlab-SwayInPlace-Flat-MicroDuck', 'Mjlab-SwayHead-Flat-MicroDuck')
    head_sway = metadata.get('task_id') == 'Mjlab-SwayHead-Flat-MicroDuck'
    head_yaw_amplitude = float(metadata.get('head_sway_yaw_amplitude_rad', '0'))
    if head_sway and (not math.isfinite(head_yaw_amplitude) or head_yaw_amplitude <= 0
                      or metadata.get('head_sway_reference') != 'camera_forward_in_trunk_frame'):
        raise ValueError('head sway requires positive amplitude and the trunk-frame camera convention')
    phase_period = float(metadata.get('phase_period_s', '1'))
    sway_amplitude = float(metadata.get('sway_amplitude_rad', '0'))
    trunk_height_target = float(metadata.get('trunk_height_m', '0'))
    if not math.isfinite(phase_period) or phase_period <= 0:
        raise ValueError('phase period must be finite and positive')
    if sway and (not math.isfinite(sway_amplitude) or sway_amplitude <= 0):
        raise ValueError('sway amplitude must be finite and positive')
    head_command = np.array([float(v) for v in metadata.get('head_command', '0,0,0,0').split(',')])
    if head_command.shape != (4,) or not np.isfinite(head_command).all():
        raise ValueError('head command metadata must have 4 finite values')
    mouth_enabled = args.mouth_preview or metadata.get('mouth_control') == 'independent_periodic_preview'
    mouth = MouthPreview(model) if mouth_enabled else None
    head_indices = [model.jnt_qposadr[model.joint(name).id] for name in
                    ('neck_pitch','head_pitch','head_yaw','head_roll')]
    head_target = DEFAULT_POSE[5:9] + head_command
    hz = int(float(metadata.get('control_hz', args.hz)))
    if hz not in (20, 50):
        raise ValueError('unsupported policy frequency')
    dt = 1. / hz
    substeps = round(dt / model.opt.timestep)
    episodes = []
    trajectory = []
    viewer_ctx = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) if args.viewer else contextlib.nullcontext(None)
    rng = np.random.default_rng(args.seed)
    with contextlib.ExitStack() as resources:
        trace = None
        if getattr(args, 'trace', None):
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            trace = resources.enter_context(args.trace.open('w'))
        viewer = resources.enter_context(viewer_ctx)
        renderer = writer = None
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation, camera.azimuth = .65, -15, 135
        video_stride = max(1, round(hz/25))
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            renderer = resources.enter_context(mujoco.Renderer(model, height=480, width=640))
            writer = resources.enter_context(imageio.get_writer(str(args.video), fps=hz/video_stride))
        if viewer:
            viewer.cam.distance = .65
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 135
        for episode in range(args.episodes):
            mujoco.mj_resetData(model, data)
            data.qpos[:7] = [0,0,.125,1,0,0,0]
            data.qpos[policy.joint_qpos_indices] = DEFAULT_POSE + rng.uniform(-.005,.005,14)
            bam.reset(data.qpos)
            policy.last_action[:] = 0
            policy.previous_velocity = None
            policy.set_position_targets(policy.default_pose)
            mujoco.mj_forward(model, data)
            max_drift = max_tilt = max_yaw = 0.
            lifts = np.zeros(2, dtype=int)
            last_air = np.zeros(2, dtype=bool)
            max_height = np.zeros(2)
            max_action = max_target_jump = max_target_limit_violation = 0.
            first_target_jump = max_raw_target_jump = 0.
            max_actual_joint_violation = max_actual_joint_speed = 0.
            head_pitch_total = head_error_total = 0.
            head_yaw_min, head_yaw_max = math.inf, -math.inf
            head_yaw_error_squared = 0.
            head_samples = 0
            roll_min, roll_max = math.inf, -math.inf
            roll_error_squared = 0.
            feet_near_ground = 0
            trunk_height_min = math.inf
            trunk_height_total = 0.
            mouth_min, mouth_max = math.inf, -math.inf
            previous_raw_target = policy.default_pose.copy()
            previous_target = policy.default_pose.copy()
            joint_limits = model.jnt_range[model.actuator_trnid[:, 0]]
            height_error = 0.
            contact_match = 0.
            both_air = 0
            steps = 0
            fallen = False
            delayed_targets = None
            limit = round(args.seconds * hz) if args.seconds else 10**9
            for step in range(limit):
                if viewer and not viewer.is_running():
                    break
                started = time.monotonic()
                phase = (step * dt / phase_period) % 1.0
                policy.command[:] = 0.
                policy.command[:3] = [math.cos(2*math.pi*phase), math.sin(2*math.pi*phase), 0]
                policy.command[3:7] = head_command
                if mouth:
                    angle = mouth_target(step*dt)
                    mouth_min, mouth_max = min(mouth_min,angle), max(mouth_max,angle)
                    mouth.update(angle)
                action = policy.infer()
                if not np.isfinite(action).all():
                    raise ValueError('nonfinite policy output')
                raw_target = policy.default_pose + action * policy.action_scale
                max_raw_target_jump = max(max_raw_target_jump,
                    float(np.abs(raw_target-previous_raw_target).max()))
                previous_raw_target = raw_target.copy()
                applied_action = transition_action(action, step*dt, args.blend_seconds)
                # infer() retains raw policy output for the next observation,
                # independent of the one-time actuator entry blend.
                target_positions = policy.default_pose + applied_action * policy.action_scale
                if step == 0:
                    first_target_jump = float(np.abs(target_positions-previous_target).max())
                max_action = max(max_action, float(np.abs(action).max()))
                max_target_jump = max(max_target_jump, float(np.abs(target_positions-previous_target).max()))
                max_target_limit_violation = max(max_target_limit_violation,
                    float(np.maximum(joint_limits[:,0]-target_positions, target_positions-joint_limits[:,1]).max()))
                previous_target = target_positions.copy()
                if delayed_targets is None:
                    delayed_targets = deque([target_positions.copy() for _ in range(args.actuator_lag)])
                for _ in range(substeps):
                    delayed_targets.append(target_positions)
                    policy.set_position_targets(delayed_targets.popleft())
                    bam.update()
                    mujoco.mj_step(model, data)
                q = data.qpos[3:7]
                joint_positions = data.qpos[policy.joint_qpos_indices]
                if trace is not None:
                    trace.write(json.dumps(dict(episode=episode, step=step, t=step*dt,
                        observation=policy.last_observation.tolist(),
                        raw_action=action.tolist(), target=target_positions.tolist(),
                        positions_after=joint_positions.tolist(),
                        physics='stock XL330 BAM, not HD1910'), allow_nan=False)+'\n')
                max_actual_joint_violation = max(max_actual_joint_violation,
                    float(np.maximum(joint_limits[:,0]-joint_positions,
                                     joint_positions-joint_limits[:,1]).max()))
                max_actual_joint_speed = max(max_actual_joint_speed,
                    float(np.abs(data.qvel[model.jnt_dofadr[model.actuator_trnid[:,0]]]).max()))
                tilt = math.acos(np.clip(1-2*(q[1]**2+q[2]**2), -1,1))
                yaw = math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
                drift = float(np.linalg.norm(data.qpos[:2]))
                max_drift = max(max_drift, drift)
                max_tilt = max(max_tilt, tilt)
                max_yaw = max(max_yaw, abs(yaw))
                if step*dt >= 2.:
                    if args.trajectory and episode == 0:
                        trajectory.append(dict(t=round(step*dt-2., 8),
                            positions=joint_positions.tolist(),
                            mouth_rad=mouth_target(step*dt) if mouth_enabled else 0.))
                    trunk_height_min = min(trunk_height_min, float(data.qpos[2]))
                    trunk_height_total += float(data.qpos[2])
                    roll = math.atan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]**2+q[2]**2))
                    roll_min, roll_max = min(roll_min, roll), max(roll_max, roll)
                    roll_error_squared += (roll-sway_amplitude*math.sin(2*math.pi*phase))**2
                    forward = data.site('head_camera').xmat.reshape(3,3)[:,0]
                    forward_b = data.xmat[policy.trunk_base_id].reshape(3,3).T @ forward
                    head_yaw = math.atan2(forward_b[1],forward_b[0])
                    head_yaw_min, head_yaw_max = min(head_yaw_min,head_yaw), max(head_yaw_max,head_yaw)
                    head_yaw_target = head_yaw_amplitude*math.sin(2*math.pi*phase)
                    head_yaw_error = math.atan2(math.sin(head_yaw-head_yaw_target),math.cos(head_yaw-head_yaw_target))
                    head_yaw_error_squared += head_yaw_error**2
                    head_pitch_total += math.atan2(forward[2], math.hypot(forward[0],forward[1]))
                    head_error_total += float(np.abs(data.qpos[head_indices]-head_target).mean())
                    head_samples += 1
                height = np.array([data.site(name).xpos[2] for name in ('left_foot','right_foot')])
                air = height > .008
                if step*dt >= 2.:
                    feet_near_ground += int(not air.any())
                target = np.zeros(2) if sway else .02 * np.maximum(
                    [math.sin(2*math.pi*phase), -math.sin(2*math.pi*phase)], 0)
                height_error += float(np.abs(height-target).mean())
                contact_match += float((air == (target > .008)).mean())
                both_air += int(air.all())
                lifts += (air & ~last_air)
                last_air = air
                max_height = np.maximum(max_height,height)
                steps += 1
                fallen = tilt > FALL_ANGLE or not np.isfinite(data.qpos).all()
                if writer and step % video_stride == 0:
                    camera.lookat[:] = data.xpos[policy.trunk_base_id] + [0., 0., .06]
                    renderer.update_scene(data, camera=camera)
                    writer.append_data(renderer.render())
                if viewer:
                    viewer.cam.lookat[:] = data.xpos[policy.trunk_base_id] + [0., 0., .06]
                    viewer.sync()
                    time.sleep(max(0., dt-(time.monotonic()-started)))
                if fallen:
                    break
            completed = steps >= limit
            gait_passed = (completed and not fallen and max_drift < .08 and max_yaw < .25
                      and min(lifts) >= 4 and height_error / max(steps,1) < .015
                      and contact_match / max(steps,1) >= .6 and both_air / max(steps,1) < .1)
            head_passed = head_samples > 0 and head_pitch_total/head_samples >= 0.
            head_yaw_rmse = math.sqrt(head_yaw_error_squared/max(head_samples,1))
            head_sway_passed = (head_samples > 0 and head_yaw_min < -.8*head_yaw_amplitude
                                and head_yaw_max > .8*head_yaw_amplitude
                                and head_yaw_rmse < math.radians(5.)) if head_sway else True
            roll_rmse = math.sqrt(roll_error_squared/max(head_samples,1))
            sway_passed = (completed and not fallen and head_samples > 0
                           and max_drift < .08 and max_yaw < .25
                           and roll_min < -(.8 if head_sway else .5)*sway_amplitude
                           and roll_max > (.8 if head_sway else .5)*sway_amplitude
                           and roll_rmse < math.radians(4.)
                           and feet_near_ground/head_samples >= .9)
            height_passed = (trunk_height_target == 0 or (head_samples > 0
                             and trunk_height_total/head_samples >= trunk_height_target-.005
                             and trunk_height_min >= trunk_height_target-.010))
            if sway:
                sway_passed = sway_passed and height_passed
            passed = (sway_passed if sway else gait_passed) and (head_passed or 'head_command' not in metadata) and head_sway_passed
            episodes.append(dict(episode=episode,seconds=steps*dt,passed=bool(passed),
                gait_passed=bool(gait_passed) if not sway else None,head_up_passed=bool(head_passed),
                sway_passed=bool(sway_passed) if sway else None,
                head_sway_passed=bool(head_sway_passed) if head_sway else None,
                head_yaw_range_deg=[math.degrees(head_yaw_min),math.degrees(head_yaw_max)] if head_samples else None,
                head_yaw_rmse_deg=math.degrees(head_yaw_rmse) if head_sway else None,
                roll_range_deg=[math.degrees(roll_min),math.degrees(roll_max)] if head_samples else None,
                roll_rmse_deg=math.degrees(roll_rmse),
                both_feet_below_8mm_fraction=feet_near_ground/max(head_samples,1),
                min_trunk_height_m=trunk_height_min if head_samples else None,
                mean_trunk_height_m=trunk_height_total/max(head_samples,1),
                trunk_height_passed=bool(height_passed) if trunk_height_target else None,
                mean_head_pitch_deg=math.degrees(head_pitch_total/head_samples) if head_samples else None,
                mean_head_joint_error_deg=math.degrees(head_error_total/head_samples) if head_samples else None,
                mouth_command_range_deg=[math.degrees(mouth_min),math.degrees(mouth_max)] if mouth and steps else None,
                fallen=bool(fallen),max_drift_m=max_drift,max_tilt_deg=math.degrees(max_tilt),
                max_yaw_deg=math.degrees(max_yaw),foot_lifts=lifts.tolist(),max_foot_height_m=max_height.tolist(),
                mean_height_error_m=height_error/max(steps,1),phase_contact_match=contact_match/max(steps,1),
                both_feet_air_fraction=both_air/max(steps,1),max_action_rad=max_action,
                max_target_jump_rad=max_target_jump,max_target_limit_violation_rad=max_target_limit_violation,
                first_target_jump_rad=first_target_jump,max_raw_target_jump_rad=max_raw_target_jump,
                max_actual_joint_limit_violation_rad=max_actual_joint_violation,
                max_actual_joint_speed_rad_s=max_actual_joint_speed,
                transition_completed=bool(steps > 0 and (steps-1)*dt >= args.blend_seconds)))
            print(json.dumps(episodes[-1]),flush=True)
            if viewer and not viewer.is_running():
                break
    result=dict(task='sway_in_place' if sway else 'step_in_place',model_sha256=model_sha,
        control_hz=hz,phase_period_s=phase_period,command_semantics='phase_cos_sin_zero',
        sway_amplitude_rad=sway_amplitude,
        head_sway_yaw_amplitude_rad=head_yaw_amplitude,
        trunk_height_target_m=trunk_height_target,
        actuator_delay_ms=args.actuator_lag*model.opt.timestep*1000,
        motor_profile=motor_profile,
        motor_calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest() if calibration else None,
        blend_seconds=args.blend_seconds,target_rate_limiter=False,
        action_history='raw_policy_output',
        head_command=head_command.tolist(),mouth_visual_only=mouth_enabled,
        acceptance_version='xl330_sway_head_v1' if head_sway else ('xl330_sway_v1' if sway else 'xl330_gait_v2'),fall_angle_deg=math.degrees(FALL_ANGLE),
        target_jump_and_overshoot_diagnostic_only=True,
        hardware_tested=False,hardware_approved=False,physics='stock XL330 BAM, not HD1910',
        passed=len(episodes)==args.episodes and all(e['passed'] for e in episodes),episodes=episodes)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2))
    if args.trajectory:
        args.trajectory.parent.mkdir(parents=True,exist_ok=True)
        args.trajectory.write_text(json.dumps(dict(
            schema=1,source='mujoco_measured_joint_positions',
            model_sha256=model_sha,control_hz=hz,sim_validation_passed=result['passed'],
            joint_names=[model.joint(int(j)).name for j in model.actuator_trnid[:,0]],
            frames=trajectory,hardware_approved=False)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--report',type=Path)
    p.add_argument('--trace',type=Path,help='optional per-frame observations, raw actions and actual simulated positions')
    p.add_argument('--hd1910-calibration',type=Path,
                   help='identified physical mode-4 PD parameters; never XL330 firmware gain numbers')
    p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--hz',type=int,default=50,choices=(20,50),
                   help='fallback for raw checkpoints; exported ONNX metadata takes precedence')
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--video',type=Path,help='optional 640x480 MuJoCo MP4 recording')
    p.add_argument('--trajectory',type=Path,help='actual simulated joint positions for a separate supported replay bench')
    p.add_argument('--mouth-preview',action='store_true',help='independent visual-only mouth animation, no actuator')
    p.add_argument('--blend-seconds',type=float,default=0.,
                   help='simulation-only HOME-to-RL blend; 0 preserves the baseline, no rate limiter')
    p.add_argument('--actuator-lag',type=int,default=4,choices=range(3,7),
                   help='5ms substeps, matches training actuator delay range 3..6')
    args=p.parse_args()
    if args.seconds<0 or args.episodes<1 or (not args.seconds and not args.viewer):
        p.error('seconds must be positive (0 allowed only for viewer); episodes >= 1')
    if not math.isfinite(args.blend_seconds) or args.blend_seconds < 0:
        p.error('blend-seconds must be finite and nonnegative')
    return 0 if run(args)['passed'] or args.viewer else 2


if __name__=='__main__':
    raise SystemExit(main())
