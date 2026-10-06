#!/usr/bin/env python3
"""Bounded CPU MuJoCo replay of the reference velocity task. No hardware I/O.

50 Hz policy, 200 Hz physics, configurable actuator and sensor delay.
Coherent joint snapshots follow policy metadata; legacy velocity lag can be
reproduced explicitly for diagnostics.
No extra action low-pass or rate limiter. Falling is reported, not hidden.
"""
import argparse
from collections import deque
import contextlib
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import mjlab  # Complete task auto-discovery before importing local actuators.
from infer_policy import PolicyInference, DEFAULT_POSE
from mjlab_microduck.actuator.reference_hd1910 import load_cpu_model, PROFILE_PATH
from mjlab_microduck.tasks.microduck_hd1910_env_cfg import make_reference_hd1910_velocity_env_cfg

ROOT=Path(__file__).resolve().parents[1]

# Local screening for repeated near-limit reversals, not a hardware rating.
MAX_SATURATED_REVERSAL_FRACTION = .25


def straight_path_metrics(samples):
    """Ground-truth poses from segment start, not integrated body gyro rates."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        return None
    yaw = np.unwrap(values[:, 2]) - values[0, 2]
    delta = values[:, :2] - values[0, :2]
    lateral = delta @ np.array([-math.sin(values[0, 2]), math.cos(values[0, 2])])
    return dict(final_heading_error_deg=float(np.degrees(yaw[-1])),
                max_heading_error_deg=float(np.degrees(np.abs(yaw).max())),
                max_cross_track_m=float(np.abs(lateral).max()))


def planar_pose(qpos):
    w, x, y, z = qpos[3:7]
    return [float(qpos[0]), float(qpos[1]),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))]


def posture_metrics(samples):
    """Signed pitch and collision-mesh clearance; flat floor at world z=0."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or not len(values) or not np.isfinite(values).all():
        return dict(posture_metrics_valid=False)
    pitch, height = values[:, 0], values[:, 1]
    return dict(posture_metrics_valid=True, mean_pitch_deg=float(pitch.mean()),
                pitch_p05_p95_deg=np.quantile(pitch, [.05, .95]).tolist(),
                pitch_over_20deg_fraction=float(np.mean(np.abs(pitch) > 20)),
                low_trunk_fraction=float(np.mean(height < .085)),
                foot_clearance_p95_m=np.quantile(values[:, 2:], .95, axis=0).tolist(),
                foot_clearance_max_m=values[:, 2:].max(axis=0).tolist())


def head_center_metrics(errors):
    """DC head error in HOME-delta coordinates, not instantaneous walking shake."""
    values = np.asarray(errors, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or not len(values) or not np.isfinite(values).all():
        return dict(head_mean_error_deg=None, head_center_check_passed=False)
    mean = np.degrees(values.mean(axis=0))
    return dict(head_mean_error_deg=mean.tolist(),
                head_p95_p05_span_deg=np.degrees(np.quantile(values,.95,axis=0)-np.quantile(values,.05,axis=0)).tolist(),
                head_ac_rms_deg=np.degrees(values.std(axis=0)).tolist(),
                head_center_check_passed=bool(np.max(np.abs(mean)) <= 5.))


def head_optical_pitch_deg(camera_xmat):
    """World pitch of the head camera's -Z optical axis; positive is looking up."""
    optical_axis = -np.asarray(camera_xmat).reshape(3, 3)[:, 2]
    return math.degrees(math.atan2(float(optical_axis[2]),
                                   float(np.linalg.norm(optical_axis[:2]))))


def target_chatter_metrics(steps, step_limit):
    deltas = np.asarray(steps, dtype=float)
    if (deltas.ndim != 2 or deltas.shape[0] < 2 or deltas.shape[1] != 14
            or not np.isfinite(deltas).all() or step_limit is None
            or not math.isfinite(step_limit) or step_limit <= 0):
        return dict(target_saturation_fraction=None,
                    saturated_reversal_fraction_max_joint=None,
                    motion_quality_check_passed=False)
    saturated = np.abs(deltas) >= .99 * step_limit
    reversing = saturated[1:] & saturated[:-1] & (deltas[1:] * deltas[:-1] < 0)
    worst = float(np.max(np.mean(reversing, axis=0)))
    return dict(target_saturation_fraction=float(np.mean(saturated)),
                saturated_reversal_fraction_max_joint=worst,
                motion_quality_check_passed=worst <= MAX_SATURATED_REVERSAL_FRACTION)


def replay_cases(extended=False, curves=False):
    cases=[('stand',(0.,0.,0.)),('forward',(.1,0.,0.)),('turn',(0.,0.,.4))]
    if extended:
        cases.extend([('backward',(-.1,0.,0.)),('turn_right',(0.,0.,-.4))])
    if curves:
        cases.extend([('forward_left', (.1, 0., .4)), ('forward_right', (.1, 0., -.4)),
                      ('backward_left', (-.1, 0., .4)), ('backward_right', (-.1, 0., -.4))])
    return cases


def transition_cases():
    cases = [('stand', (0.,0.,0.))]
    for name, command in replay_cases(True)[1:]:
        cases.extend([(name,command),('stand_after_'+name,(0.,0.,0.))])
    return cases


@contextlib.contextmanager
def replay_viewer(model,data):
    handle=mujoco.viewer.launch_passive(model,data)
    try:
        yield handle
    finally:
        handle.close()
        # MuJoCo 3.10 close() only requests exit. Its daemon render thread must
        # destroy Simulate before Python's atexit calls glfw.terminate().
        deadline=time.monotonic()+5
        while handle._sim() is not None:
            if time.monotonic()>deadline:
                raise RuntimeError('MuJoCo render thread did not finish closing')
            time.sleep(.01)


class ReplayPolicy(PolicyInference):
    previous_velocity=None

    def set_head_command(self, value):
        self.head_offset[:] = value
        self._update_command()

    def set_imu_observation_delay(self, milliseconds):
        self.imu_age_s = milliseconds / 1000.
        self.imu_samples = deque(maxlen=32)

    def record_imu_sample(self):
        if getattr(self, 'imu_age_s', 0.) > 0:
            self.imu_samples.append((float(self.data.time),
                np.concatenate((super().get_base_ang_vel(), super().get_projected_gravity()))))

    def get_observations(self):
        obs = super().get_observations()
        if getattr(self, 'imu_age_s', 0.) > 0:
            if not self.imu_samples:
                self.record_imu_sample()
            cutoff = self.data.time - self.imu_age_s + 1e-9
            sample = self.imu_samples[0][1]
            for stamp, value in self.imu_samples:
                if stamp > cutoff:
                    break
                sample = value
            obs[:6] = sample
        return obs

    def set_joint_observation_delay(self, steps):
        self.joint_observation_delay_steps = steps
        self.current_joint_age_steps = steps
        self.reset_joint_observation_history()

    def reset_joint_observation_history(self):
        self.previous_velocity = None
        self.joint_samples = None
        self.delayed_joint_velocity = None
        if hasattr(self, 'imu_samples'):
            self.imu_samples.clear()

    def get_joint_pos_relative(self):
        current_pos = super().get_joint_pos_relative()
        steps = getattr(self, 'joint_observation_delay_steps', 0)
        if steps == 0:
            return current_pos
        current_vel = super().get_joint_vel()
        if self.joint_samples is None:
            self.joint_samples = deque([(current_pos.copy(), current_vel.copy())] * steps,
                                       maxlen=steps + 1)
        self.joint_samples.append((current_pos.copy(), current_vel.copy()))
        delayed_pos, self.delayed_joint_velocity = self.joint_samples[-1-self.current_joint_age_steps]
        return delayed_pos.copy()

    def get_joint_vel(self):
        if getattr(self, 'joint_observation_delay_steps', 0):
            current = self.delayed_joint_velocity.copy()
            if not getattr(self, 'legacy_velocity_lag', False):
                return current
        else:
            current = super().get_joint_vel()
            if getattr(self, 'coherent_joint_snapshot', False) and not getattr(self, 'legacy_velocity_lag', False):
                return current
        previous=current if self.previous_velocity is None else self.previous_velocity
        self.previous_velocity=current
        return previous


class ReferenceReplayPolicy(ReplayPolicy):
    """Simulation-only upstream contract: raw history, optional runtime EMA."""

    def reset_joint_observation_history(self):
        super().reset_joint_observation_history()
        self.filtered_action = np.zeros(14, dtype=np.float32)

    def infer(self):
        action = super().infer()
        self.filtered_action *= self.action_alpha
        self.filtered_action += (1. - self.action_alpha) * action
        return self.filtered_action.copy()


def reference_policy_pose(metadata, joint_names):
    expected_obs = 'base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command'
    if (metadata.get('joint_names') != ','.join(joint_names)
            or metadata.get('observation_names') != expected_obs
            or metadata.get('action_scale') != '1.0'
            or metadata.get('action_semantics', 'raw_home_delta') != 'raw_home_delta'):
        raise ValueError('unsupported upstream joint/observation/action contract')
    pose = np.asarray([float(x) for x in metadata.get('default_joint_pos', '').split(',')], dtype=np.float32)
    if pose.shape != (14,) or not np.isfinite(pose).all():
        raise ValueError('invalid upstream HOME pose')
    return pose


def joint_age_lags_from_capture(path):
    ages = []
    for line in path.open():
        frame = json.loads(line)
        feedback = frame.get('feedback') or {}
        age = feedback.get('joint_age_s')
        if (frame.get('policy') == 'walk' and feedback.get('control_valid') is True
                and not feedback.get('error') and isinstance(age, (int, float))
                and math.isfinite(age) and 0 <= age < .1):
            ages.append(max(1, min(4, math.ceil(age / .02))))
    if not ages:
        raise ValueError('capture has no fresh walk joint ages')
    return np.asarray(ages, dtype=np.int8)


def load_replay_model(voltage, posture=False, bam_reference=False, voltage_extrapolation=False,
                      repair_variant=None, ground_contact=False):
    if ground_contact and not bam_reference:
        raise ValueError('ground-contact gait evaluation requires the M6 model')
    if bam_reference:
        if posture:
            raise ValueError('M6 reference currently supports velocity only')
        from mjlab_microduck.tasks.hd1910_bam import make_xgo_bam_env_cfg
        from mjlab_microduck.actuator.cpu_hd1910_bam import XgoBamCpuController
        cfg = make_xgo_bam_env_cfg(play=True, repair_variant=repair_variant)
        if ground_contact:
            from functools import partial
            from mjlab_microduck.actuator.reference_hd1910 import make_hd1910_spec
            from mjlab_microduck.robot.microduck_constants import MICRODUCK_GROUNDCONTACT_XML
            cfg.scene.entities['robot'].spec_fn = partial(
                make_hd1910_spec, MICRODUCK_GROUNDCONTACT_XML, clear_actuators=False)
        motor_cfg = cfg.scene.entities['robot'].articulation.actuators[0]
        motor_cfg.vin_range = (voltage, voltage)
        spec = cfg.scene.entities['robot'].build().spec
        spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .1])
        model = spec.compile()
        cfg.sim.mujoco.apply(model)
        data = mujoco.MjData(model)
        return model, data, XgoBamCpuController(model, data, voltage,
                                               voltage_extrapolation=voltage_extrapolation)
    scene = 'scene_allcollisions.xml' if posture else 'scene_walk.xml'
    model,data,motor=load_cpu_model(ROOT/'src/mjlab_microduck/robot/microduck'/scene,voltage)
    if posture:
        from mjlab_microduck.tasks.hd1910_suite import adapt_task
        from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg
        cfg = adapt_task(make_microduck_sitstand_env_cfg(play=True),slew=True)
    else:
        cfg=make_reference_hd1910_velocity_env_cfg(play=True)
    cfg.sim.mujoco.apply(model)
    # The scene XML alone omits the entity's contact overrides. Copy only robot
    # geom fields, preserving the replay scene's floor, camera and lights.
    trained=cfg.scene.entities['robot'].build().spec.compile()
    for i in range(trained.ngeom):
        name=trained.geom(i).name
        if not name:
            continue
        j=model.geom(name).id
        for field in ('geom_contype','geom_conaffinity','geom_condim','geom_priority',
                      'geom_friction','geom_solref','geom_solimp','geom_margin',
                      'geom_gap','geom_solmix'):
            getattr(model,field)[j]=getattr(trained,field)[i]
    return model,data,motor


def step_control_period(model,data,motor, observation_policy=None):
    for _ in range(4):
        motor.update()
        mujoco.mj_step(model,data)
        if observation_policy is not None:
            mujoco.mj_forward(model,data)
            observation_policy.record_imu_sample()
    # ManagerBasedRlEnv refreshes derived poses and sensors before observations.
    # mj_step alone leaves them at the pre-integration physics substep.
    mujoco.mj_forward(model,data)


def validate_metadata(metadata,joint_names,*,posture=False,bam_reference=False,roulade=False,step=False):
    profile_path = PROFILE_PATH
    if bam_reference:
        from mjlab_microduck.actuator.cpu_hd1910_bam import PROFILE_PATH as profile_path, TASK_ID, KP_FW
        from mjlab_microduck.tasks.hd1910_bam import SITSTAND_TASK_ID, ROULADE_TASK_ID, STEP_TASK_ID
        expected_task = STEP_TASK_ID if step else ROULADE_TASK_ID if roulade else SITSTAND_TASK_ID if posture else TASK_ID
        if metadata.get('actuator_backend') != 'hd1910_bam_m6' or metadata.get('task_id') != expected_task:
            raise ValueError('M6 replay requires its own explicitly tagged policy')
    digest=hashlib.sha256(profile_path.read_bytes()).hexdigest()
    expected={
        'task_id':'Mjlab-Velocity-Flat-MicroDuck-HD1910-Reference',
        'calibration_sha256':digest,
        'joint_names':','.join(joint_names),
        'observation_names':'base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command',
    }
    if bam_reference:
        expected['task_id'] = TASK_ID.removesuffix('-Slew')
        expected['kp_fw'] = str(KP_FW)
    if roulade:
        if not bam_reference or posture:
            raise ValueError('roulade requires M6 and its own contract')
        expected.update(task_id=ROULADE_TASK_ID,policy_role='roulade',
            command_semantics='episodic_roll_zero_command',action_semantics='bounded_slew_home_delta_v2',
            previous_action_semantics='bounded_slew_home_delta_v2')
        if float(metadata.get('policy_period_s','nan')) != .02 or not 0 < float(metadata.get('max_action_step_rad','nan')) <= .12:
            raise ValueError('roulade action contract mismatch')
    if step:
        expected.update(task_id=STEP_TASK_ID, policy_role='step', command_semantics='phase_cos_sin_zero',
            action_semantics='bounded_slew_home_delta_v2', previous_action_semantics='bounded_slew_home_delta_v2',
            step_period_s='1.0')
    if posture:
        if bam_reference:
            expected['task_id'] = SITSTAND_TASK_ID
        else:
            expected['task_id']='Mjlab-SitStand-Flat-MicroDuck-HD1910-Reference'
            allowed=tuple(expected['task_id']+s for s in ('-Slew','-Slew-Refine','-Slew-Balanced'))
            if metadata.get('task_id') not in allowed:
                raise ValueError('posture replay requires a bounded SitStand policy')
        expected.update(command_semantics='sit_flag_zero_zero',stand_flag='0',sit_flag='1')
        if float(metadata.get('posture_ramp_s','nan')) != 2.:
            raise ValueError('posture ramp contract mismatch')
    if metadata.get('task_id') == expected['task_id']+'-Bounded':
        expected['task_id'] += '-Bounded'
        if (metadata.get('action_semantics') != 'bounded_home_delta_v1'
                or metadata.get('previous_action_semantics') != 'bounded_home_delta_v1'):
            raise ValueError('bounded policy history/output contract missing')
    elif metadata.get('task_id') in tuple(expected['task_id']+s for s in (
            ('-Slew','-Slew-Refine','-Slew-Balanced') if posture
            else ('-Slew','-Slew-Discovery','-Slew-Refine'))):
        expected['task_id'] = metadata['task_id']
        if (metadata.get('action_semantics') != 'bounded_slew_home_delta_v2'
                or metadata.get('previous_action_semantics') != 'bounded_slew_home_delta_v2'
                or not 0 < float(metadata.get('max_action_step_rad','nan')) <= .12
                or float(metadata.get('policy_period_s','nan')) != .02):
            raise ValueError('slew policy history/output contract missing')
    for key,value in expected.items():
        if metadata.get(key)!=value:
            raise ValueError(f'policy {key} mismatch; cannot silently replay a different contract')
    if float(metadata.get('action_scale','nan'))!=1.0:
        raise ValueError('policy action scale mismatch')
    pose=np.asarray([float(x) for x in metadata.get('default_joint_pos','').split(',')])
    # The upstream exporter rounds this metadata to three decimal places.
    if pose.shape!=(14,) or not np.allclose(pose,DEFAULT_POSE,rtol=0,atol=.00051):
        raise ValueError('policy default pose mismatch')
    return digest


def baseline_check(row,seconds):
    """Local screening only: a stationary forward policy must not pass."""
    if seconds<20 or not row['completed'] or not row['no_fall'] or row['max_tilt_deg']>45 or row.get('target_limit_violations',0):
        return False
    vx,vy,wz=row['mean_body_velocity_after_1s']
    cx,cy,cw=row['command']
    linear_tolerance=.02 if row['case'].startswith('stand') else .05
    return (abs(vx-cx)<linear_tolerance and abs(vy-cy)<linear_tolerance
            and abs(wz-cw)<.15 and row['rms_vx_error_after_1s']<.2
            and row['rms_yaw_error_after_1s']<.6)


def straight_gait_check(row):
    """Local 20 s screening, separate from historical velocity-only results."""
    path = row.get('prefall_straight_path')
    peaks = row['prefall_feet']['swing_peak_median_mm']
    if path is None:
        return None
    return bool(row['steps'] >= 1000 and row['completed'] and row['no_fall']
        and row['first_head_floor_contact_s'] is None
        and row['baseline_check_passed']
        and all(p is not None and p >= 15. for p in peaks)
        and min(peaks)/max(peaks) >= .8
        and path['max_heading_error_deg'] <= 15.
        and path['max_cross_track_m'] <= .15)


def replay(args):
    if not math.isfinite(args.seconds) or not .02<=args.seconds<=300:
        raise ValueError('replay duration must be finite and within 0.02..300 seconds per case')
    bam_reference = getattr(args, 'bam_reference', False)
    model,data,motor=load_replay_model(args.voltage, bam_reference=bam_reference,
        voltage_extrapolation=getattr(args, 'voltage_extrapolation', False),
        ground_contact=getattr(args, 'ground_contact', False))
    from mjlab_microduck.actuator.payload_uncertainty import apply_mass_scenario
    mass_scenario = apply_mass_scenario(model, data, getattr(args, 'mass_scenario', 'cad_nominal'))
    if args.viewer or args.video or args.snapshot:
        # Robot-only M6 specs do not carry the scene XML's lighting.
        model.vis.headlight.active = 1
        model.vis.headlight.ambient[:] = [.35, .35, .35]
        model.vis.headlight.diffuse[:] = [.7, .7, .7]
        model.vis.headlight.specular[:] = [.1, .1, .1]
        model.geom_rgba[model.geom('floor').id] = [.64, .79, .87, 1.]
    delay = getattr(args, 'delay_steps', 4)
    tilt_range = getattr(args, 'initial_tilt_deg', 0.)
    loss_probability = getattr(args, 'command_loss_probability', 0.)
    hold_max = getattr(args, 'command_hold_max_steps', 3)
    if not math.isfinite(loss_probability) or not 0 <= loss_probability < .5 or not 1 <= hold_max <= 5:
        raise ValueError('invalid command delivery stress')
    if delay not in range(3,11) or not math.isfinite(tilt_range) or not 0 <= tilt_range <= 10:
        raise ValueError('stress replay requires delay 3..10 and initial tilt 0..10 degrees')
    motor.delay = delay
    external_reference = getattr(args, 'reference_policy', False)
    if external_reference and not bam_reference:
        raise ValueError('--reference-policy requires --bam-reference; simulation only')
    policy_class = ReferenceReplayPolicy if external_reference else ReplayPolicy
    policy=policy_class(model,data,walking_onnx_path=str(args.policy),bam_ctrl=motor,
                        new_cmd_obs=True,use_projected_gravity=True)
    if getattr(args, 'action_diagnostics', False):
        from action_request_probe import ActionRequestProbe
        policy.ort_session = ActionRequestProbe(policy.ort_session, args.policy)
    if external_reference:
        policy.action_alpha = getattr(args, 'reference_action_alpha', .45)
        if not math.isfinite(policy.action_alpha) or not 0 <= policy.action_alpha < 1:
            raise ValueError('invalid Reference EMA coefficient')
    joint_age_steps = getattr(args, 'joint_age_steps', 0)
    age_capture = getattr(args, 'joint_age_capture', None)
    if age_capture and joint_age_steps:
        raise ValueError('choose a fixed joint age or a captured age sequence')
    captured_ages = joint_age_lags_from_capture(age_capture) if age_capture else None
    if joint_age_steps not in range(5):
        raise ValueError('joint observation age must be 0..4 control steps')
    policy.set_joint_observation_delay(4 if captured_ages is not None else joint_age_steps)
    imu_age_ms = getattr(args, 'imu_age_ms', 0.)
    if not math.isfinite(imu_age_ms) or not 0 <= imu_age_ms <= 80:
        raise ValueError('IMU observation age must be within 0..80 ms')
    policy.set_imu_observation_delay(imu_age_ms)
    policy.coherent_joint_snapshot = policy.ort_session.get_modelmeta().custom_metadata_map.get(
        'joint_snapshot_training') == 'coherent_pos_vel_delay_v1'
    policy.legacy_velocity_lag = getattr(args, 'legacy_velocity_lag', False)
    neck_deg = getattr(args, 'head_neck_deg', 0.)
    pitch_deg = getattr(args, 'head_pitch_deg', 0.)
    yaw_deg = getattr(args, 'head_yaw_deg', 0.)
    roll_deg = getattr(args, 'head_roll_deg', 0.)
    if not all(math.isfinite(x) and abs(x) <= 20 for x in (neck_deg, pitch_deg)):
        raise ValueError('head command offsets must be finite and within +/-20 degrees')
    if not math.isfinite(yaw_deg) or abs(yaw_deg) > 80 or not math.isfinite(roll_deg) or abs(roll_deg) > 17:
        raise ValueError('simulation head yaw/roll must be finite and within +/-80/17 degrees')
    head_offset = np.radians([neck_deg, pitch_deg, yaw_deg, roll_deg])
    policy.set_head_command(head_offset)
    head_at = getattr(args, 'head_command_at_s', 0.)
    pitch_push = getattr(args, 'pitch_push_rad_s', 0.)
    head_push = getattr(args, 'head_push_n', 0.)
    if not math.isfinite(head_at) or not 0 <= head_at < args.seconds:
        raise ValueError('head command time must lie within the replay')
    if not math.isfinite(pitch_push) or not 0 <= pitch_push <= 2.:
        raise ValueError('pitch push must be within 0..2 rad/s')
    if not math.isfinite(head_push) or not 0 <= head_push <= 2.:
        raise ValueError('simulation head force must be within 0..2 N')
    if policy.ort_session.get_inputs()[0].shape != [1,61] or policy.ort_session.get_outputs()[0].shape != [1,14]:
        raise ValueError('expected normalized 61 -> 14 policy')
    metadata=policy.ort_session.get_modelmeta().custom_metadata_map
    if metadata.get('policy_role') == 'recovery':
        raise ValueError('Use replay_m6_recovery.py for full-collision recovery evaluation')
    joint_names=[model.joint(int(j)).name for j in motor.joint_ids]
    head_camera_id = model.camera('head_camera').id
    feet = []
    for name in ('left_foot_collision', 'right_foot_collision'):
        gid = model.geom(name).id
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError('clearance metrics require the MicroDuck foot collision meshes')
        mesh = model.geom_dataid[gid]
        start = model.mesh_vertadr[mesh]
        feet.append((gid, model.mesh_vert[start:start+model.mesh_vertnum[mesh]]))
    if external_reference:
        policy.default_pose = reference_policy_pose(metadata, joint_names)
        policy.coherent_joint_snapshot = True
        from mjlab_microduck.actuator.cpu_hd1910_bam import PROFILE_PATH as replay_profile
        digest = hashlib.sha256(replay_profile.read_bytes()).hexdigest()
    else:
        digest=validate_metadata(metadata,joint_names,bam_reference=bam_reference)
    continuous=getattr(args,'transition_test',False)
    cases=transition_cases() if continuous else replay_cases(args.extended, getattr(args, 'curve_cases', False))
    required_cases=len(cases)
    if args.case!='all': cases=[c for c in cases if c[0]==args.case]
    rng=np.random.default_rng(args.seed)
    rows=[]
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        viewer=stack.enter_context(replay_viewer(model,data)) if args.viewer else None
        renderer=writer=None
        if args.video or args.snapshot:
            renderer=stack.enter_context(mujoco.Renderer(model,height=480,width=640))
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True,exist_ok=True)
            writer=stack.enter_context(imageio.get_writer(str(args.video),fps=25))
        camera=mujoco.MjvCamera()
        camera.distance,camera.elevation,camera.azimuth=.75,-20,135
        if viewer:
            viewer.cam.distance,viewer.cam.elevation,viewer.cam.azimuth=.75,-20,135
        captured=False
        trace=None
        if args.trace:
            args.trace.parent.mkdir(parents=True,exist_ok=True)
            trace=csv.writer(stack.enter_context(args.trace.open('w',newline='')))
            trace.writerow(['case','time_s','x_m','y_m','z_m','tilt_deg','vx_m_s','vy_m_s','wz_rad_s',
                            *[f'q_{name}_rad' for name in joint_names],
                            *[f'target_{name}_rad' for name in joint_names],
                            'left_sole_clearance_m','right_sole_clearance_m',
                            'left_ground_contact','right_ground_contact',
                            'root_qw','root_qx','root_qy','root_qz','wx_rad_s','wy_rad_s'])
        for case_index,(name,command) in enumerate(cases):
            if not continuous or case_index==0:
                mujoco.mj_resetData(model,data)
                data.qpos[:7]=[0,0,.125,1,0,0,0]
                if tilt_range:
                    roll, pitch = np.radians(rng.uniform(-tilt_range,tilt_range,2))
                    data.qpos[3:7] = [math.cos(roll/2)*math.cos(pitch/2),
                                      math.sin(roll/2)*math.cos(pitch/2),
                                      math.cos(roll/2)*math.sin(pitch/2),
                                      -math.sin(roll/2)*math.sin(pitch/2)]
                data.qpos[policy.joint_qpos_indices]=policy.default_pose+rng.uniform(-.005,.005,14)
                motor.reset(data.qpos)
                policy.last_action[:]=0
                policy.reset_joint_observation_history()
            policy.set_vel_cmd(*command)
            mujoco.mj_forward(model,data)
            max_tilt=0.; min_height=float('inf'); first_fall=None
            straight_poses = [planar_pose(data.qpos)]
            squared_error=[]; yaw_error=[]; count=0; max_speed=0.; max_target_jump=0.
            settled_velocity=[]
            head_errors=[]
            settled_head_errors=[]
            head_world_speeds=[]
            head_optical_pitches=[]
            sole_peaks=[[], []]
            prefall_peaks=[[], []]
            contact_seen=np.zeros(2, dtype=bool)
            prefall_support=np.zeros(2, dtype=int)
            prefall_samples=0
            site_sole_gaps=[]
            foot_site_ids=[model.site(s).id for s in ('left_foot', 'right_foot')]
            swing_peaks=np.zeros(2)
            swing_active=np.zeros(2, dtype=bool)
            head_body_id=model.body('jaw_soft').id
            floor_id=model.geom('floor').id
            head_floor_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == head_body_id
                and ((model.geom_contype[i] & model.geom_conaffinity[floor_id])
                     or (model.geom_contype[floor_id] & model.geom_conaffinity[i]))}
            first_head_contact = None
            head_contact_frames = 0
            settled_target_steps=[]
            settled_joint_speeds=[]
            posture_samples=[]
            previous=policy.default_pose+policy.last_action
            hold_left = 0
            held_targets = 0
            target_limit_violations = 0
            initial_target_jump = None
            joint_limits = model.jnt_range[model.actuator_trnid[:,0]]
            for step in range(round(args.seconds*50)):
                if viewer and not viewer.is_running(): break
                begin=time.monotonic()
                policy.set_head_command(head_offset if step >= round(head_at*50) else 0.)
                if pitch_push and step in (350, 650):
                    data.qvel[4] += pitch_push * (1 if step == 350 else -1)
                    mujoco.mj_forward(model, data)
                data.xfrc_applied[head_body_id] = 0.
                if head_push and (450 <= step < 460 or 750 <= step < 760):
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
                    data.xfrc_applied[head_body_id, :3] = (
                        rotation.reshape(3, 3)[:, 0] * head_push * (1 if step < 460 else -1))
                if captured_ages is not None:
                    policy.current_joint_age_steps = int(captured_ages[(step + case_index*1000) % len(captured_ages)])
                action=policy.infer()
                target=policy.default_pose+action*policy.action_scale
                if hold_left == 0 and loss_probability > 0 and rng.random() < loss_probability:
                    hold_left = int(rng.integers(1, hold_max + 1))
                if hold_left:
                    hold_left -= 1
                    held_targets += 1
                    target = previous.copy()
                    if not external_reference:
                        policy.last_action = ((target-policy.default_pose)/policy.action_scale).astype(np.float32)
                if not np.isfinite(target).all(): raise ValueError('nonfinite target')
                target_limit_violations += int(np.any((target < joint_limits[:,0]-1e-6) | (target > joint_limits[:,1]+1e-6)))
                if initial_target_jump is None:
                    initial_target_jump = float(np.max(np.abs(target-previous)))
                max_target_jump=max(max_target_jump,float(np.abs(target-previous).max()))
                if step>=50:
                    settled_target_steps.append((target-previous).copy())
                previous=target.copy()
                policy.set_position_targets(target)
                step_control_period(model,data,motor, policy if imu_age_ms else None)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError('nonfinite simulation')
                gravity=policy.get_projected_gravity()
                tilt=math.acos(float(np.clip(-gravity[2],-1,1)))
                height=float(data.qpos[2])
                max_tilt=max(max_tilt,tilt); min_height=min(min_height,height)
                head_contact = any(contact.dist <= 0 and floor_id in contact.geom
                    and any(int(g) in head_floor_geoms for g in contact.geom) for contact in data.contact)
                if head_contact:
                    head_contact_frames += 1
                    if first_head_contact is None:
                        first_head_contact = (step+1)/50
                if first_fall is None and (tilt>math.radians(60) or height<.06):
                    first_fall=(step+1)/50
                if first_fall is None and first_head_contact is None:
                    straight_poses.append(planar_pose(data.qpos))
                velocity=policy.quat_rotate_inverse(data.qpos[3:7],data.qvel[:3])
                body_omega=policy.get_base_ang_vel()
                wz=float(body_omega[2])
                squared_error.append((float(velocity[0])-command[0])**2)
                yaw_error.append((wz-command[2])**2)
                if step>=50 or trace:
                    clearance = [float(np.min(vertices @ data.geom_xmat[gid].reshape(3,3)[2]
                                                 + data.geom_xpos[gid,2])) for gid,vertices in feet]
                    contacts=set()
                    for contact in data.contact:
                        pair=set(map(int,contact.geom))
                        if floor_id in pair and contact.dist <= .001:
                            contacts.update(pair-{floor_id})
                if step>=50:
                    settled_velocity.append([float(velocity[0]),float(velocity[1]),wz])
                    head_errors.append((data.qpos[policy.joint_qpos_indices][5:9]-policy.default_pose[5:9]
                                        -policy.head_offset).copy())
                    head_optical_pitches.append(head_optical_pitch_deg(data.cam_xmat[head_camera_id]))
                    spatial_velocity=np.empty(6)
                    mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,head_body_id,spatial_velocity,0)
                    head_world_speeds.append(float(np.linalg.norm(spatial_velocity[:3])))
                    settled_joint_speeds.append(float(np.mean(data.qvel[policy.joint_qvel_indices]**2)))
                    site_sole_gaps.append(data.site_xpos[foot_site_ids,2] - clearance)
                    posture_samples.append([math.degrees(math.asin(float(np.clip(gravity[0],-1,1)))),
                                            height, *clearance])
                    for foot_index,(gid,_) in enumerate(feet):
                        if first_fall is None and first_head_contact is None:
                            prefall_support[foot_index] += int(gid in contacts)
                        if gid not in contacts:
                            swing_active[foot_index]=True
                            swing_peaks[foot_index]=max(swing_peaks[foot_index],clearance[foot_index])
                        elif swing_active[foot_index]:
                            sole_peaks[foot_index].append(float(swing_peaks[foot_index]))
                            if contact_seen[foot_index] and first_fall is None and first_head_contact is None:
                                prefall_peaks[foot_index].append(float(swing_peaks[foot_index]))
                            swing_peaks[foot_index]=0.
                            swing_active[foot_index]=False
                        contact_seen[foot_index] |= gid in contacts
                    prefall_samples += int(first_fall is None and first_head_contact is None)
                if trace:
                    trace.writerow([name,(step+1)/50,*data.qpos[:3],math.degrees(tilt),
                                    velocity[0],velocity[1],wz,
                                    *data.qpos[policy.joint_qpos_indices],*target,
                                    *clearance,*[int(gid in contacts) for gid,_ in feet],
                                    *data.qpos[3:7],body_omega[0],body_omega[1]])
                max_speed=max(max_speed,float(np.abs(data.qvel[policy.joint_qvel_indices]).max()))
                count+=1
                if first_fall is None and step >= max(50, round((head_at+1.)*50)):
                    settled_head_errors.append(data.qpos[policy.joint_qpos_indices][5:9]
                                               - policy.default_pose[5:9] - policy.head_offset)
                if renderer and step%2==0:
                    camera.lookat[:]=data.qpos[:3]
                    camera.lookat[2] += .07
                    renderer.update_scene(data,camera=camera)
                    pixels=renderer.render()
                    from PIL import Image, ImageDraw, ImageFont
                    frame = Image.fromarray(pixels)
                    draw = ImageDraw.Draw(frame, 'RGBA')
                    draw.rectangle((0, 0, frame.width, 93), fill=(10, 20, 30, 200))
                    head_actual = np.degrees(data.qpos[policy.joint_qpos_indices][5:7] - policy.default_pose[5:7])
                    text_lines = [
                        f'{name}  t={(step+1)/50:.1f}s  SIMULATION  |  {"FALL" if first_fall is not None else "running"}',
                        f'vx cmd/actual {command[0]:+.2f}/{velocity[0]:+.2f} m/s  yaw {command[2]:+.2f}/{wz:+.2f} rad/s',
                        f'neck/head cmd {np.degrees(policy.head_offset[0]):+.0f}/{np.degrees(policy.head_offset[1]):+.0f}  actual {head_actual[0]:+.1f}/{head_actual[1]:+.1f} deg',
                        f'tilt {math.degrees(tilt):.1f} deg  head force {np.linalg.norm(data.xfrc_applied[head_body_id,:3]):.2f} N',
                    ]
                    draw.multiline_text((8, 5), '\n'.join(text_lines),
                                        fill=(235, 245, 255), font=ImageFont.load_default(size=15), spacing=3)
                    if step >= 50:
                        draw.rectangle((0, frame.height-27, frame.width, frame.height), fill=(10,20,30,200))
                        draw.text((8, frame.height-23),
                                  f'sole L/R {clearance[0]*1000:.1f}/{clearance[1]*1000:.1f} mm  diagnostic only',
                                  fill=(235,245,255), font=ImageFont.load_default(size=15))
                    pixels = np.asarray(frame)
                    if writer: writer.append_data(pixels)
                    if args.snapshot and not captured and step>=10:
                        args.snapshot.parent.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(pixels).save(args.snapshot)
                        captured=True
                if viewer:
                    viewer.cam.lookat[:]=data.qpos[:3]
                    viewer.cam.lookat[2] += .07
                    viewer.sync()
                    time.sleep(max(0,.02-(time.monotonic()-begin)))
            rows.append(dict(case=name,command=command,steps=count,first_fall_s=first_fall,
                             max_tilt_deg=math.degrees(max_tilt),min_trunk_height_m=min_height if count else None,
                             rms_vx_error_m_s=math.sqrt(float(np.mean(squared_error))) if count else None,
                             rms_yaw_rate_error_rad_s=math.sqrt(float(np.mean(yaw_error))) if count else None,
                             mean_body_velocity_after_1s=np.mean(settled_velocity,axis=0).tolist() if settled_velocity else None,
                             rms_vx_error_after_1s=math.sqrt(float(np.mean(squared_error[50:]))) if count>50 else None,
                             rms_yaw_error_after_1s=math.sqrt(float(np.mean(yaw_error[50:]))) if count>50 else None,
                             rms_target_step_after_1s_rad=math.sqrt(float(np.mean(np.square(settled_target_steps)))) if settled_target_steps else None,
                             rms_joint_speed_after_1s_rad_s=math.sqrt(float(np.mean(settled_joint_speeds))) if settled_joint_speeds else None,
                             displacement_xy_m=data.qpos[:2].tolist(),max_joint_speed_rad_s=max_speed,
                             max_target_jump_rad=max_target_jump,
                             initial_target_jump_rad=initial_target_jump,
                             target_limit_violations=target_limit_violations,
                             completed=count==round(args.seconds*50),no_fall=count>0 and first_fall is None))
            rows[-1]['baseline_check_passed']=baseline_check(rows[-1],args.seconds)
            rows[-1]['prefall_straight_path'] = (straight_path_metrics(straight_poses)
                if command[0] != 0 and command[1] == 0 and command[2] == 0 else None)
            # Keep the old no_fall criterion comparable; this stricter metric
            # explicitly requires a model capable of head/floor collisions.
            rows[-1].update(head_floor_collision_available=bool(head_floor_geoms),
                first_head_floor_contact_s=first_head_contact,
                head_floor_contact_frames=head_contact_frames,
                no_fall_or_head_contact=(rows[-1]['no_fall'] and first_head_contact is None
                                         if head_floor_geoms else None))
            rows[-1].update(head_center_metrics(head_errors))
            rows[-1]['settled_prefall_head_metrics'] = head_center_metrics(settled_head_errors)
            rows[-1].update(head_optical_pitch_mean_deg=float(np.mean(head_optical_pitches)) if head_optical_pitches else None,
                head_optical_pitch_p05_p95_deg=np.quantile(head_optical_pitches, [.05, .95]).tolist() if head_optical_pitches else None)
            rows[-1].update(head_world_speed_rms_rad_s=float(np.sqrt(np.mean(np.square(head_world_speeds)))) if head_world_speeds else None,
                head_world_speed_p95_rad_s=float(np.quantile(head_world_speeds,.95)) if head_world_speeds else None,
                simulated_target_hold_fraction=held_targets/max(count, 1),
                foot_landing_count=[len(x) for x in sole_peaks],
                foot_swing_peak_median_m=[float(np.median(x)) if x else None for x in sole_peaks])
            rows[-1]['prefall_feet'] = dict(order=['left', 'right'],
                sample_seconds=prefall_samples/50,
                support_fraction=(prefall_support/prefall_samples).tolist() if prefall_samples else None,
                complete_swing_count=[len(x) for x in prefall_peaks],
                swing_peak_median_mm=[float(np.median(x)*1000) if x else None for x in prefall_peaks],
                swing_over_15mm_fraction=[float(np.mean(np.asarray(x)>=.015)) if x else None for x in prefall_peaks],
                swing_over_25mm_fraction=[float(np.mean(np.asarray(x)>=.025)) if x else None for x in prefall_peaks])
            rows[-1]['straight_gait_check_passed'] = straight_gait_check(rows[-1])
            rows[-1].update(posture_metrics(posture_samples))
            # On the flat replay floor, the height sensor's site-to-floor
            # distance is not the clearance of the lowest tilted sole vertex.
            rows[-1]['foot_site_minus_sole_mean_m'] = np.mean(site_sole_gaps,axis=0).tolist() if site_sole_gaps else None
            rows[-1].update(target_chatter_metrics(settled_target_steps,
                float(metadata['max_action_step_rad']) if 'max_action_step_rad' in metadata else None))
            if getattr(args, 'action_diagnostics', False):
                rows[-1]['action_request_diagnostics'] = policy.ort_session.take_metrics()
    result=dict(policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                head_command_schedule_version=2,
                external_reference_policy=external_reference,
                reference_action_alpha=policy.action_alpha if external_reference else None,
                home_pose_rad=policy.default_pose.tolist(),
                previous_action_semantics='raw_home_delta' if external_reference else metadata.get('previous_action_semantics'),
                ground_contact_model=bool(getattr(args, 'ground_contact', False)),
                profile_sha256=digest,calibration_status='external_reference_unvalidated',
                voltage_v=args.voltage,performance_voltage_cap_v=None if bam_reference else 7.4,
                voltage_extrapolation=bool(getattr(args, 'voltage_extrapolation', False)),
                actuator_backend='hd1910_bam_m6' if bam_reference else 'hd1910_reference_pd',
                actuator_delay_ms=delay*5,command_loss_probability=loss_probability,
                command_hold_max_steps=hold_max,policy_hz=50,physics_hz=200,
                joint_position_observation_delay_ms=joint_age_steps*20 if captured_ages is None else None,
                joint_velocity_observation_delay_ms=(joint_age_steps + int(policy.legacy_velocity_lag))*20
                    if captured_ages is None and policy.coherent_joint_snapshot
                    else (max(1, joint_age_steps)*20 if captured_ages is None else None),
                legacy_velocity_lag=policy.legacy_velocity_lag,
                joint_age_capture_sha256=hashlib.sha256(age_capture.read_bytes()).hexdigest() if age_capture else None,
                joint_age_capture_lag_counts=np.bincount(captured_ages, minlength=5).tolist() if captured_ages is not None else None,
                integrator='implicitfast',imu_observation_delay_ms=imu_age_ms,
                observations_refreshed_after_integration=True,
                actuator_history_initialization='first_command',
                seed=args.seed,seconds_per_case=args.seconds,extended_cases=args.extended,
                head_command_offset_deg=[neck_deg,pitch_deg],
                head_yaw_roll_command_deg=[yaw_deg,roll_deg],
                head_command_at_s=head_at, pitch_push_rad_s=pitch_push, head_push_n=head_push,
                head_push_window_s=[[9., 9.2], [15., 15.2]],
                delay_physics_steps=delay,initial_tilt_range_deg=tilt_range,
                mass_scenario=mass_scenario,
                hardware_tested=False,deployment_ready=False,
                action_semantics=metadata.get('action_semantics','raw_home_delta'),
                max_action_step_rad=float(metadata['max_action_step_rad']) if 'max_action_step_rad' in metadata else None,
                continuous_transitions=continuous,
                replay_completed=all(r['completed'] for r in rows),
                baseline_checks_passed=len(rows)==required_cases and all(r['baseline_check_passed'] and r['motion_quality_check_passed'] for r in rows),
                motion_quality_criteria=dict(saturated_step_ratio=.99,
                    max_saturated_reversal_fraction_per_joint=MAX_SATURATED_REVERSAL_FRACTION),
                baseline_criteria=dict(min_seconds_per_case=20,max_tilt_deg=45,
                                       stand_mean_linear_tolerance_m_s=.02,moving_mean_linear_tolerance_m_s=.05,
                                       mean_yaw_tolerance_rad_s=.15,rms_vx_error_m_s=.2,rms_yaw_error_rad_s=.6),
                gait_accepted=False,cases=rows)
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--reference-policy', action='store_true', help='Simulation only: upstream raw-action history and metadata HOME')
    p.add_argument('--reference-action-alpha', type=float, default=.45, help='Upstream old-action EMA weight; 0 tests the unfiltered training contract')
    p.add_argument('--bam-reference',action='store_true',help='Explicit external M6 model; rejects PD policies')
    p.add_argument('--ground-contact', action='store_true',
                   help='Use existing body/head ground-contact MJCF; report head impact separately')
    from mjlab_microduck.actuator.payload_uncertainty import SCENARIOS
    p.add_argument('--mass-scenario', choices=SCENARIOS, default='cad_nominal',
                   help='Explicit uncalibrated payload sensitivity, not measured replica mass')
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=10)
    p.add_argument('--voltage',type=float,default=7.4)
    p.add_argument('--voltage-extrapolation',action='store_true',
                   help='Simulation-only 6..8 V stress, outside identified/training voltage; no hardware approval')
    p.add_argument('--delay-steps',type=int,default=4,choices=range(3,11))
    p.add_argument('--command-loss-probability',type=float,default=0.)
    p.add_argument('--command-hold-max-steps',type=int,default=3)
    p.add_argument('--joint-age-steps', type=int, default=0, choices=range(5),
                   help='Extra replay of a shared joint snapshot, in 20 ms control steps')
    p.add_argument('--joint-age-capture', type=Path,
                   help='Replay the age sequence of fresh walk snapshots from a passive JSONL capture')
    p.add_argument('--imu-age-ms', type=float, default=0.,
                   help='Delay policy gyro/gravity only; physics and evaluation remain undelayed')
    p.add_argument('--legacy-velocity-lag', action='store_true',
                   help='Diagnostic only: reproduce the old runtime extra velocity-only cycle')
    p.add_argument('--initial-tilt-deg',type=float,default=0.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--case',choices=('all','stand','forward','turn','backward','turn_right'),default='all')
    p.add_argument('--extended',action='store_true',help='Also evaluate backward motion and negative yaw')
    p.add_argument('--transition-test',action='store_true',help='Keep simulation and action history across move/stop stages')
    p.add_argument('--curve-cases', action='store_true', help='Also test forward/reverse turning together')
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--video',type=Path)
    p.add_argument('--snapshot',type=Path)
    p.add_argument('--trace',type=Path,help='Optional per-step pose, velocity, joint and target CSV')
    p.add_argument('--action-diagnostics', action='store_true',
                   help='Read latent requests from a separate in-memory graph; policy output unchanged')
    p.add_argument('--head-neck-deg', type=float, default=0., help='Simulation-only neck pitch command offset')
    p.add_argument('--head-pitch-deg', type=float, default=0., help='Simulation-only head pitch command offset')
    p.add_argument('--head-yaw-deg', type=float, default=0., help='Simulation-only head yaw command offset')
    p.add_argument('--head-roll-deg', type=float, default=0., help='Simulation-only head roll command offset')
    p.add_argument('--head-command-at-s', type=float, default=0., help='Start head command during ongoing locomotion')
    p.add_argument('--pitch-push-rad-s', type=float, default=0., help='Opposite body pitch velocity impulses at 7s and 13s')
    p.add_argument('--head-push-n', type=float, default=0., help='Simulation fore/aft force on head for 0.2s at 9s/15s')
    args=p.parse_args()
    if args.case in ('backward','turn_right'):
        args.extended=True
    replay(args)


if __name__=='__main__': main()
