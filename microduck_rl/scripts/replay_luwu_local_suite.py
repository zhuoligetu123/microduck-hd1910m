#!/usr/bin/env python3
"""Unmodified Luwu weights, LOCAL geometry/calibration, simulated HD1910/BNO I/O.

No serial port, SSH, enable command, or live policy replacement. Every case is
isolated; episodic pick/roll switches back to idle walk without resetting physics.
"""
import argparse
from collections import deque
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'radxa'))
from luwu_policy import (LocalCalibration, LuwuSuite, JOINTS, MODEL_DIR, INSTALLATION,
                         rotate, conjugate, BodyFeedbackFilter)
from replay_hd1910 import load_replay_model, step_control_period
from replay_m6_recovery import floor_clearance, recovery_success
from evaluate_luwu_direct import foot_vertices, clearance, summarize_swings
from mjlab_microduck.tasks.mdp import (_HEAD_TOP_AXIS, _HEAD_TOP_DOWN_MIN,
    _HEAD_LATCH_LO, _HEAD_LATCH_HI, _FLAT_FULL, _FLAT_ZERO)
from mjlab_microduck.actuator.cpu_xgoduck_bam import PROFILE_PATH


class SimulatedFeedback:
    """100 Hz coherent encoder + fused-IMU snapshots on a 200 Hz physics clock."""
    def __init__(self, model, data, motor, cal, filter_enabled):
        self.model,self.data,self.motor,self.cal = model,data,motor,cal
        self.filter = BodyFeedbackFilter() if filter_enabled else None
        self.samples = deque(maxlen=100)
        self.max_position_error = self.max_imu_error = 0.
        self.imu = model.sensor_adr[model.sensor('imu_ang_vel').id]
        self.record_imu_sample()

    def record_imu_sample(self):
        now = float(self.data.time)
        if self.samples and now-self.samples[-1][0] < .01-1e-8:
            return
        q,dq = self.data.qpos[self.motor.qids],self.data.qvel[self.motor.vids]
        ticks,speeds = self.cal.simulate_encoders(q,dq)
        measured_q,measured_dq = self.cal.positions(ticks),self.cal.velocities(speeds)
        raw_gyro = self.data.sensordata[self.imu:self.imu+3]
        sensor_q,sensor_gyro = self.cal.simulate_imu(self.data.qpos[3:7],raw_gyro)
        gyro,gravity = self.cal.raw_imu_to_body(sensor_q,sensor_gyro)
        self.max_position_error = max(self.max_position_error,float(np.max(np.abs(measured_q-q))))
        self.max_imu_error = max(self.max_imu_error,float(np.max(np.abs(gyro-raw_gyro))),
            float(np.max(np.abs(gravity-rotate(conjugate(self.data.qpos[3:7]),[0,0,-1])))))
        if self.filter is not None:
            gyro,measured_dq = self.filter.update(now,gyro,measured_dq)
        self.samples.append((now,measured_q,measured_dq,gyro,gravity))

    def read(self, delay_ms):
        selected = self.samples[0]
        for sample in self.samples:
            if sample[0] <= self.data.time-delay_ms/1000+1e-9:
                selected = sample
        return selected[1:]


def cases():
    return [
        ('idle','walk',(0,0,0),0,0,0),
        ('forward','walk',(.3,0,0),0,0,0),
        ('backward','walk',(-.3,0,0),0,0,0),
        ('left','walk',(0,0,.8),0,0,0),
        ('right','walk',(0,0,-.8),0,0,0),
        ('head_mouth','walk',(0,0,0),0,0,0),
        ('push_forward','walk',(.3,0,0),0,0,1),
        ('getup_front','getup',(0,0,0),90,0,0),
        ('getup_back','getup',(0,0,0),-90,0,0),
        ('getup_left','getup',(0,0,0),0,90,0),
        ('getup_right','getup',(0,0,0),0,-90,0),
        ('pick','pick',(0,0,0),0,0,0),
        ('roulade','roulade',(0,0,0),0,0,0),
    ]


def contact_state(model, data, floor):
    touching = set()
    for contact in data.contact:
        if floor in contact.geom and contact.dist <= .001:
            touching.update(set(map(int, contact.geom)) - {floor})
    return touching


def run_case(args, case, seed, suite, cal):
    name, role, twist, pitch, roll, push = case
    model, data, motor = load_replay_model(args.voltage, bam_reference=True,
                                          repair_variant='recovery')
    if [model.joint(int(j)).name for j in motor.joint_ids] != list(JOINTS):
        raise ValueError('local MJCF joint names differ from policy')
    if not np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE):
        raise ValueError('BAM requires torque motors, not XML position servos')
    motor.kp_fw = 6.
    motor.set_gain(200)
    motor.delay = args.delay_steps
    suite.select('walk' if role in ('pick','roulade') else role, 0.)
    policy = suite.policies[suite.role]
    rng = np.random.default_rng(seed)
    pitch, roll = math.radians(pitch)/2, math.radians(roll)/2
    data.qpos[:7] = [0,0,.15,math.cos(pitch)*math.cos(roll),
                     math.cos(pitch)*math.sin(roll),math.sin(pitch)*math.cos(roll),
                     -math.sin(pitch)*math.sin(roll)]
    data.qpos[motor.qids] = policy.home + rng.uniform(-.003,.003,14)
    mujoco.mj_forward(model, data)
    data.qpos[2] += .002-floor_clearance(model, data)
    mujoco.mj_forward(model, data)
    motor.reset(data.qpos)
    feedback = SimulatedFeedback(model,data,motor,cal,args.sensor_filter=='luwu-bno')
    feet = foot_vertices(model)
    feet_ids = {g for g,_ in feet}
    floor = model.geom('floor').id
    head_body = model.body('jaw_soft').id
    imu = model.sensor_adr[model.sensor('imu_ang_vel').id]
    rows, observations, targets, records, recovery_rows = [], [], [], [], []
    inference_frames = 0
    swings = [[],[]]
    transitions = []
    target_limit_frames = encoder_saturation_frames = 0
    saturated_ids = set()
    first_fall = None
    pitch_travel = 0.
    supported_pitch = maximum_supported_pitch = 0.
    head_top_contact = False
    error = None
    renderer = writer = None
    model.geom_rgba[floor] = [.64,.79,.87,1]
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = .5
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = .7,-18,130
    if args.video:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, height=480, width=640)
        writer = imageio.get_writer(str(args.output/f'{name}_seed{seed}.mp4'), fps=25)
    prefix = args.output/f'{name}_seed{seed}'
    trace_path = prefix.with_suffix('.csv')
    try:
        with trace_path.open('w', newline='') as stream:
            trace = csv.writer(stream)
            trace.writerow(['time_s','policy','tilt_deg','z_m','vx','vy','vz','wz',
                            'left_sole_m','right_sole_m','left_contact','right_contact',
                            'other_floor_contacts','mouth_target_rad',
                            *[f'q_{n}' for n in JOINTS], *[f'target_{n}' for n in JOINTS],
                            *[f'id{i}_ticks' for i in range(1,16)]])
            for step in range(round(args.seconds*50)):
                now = step/50
                old = suite.role
                if step == 100 and role in ('pick','roulade'):
                    suite.select(role, now)
                policy, elapsed = suite.advance(now)
                if old != suite.role:
                    transitions.append(dict(time_s=now, previous=old, selected=suite.role))
                q, dq = data.qpos[motor.qids].copy(), data.qvel[motor.vids].copy()
                measured_q, measured_dq, gyro, gravity = feedback.read(args.feedback_delay_ms)
                head_command = [0,0,0,0]
                if name == 'head_mouth':
                    head_command = [0,.12*math.sin(now),.25*math.sin(now),0]
                command = twist if now >= 2 else (0,0,0)
                if role == 'getup' and now < 1.:
                    policy.reset()
                    obs, mouth = policy.observation(measured_q, measured_dq, gyro, gravity,
                                                     elapsed, command, head_command)
                    target = policy.home.copy()
                else:
                    target, mouth, obs = policy.infer(measured_q, measured_dq, gyro, gravity,
                                                     elapsed, command, head_command)
                    inference_frames += 1
                if name == 'head_mouth':
                    mouth = .25*(1+math.sin(now))
                raw_goals = cal.requested_ticks(target, mouth)
                clipped_ids = {i for i,v in raw_goals.items() if not 0 <= v <= 4095}
                encoder_saturation_frames += int(bool(clipped_ids))
                saturated_ids.update(clipped_ids)
                goal_ticks = cal.target_ticks(target, mouth, saturate=args.output_mode=='upstream-saturate')
                applied = cal.positions(goal_ticks)
                limits = model.jnt_range[motor.joint_ids]
                target_limit_frames += int(np.any((applied < limits[:,0]) | (applied > limits[:,1])))
                motor.q_target = applied
                data.xfrc_applied[head_body] = 0
                if push and 5 <= now < 5.2:
                    data.xfrc_applied[head_body,:3] = rotate(data.qpos[3:7], [push,0,0])
                pitch_travel += float(data.qvel[4])*.02
                step_control_period(model, data, motor, feedback)
                if not np.isfinite(data.qpos).all():
                    raise ValueError('nonfinite physics')
                now = (step+1)/50
                gravity = rotate(conjugate(data.qpos[3:7]), [0,0,-1])
                tilt = math.degrees(math.acos(float(np.clip(-gravity[2], -1,1))))
                velocity = rotate(conjugate(data.qpos[3:7]), data.qvel[:3])
                wz = float(data.sensordata[imu+2])
                heights = clearance(data, feet)
                touching = contact_state(model, data, floor)
                contacts = [int(g in touching) for g,_ in feet]
                other = len(touching-feet_ids)
                w,x,y,z = data.qpos[3:7]
                flat = np.clip((_FLAT_ZERO-abs(2*(y*z+w*x)))/(_FLAT_ZERO-_FLAT_FULL),0,1)
                supported_pitch += float(data.qvel[4])*.02*bool(touching)*flat*flat*(3-2*flat)
                maximum_supported_pitch = max(maximum_supported_pitch,supported_pitch)
                top_down = float(data.xmat[head_body].reshape(3,3)[2] @ np.asarray(_HEAD_TOP_AXIS)) < -_HEAD_TOP_DOWN_MIN
                head_touch = any(model.geom_bodyid[g]==head_body for g in touching)
                head_top_contact |= bool(head_touch and top_down and _HEAD_LATCH_LO<supported_pitch<_HEAD_LATCH_HI)
                if first_fall is None and (tilt>60 or data.qpos[2]<.06):
                    first_fall = now
                if role == 'walk' and first_fall is None:
                    for i in range(2):
                        swings[i].append((now, heights[i], contacts[i]))
                row = [now,tilt,float(data.qpos[2]),*velocity,wz,*heights,*contacts,other,mouth]
                rows.append(row)
                recovery_rows.append([now,tilt,float(data.qpos[2]),sum(contacts),other])
                observations.append(obs)
                targets.append(target)
                records.append([goal_ticks[i] for i in range(1,16)])
                trace.writerow([now,suite.role,*row[1:],*q,*applied,*records[-1]])
                if step % 50 == 0:
                    stream.flush()
                if renderer is not None and step%2==0:
                    camera.lookat[:] = data.qpos[:3]
                    renderer.update_scene(data,camera)
                    from PIL import Image, ImageDraw
                    frame = Image.fromarray(renderer.render())
                    ImageDraw.Draw(frame).text((12,12), f'LOCAL | {name} | {suite.role} | {now:.2f}s\n'
                        f'tilt {tilt:.1f} deg  vx {velocity[0]:.2f}  wz {wz:.2f}',
                        fill=(240,245,250), stroke_width=1, stroke_fill=(20,35,45))
                    writer.append_data(np.asarray(frame))
                if now >= 1. and suite.recovery_ready(gravity, now):
                    transitions.append(dict(time_s=now, previous='getup', selected='walk'))
    except (ValueError, RuntimeError) as exc:
        error = str(exc)
    finally:
        if renderer is not None:
            renderer.close()
            writer.close()
    np.savez_compressed(prefix.with_suffix('.npz'), observations=np.asarray(observations),
                        targets=np.asarray(targets), id_1_to_15_ticks=np.asarray(records))
    settled = np.asarray([row for row in rows if row[0]>=2])
    upright = recovery_success(recovery_rows)
    result = dict(case=name,seed=seed,role=role,frames=len(rows),inference_frames=inference_frames,error=error,
                  completed=error is None and len(rows)==round(args.seconds*50),
                  mass_kg=float(model.body_mass.sum()), voltage=args.voltage, kp_fw=6,
                  motor_delay_ms=args.delay_steps*5, observation_delay_ms=args.feedback_delay_ms,
                  feedback_hz=100, policy_hz=50, physics_hz=200, sensor_filter=args.sensor_filter,
                  output_mode=args.output_mode, encoder_saturation_frames=encoder_saturation_frames,
                  saturated_servo_ids=sorted(saturated_ids),
                  first_fall_s=first_fall if role=='walk' else None,
                  final_supported_upright=upright, transitions=transitions,
                  target_outside_mjcf_limit_frames=target_limit_frames,
                  max_encoder_roundtrip_error_rad=feedback.max_position_error,
                  max_imu_roundtrip_error=feedback.max_imu_error,
                  pitch_integral_deg=math.degrees(pitch_travel),
                  supported_pitch_deg=math.degrees(maximum_supported_pitch),head_top_contact=head_top_contact,
                  feet_before_fall=[summarize_swings(s) for s in swings],
                  mean_velocity=settled[:,3:7].mean(axis=0).tolist() if len(settled) else None,
                  max_tilt_deg=max((r[1] for r in rows),default=None),
                  task_motion_success=(upright and math.degrees(maximum_supported_pitch)>330 and head_top_contact)
                    if role=='roulade' else upright if role=='getup' else None,
                  pick_object_success=None, mouth_physics='not actuated in local 14-joint MJCF; protocol only',
                  getup_home_prelude_s=1 if role=='getup' else 0,
                  calibration_sha256=cal.sha256, calibration_unchanged=cal.unchanged())
    prefix.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,default=MODEL_DIR)
    parser.add_argument('--installation',type=Path,default=INSTALLATION)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=12)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,123])
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--voltage',type=float,default=7.4)
    parser.add_argument('--delay-steps',type=int,choices=range(3,11),default=4)
    parser.add_argument('--feedback-delay-ms',type=int,choices=(0,20,40,60),default=0)
    parser.add_argument('--output-mode',choices=('upstream-saturate','local-strict'),default='upstream-saturate')
    parser.add_argument('--sensor-filter',choices=('luwu-bno','none'),default='luwu-bno')
    parser.add_argument('--video',action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 4:
        parser.error('seconds must be finite and >=4')
    args.output.mkdir(parents=True,exist_ok=True)
    suite, cal = LuwuSuite(args.models), LocalCalibration(args.installation)
    selected = [c for c in cases() if args.cases is None or c[0] in args.cases]
    if not selected or (args.cases and set(args.cases)-{c[0] for c in selected}):
        parser.error('unknown case')
    result = dict(simulation_only=True, hardware_deployed=False,
                  source_commit='8cdbbd84710d856581982c9eaf0d5e2970666232',
                  bam_profile_path=str(PROFILE_PATH),
                  bam_profile_sha256=hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
                  source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__),Path(__file__).resolve().parents[2]/'radxa/luwu_policy.py')},
                  unavailable_published_weights=['sitstand','kick_left','kick_right'],
                  policy_hashes={r:hashlib.sha256(p.path.read_bytes()).hexdigest()
                                 for r,p in suite.policies.items()}, cases=[])
    for seed in args.seeds:
        for case in selected:
            report = run_case(args,case,seed,suite,cal)
            result['cases'].append(report)
            (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({k:report[k] for k in ('case','seed','frames','error','first_fall_s',
                                                  'final_supported_upright')}),flush=True)
    if not cal.unchanged():
        raise RuntimeError('installation file changed during simulation')


if __name__ == '__main__':
    main()
