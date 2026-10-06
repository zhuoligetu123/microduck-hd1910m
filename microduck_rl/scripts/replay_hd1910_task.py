#!/usr/bin/env python3
"""Replay an HD task in its OWN environment/command generator, without hardware.

This is a numerical/contract smoke check, not an assertion of task mastery.
Body-on-ground skills cannot use the walking no-fall criterion.
--posture-cycle checks a deterministic STAND -> SIT -> STAND sequence instead.
It exits 2 when the measured posture acceptance fails, still saving the report.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def posture_phase_metrics(samples, target_height):
    """Evaluate the final two seconds, not the transition's moving setpoint."""
    from statistics import mean
    if not samples:
        return dict(passed=False, reason='no settled samples')
    height, tilt, vertical_speed = zip(*samples)
    in_band = mean(abs(z - target_height) <= .015 for z in height)
    return dict(target_height_m=target_height, mean_height_m=mean(height),
                height_in_band_fraction=in_band, max_tilt_deg=max(tilt),
                mean_abs_vertical_speed_m_s=mean(abs(v) for v in vertical_speed),
                passed=in_band >= .8 and max(tilt) <= 20
                and mean(abs(v) for v in vertical_speed) <= .03)


def configure_posture_cycle(cfg, seconds):
    from mjlab.envs import mdp
    cfg.episode_length_s = seconds + 1
    cfg.events = {}
    cfg.curriculum = {}
    cfg.observations['actor'].enable_corruption = False
    for name, func in (('base_ang_vel', mdp.base_ang_vel),
                       ('projected_gravity', mdp.projected_gravity)):
        term = cfg.observations['actor'].terms[name]
        term.func, term.params = func, {}
        term.delay_min_lag = term.delay_max_lag = 0
    for command in cfg.commands.values():
        command.resampling_time_range = (1e6, 1e6)
    cfg.commands['head_pose'].ranges = ((0., 0.),) * 4
    motor = cfg.scene.entities['robot'].articulation.actuators[0]
    motor.voltage_range = (7.4, 7.4)
    motor.delay_min_lag = motor.delay_max_lag = 4


def replay_posture_cpu(args, meta):
    """Independent engine, same three phases and the same 12-replica criteria."""
    import math
    import mujoco
    import numpy as np
    from replay_hd1910 import load_replay_model, ReplayPolicy, DEFAULT_POSE, step_control_period, validate_metadata
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import STAND_Z, SIT_Z
    m6 = meta.get('actuator_backend') == 'hd1910_bam_m6'
    model,data,motor = (load_replay_model(7.4,bam_reference=True,repair_variant='sitstand')
                        if m6 else load_replay_model(7.4,posture=True))
    policy = ReplayPolicy(model,data,bam_ctrl=motor,new_cmd_obs=True,use_projected_gravity=True,
                          **({'sitstand_onnx_path':str(args.policy)} if m6 else
                             {'walking_onnx_path':str(args.policy)}))
    names = [model.joint(int(i)).name for i in motor.joint_ids]
    validate_metadata(meta,names,posture=True,bam_reference=m6)
    if policy.ort_session.get_inputs()[0].shape != [1,61] or policy.ort_session.get_outputs()[0].shape != [1,14]:
        raise ValueError('invalid posture tensor shape')
    bounds = model.jnt_range[model.actuator_trnid[:,0]]
    rng = np.random.default_rng(args.seed)
    phases = [dict(phase=name,replicas=[]) for name in ('stand','sit','stand_return')]
    max_jump, violations = 0., 0
    renderer = writer = trace_stream = None
    try:
        trace = None
        if args.trace:
            args.trace.parent.mkdir(parents=True,exist_ok=True)
            trace_stream = args.trace.open('w',newline='')
            trace = csv.writer(trace_stream)
            trace.writerow(['replica','phase','time_s','command','z_m','tilt_deg','vz_m_s',
                            *['q_'+n for n in names],*['target_'+n for n in names]])
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True,exist_ok=True)
            renderer = mujoco.Renderer(model,height=480,width=640)
            writer = imageio.get_writer(str(args.video),fps=25)
        camera = mujoco.MjvCamera()
        camera.distance,camera.elevation,camera.azimuth = .65,-15,135
        for replica in range(4):
            mujoco.mj_resetData(model,data)
            data.qpos[:7] = [0,0,.125,1,0,0,0]
            data.qpos[motor.qids] = DEFAULT_POSE+rng.uniform(-.005,.005,14)
            motor.reset(data.qpos)
            if m6:
                motor.delay = 4
            policy.last_action[:] = 0
            policy.previous_velocity = None
            previous = DEFAULT_POSE.copy()
            mujoco.mj_forward(model,data)
            for phase in range(3):
                if m6:
                    policy.sit_mode = phase == 1
                    policy._update_command()
                else:
                    policy.set_vel_cmd(float(phase == 1),0.,0.)
                samples = []
                count = round(args.seconds*50)
                for step in range(count):
                    action = policy.infer()
                    target = policy.default_pose + action*policy.action_scale
                    if not np.isfinite(target).all():
                        raise ValueError('nonfinite CPU posture target')
                    violations += int(np.any((target<bounds[:,0]-1e-6)|(target>bounds[:,1]+1e-6)))
                    max_jump = max(max_jump,float(np.max(np.abs(target-previous))))
                    previous = target.copy()
                    policy.set_position_targets(target)
                    step_control_period(model,data,motor)
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                        raise ValueError('nonfinite CPU posture state')
                    tilt = math.degrees(math.acos(float(np.clip(-policy.get_projected_gravity()[2],-1,1))))
                    if trace is not None:
                        trace.writerow([replica,phase,(phase*count+step+1)/50,float(phase==1),
                                        data.qpos[2],tilt,data.qvel[2],*data.qpos[motor.qids],*target])
                    if step >= count-100:
                        samples.append((float(data.qpos[2]),tilt,float(data.qvel[2])))
                    if writer is not None and replica == 0 and step % 2 == 0:
                        camera.lookat[:] = data.qpos[:3]
                        renderer.update_scene(data,camera=camera)
                        writer.append_data(renderer.render())
                phases[phase]['replicas'].append(posture_phase_metrics(samples,SIT_Z if phase == 1 else STAND_Z))
        contract_ok = violations == 0 and max_jump <= .10001
        result = dict(engine='mujoco_cpu',task_id=meta['task_id'],seed=args.seed,
                      actuator_backend=meta.get('actuator_backend','hd1910_reference_pd'),
                      seconds=args.seconds*3,environments=4,phases=phases,
                      policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                      profile_sha256=meta['calibration_sha256'],
                      numerical_replay_passed=True,action_contract_passed=contract_ok,
                      hard_target_limit_frames=violations,max_target_jump_rad=max_jump,
                      task_success_evaluated=True,task_success=contract_ok and all(
                          r['passed'] for p in phases for r in p['replicas']),
                      deployment_ready=False,hardware_opened=False)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result))
        return 0 if result['task_success'] else 2
    finally:
        if trace_stream is not None:
            trace_stream.close()
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--video', type=Path, help='Optional 25 fps RGB replay of environment zero')
    parser.add_argument('--trace', type=Path, help='Per-step posture and targets; Warp also records weighted rewards')
    parser.add_argument('--engine',choices=('cpu','warp'),default='warp')
    parser.add_argument('--posture-cycle', action='store_true',
                        help='SitStand only: --seconds per phase, minimum 4 s')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error('seconds must be within 1..300')
    if args.posture_cycle and args.seconds < 4:
        parser.error('posture phases need >=4 s for ramp and settled measurement')
    if args.engine == 'cpu' and not args.posture_cycle:
        parser.error('CPU task replay currently requires posture-cycle')
    import numpy as np
    import onnxruntime as ort
    import torch
    session = ort.InferenceSession(str(args.policy), providers=['CPUExecutionProvider'])
    meta = session.get_modelmeta().custom_metadata_map
    profile = args.policy.parent / 'motor_calibration.json'
    if hashlib.sha256(profile.read_bytes()).hexdigest() != meta['calibration_sha256']:
        raise ValueError('policy/profile digest mismatch')
    if meta.get('actuator_backend') == 'hd1910_bam_m6':
        os.environ['MICRODUCK_BAM_PROFILE'] = str(profile.resolve())
        os.environ['MICRODUCK_BAM_KP'] = meta['kp_fw']
        if args.engine != 'cpu':
            parser.error('M6 SitStand replay currently supports the independent CPU engine only')
    else:
        os.environ['MICRODUCK_HD1910_REFERENCE'] = str(profile.resolve())
        os.environ['MICRODUCK_HD1910_SUITE'] = '1'
    from mjlab_microduck.tasks.hd1910_bam import SITSTAND_TASK_ID
    if args.posture_cycle and meta.get('task_id') not in (
            'Mjlab-SitStand-Flat-MicroDuck-HD1910-Reference-Slew',
            'Mjlab-SitStand-Flat-MicroDuck-HD1910-Reference-Slew-Refine',
            'Mjlab-SitStand-Flat-MicroDuck-HD1910-Reference-Slew-Balanced',
            SITSTAND_TASK_ID):
        parser.error('posture-cycle requires the bounded SitStand task, not a walking model')
    if args.engine == 'cpu':
        return replay_posture_cpu(args,meta)
    import mjlab_microduck.tasks
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    cfg = load_env_cfg(meta['task_id'], play=True)
    cfg.scene.num_envs = 4
    total_seconds = args.seconds * (3 if args.posture_cycle else 1)
    if args.posture_cycle:
        configure_posture_cycle(cfg, total_seconds)
    if args.video:
        cfg.viewer.width, cfg.viewer.height = 640, 480
        cfg.viewer.distance, cfg.viewer.elevation = .65, -15.
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode='rgb_array' if args.video else None)
    writer = trace_stream = None
    resets = 0
    limit_frames = 0
    samples = []
    phase_samples = [[[] for _ in range(4)] for _ in range(3)]
    hard_limit_frames = 0
    max_target_jump = 0.
    onnx_limit_frames = 0
    onnx_max_target_jump = 0.
    try:
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True,exist_ok=True)
            writer = imageio.get_writer(str(args.video),fps=25)
        obs, _ = env.reset(seed=args.seed)
        robot = env.scene['robot']
        servo_ids = [i for i,n in enumerate(robot.joint_names) if not n.startswith('passive_')]
        if len(servo_ids) != 14:
            raise ValueError('expected 14 policy joints')
        if meta.get('joint_names', '').split(',') != [robot.joint_names[i] for i in servo_ids]:
            raise ValueError('policy/environment joint order mismatch')
        if args.posture_cycle:
            from replay_hd1910 import validate_metadata
            validate_metadata(meta,[robot.joint_names[i] for i in servo_ids],posture=True)
        joint_limits = robot.data.joint_pos_limits[:,servo_ids,:].cpu().numpy()
        previous_target = robot.data.default_joint_pos[:,servo_ids].cpu().numpy().copy()
        home = previous_target.copy()
        previous_onnx_target = home.copy()
        trace = None
        if args.trace:
            args.trace.parent.mkdir(parents=True,exist_ok=True)
            trace_stream = args.trace.open('w',newline='')
            trace = csv.writer(trace_stream)
            trace.writerow(['replica','phase','time_s','command','alpha','z_m','tilt_deg','vz_m_s',
                            *['q_'+robot.joint_names[i] for i in servo_ids],
                            *['target_'+robot.joint_names[i] for i in servo_ids],
                            *['reward_'+n for n in env.reward_manager.active_terms]])
        if args.posture_cycle:
            root = robot.data.default_root_state.clone()
            root[:,:3] = env.scene.env_origins + torch.tensor([0.,0.,.125],device=env.device)
            root[:,3:7] = torch.tensor([1.,0.,0.,0.],device=env.device)
            root[:,7:] = 0
            robot.write_root_state_to_sim(root)
            rng = np.random.default_rng(args.seed)
            pos = robot.data.default_joint_pos + torch.tensor(
                rng.uniform(-.005,.005,robot.data.default_joint_pos.shape),
                device=env.device,dtype=torch.float32)
            robot.write_joint_state_to_sim(pos,torch.zeros_like(pos))
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.sim.sense()
        phase_steps = round(args.seconds / env.step_dt)
        for step in range(round(total_seconds / env.step_dt)):
            phase = min(step // phase_steps, 2)
            if args.posture_cycle:
                env.command_manager.get_command('twist')[:] = 0.
                env.command_manager.get_command('twist')[:,0] = float(phase == 1)
                obs = env.observation_manager.compute(update_history=False)
            x = obs['actor'].cpu().numpy()
            if x.shape != (4,61) or not np.isfinite(x).all():
                raise ValueError('invalid observation contract')
            actions = np.concatenate([session.run(None,{session.get_inputs()[0].name:row[None]})[0] for row in x])
            if actions.shape != (4,14) or not np.isfinite(actions).all():
                raise ValueError('invalid action contract')
            onnx_target = home + actions
            onnx_limit_frames += int(np.any((onnx_target < joint_limits[:,:,0]-1e-6)
                                            | (onnx_target > joint_limits[:,:,1]+1e-6),axis=1).sum())
            onnx_max_target_jump = max(onnx_max_target_jump,
                                      float(np.max(np.abs(onnx_target - previous_onnx_target))))
            previous_onnx_target = onnx_target.copy()
            obs, reward, done, timeout, extra = env.step(torch.from_numpy(actions).to(env.device))
            if writer is not None and step % 2 == 0:
                writer.append_data(env.render())
            if not torch.isfinite(reward).all() or not torch.isfinite(robot.data.joint_pos).all():
                raise ValueError('nonfinite simulation state')
            resets += int((done | timeout).sum())
            target = env.action_manager.get_term('joint_pos')._processed_actions
            limits = robot.data.soft_joint_pos_limits[:,servo_ids,:]
            limit_frames += int(((target < limits[:,:,0]) | (target > limits[:,:,1])).any(dim=1).sum())
            target_np = target.cpu().numpy()
            hard_limit_frames += int(np.any((target_np < joint_limits[:,:,0]-1e-6)
                                            | (target_np > joint_limits[:,:,1]+1e-6),axis=1).sum())
            max_target_jump = max(max_target_jump,float(np.max(np.abs(target_np - previous_target))))
            previous_target = target_np.copy()
            if trace is not None:
                tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:,2],-1,1))*180/np.pi
                command = env.command_manager.get_term('twist')
                for i in range(4):
                    trace.writerow([i,phase,(step+1)*env.step_dt,float(command.command[i,0]),
                        float(command.alpha[i]) if hasattr(command,'alpha') else '',
                        float(robot.data.root_link_pos_w[i,2]-env.scene.env_origins[i,2]),
                        float(tilt[i]),float(robot.data.root_link_lin_vel_w[i,2]),
                        *robot.data.joint_pos[i,servo_ids].cpu().tolist(),*target_np[i],
                        *[v[0] for _,v in env.reward_manager.get_active_iterable_terms(i)]])
            if args.posture_cycle and step % phase_steps >= phase_steps - round(2 / env.step_dt):
                tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:,2],-1,1))*180/np.pi
                for i in range(4):
                    phase_samples[phase][i].append((
                        float(robot.data.root_link_pos_w[i,2] - env.scene.env_origins[i,2]),
                        float(tilt[i]),float(robot.data.root_link_lin_vel_w[i,2])))
            if step % 50 == 0:
                samples.append(dict(t=step*env.step_dt, command=env.command_manager.get_command('twist')[0].cpu().tolist(),
                    height=float(robot.data.root_link_pos_w[0,2]),
                    projected_gravity=robot.data.projected_gravity_b[0].cpu().tolist(),
                    reward=float(reward.mean())))
        metrics = {k: float(v) for k,v in extra.get('log',{}).items() if np.size(v.cpu().numpy() if torch.is_tensor(v) else v) == 1}
        result = dict(engine='mujoco_warp',task_id=meta['task_id'], seconds=total_seconds, seed=args.seed, environments=4,
                      policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                      profile_sha256=meta['calibration_sha256'], numerical_replay_passed=True,
                      resets=resets, soft_target_limit_frames=limit_frames, samples=samples,
                      hard_target_limit_frames=hard_limit_frames, max_target_jump_rad=max_target_jump,
                      onnx_target_limit_frames=onnx_limit_frames, onnx_max_target_jump_rad=onnx_max_target_jump,
                      metrics=metrics, task_success_evaluated=False, deployment_ready=False,
                      hardware_opened=False)
        if args.posture_cycle:
            command_cfg = cfg.commands['twist']
            phases = []
            for phase, name in enumerate(('stand', 'sit', 'stand_return')):
                target_height = command_cfg.sit_z if phase == 1 else command_cfg.stand_z
                phases.append(dict(phase=name, replicas=[posture_phase_metrics(s, target_height)
                                                         for s in phase_samples[phase]]))
            contract_ok = hard_limit_frames == onnx_limit_frames == 0 and max_target_jump <= .10001 and onnx_max_target_jump <= .10001
            result.update(task_success_evaluated=True, phases=phases,
                          action_contract_passed=contract_ok,
                          task_success=resets == 0 and contract_ok
                          and all(r['passed'] for p in phases for r in p['replicas']))
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('samples','metrics')}))
        return 2 if args.posture_cycle and not result['task_success'] else 0
    finally:
        if trace_stream is not None:
            trace_stream.close()
        if writer is not None:
            writer.close()
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
