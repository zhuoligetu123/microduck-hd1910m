#!/usr/bin/env python3
"""Deterministic training-engine counterpart to CPU replay; no hardware I/O."""
import argparse
import hashlib
import json
from pathlib import Path

import mjlab
import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv, mdp
from mjlab_microduck.tasks.microduck_hd1910_env_cfg import make_reference_hd1910_velocity_env_cfg, make_bounded_hd1910_velocity_env_cfg, make_slew_hd1910_velocity_env_cfg
from replay_hd1910 import validate_metadata,replay_cases,baseline_check,target_chatter_metrics,MAX_SATURATED_REVERSAL_FRACTION,head_center_metrics


def make_replay_cfg(seconds, bounded=False, slew=False, max_step_rad=.10, voltage=7.4, delay_steps=4, bam_reference=False,
                    repair_variant=None, joint_age_steps=0, imu_age_steps=0):
    if not np.isfinite(voltage) or not 4.8 <= voltage <= 8.4 or delay_steps not in range(3,7):
        raise ValueError('stress replay requires voltage 4.8..8.4 V and delay 3..6 steps')
    if joint_age_steps not in range(9) or imu_age_steps not in range(5):
        raise ValueError('feedback ages must be integer control steps: joint 0..8, IMU 0..4')
    factory = make_bounded_hd1910_velocity_env_cfg if bounded else make_reference_hd1910_velocity_env_cfg
    if slew:
        factory = make_slew_hd1910_velocity_env_cfg
    if bam_reference:
        from mjlab_microduck.tasks.hd1910_bam import make_xgo_bam_env_cfg
        from functools import partial
        factory = partial(make_xgo_bam_env_cfg, repair_variant=repair_variant)
    cfg=factory(play=True)
    if slew:
        cfg.actions['joint_pos'].max_step_rad=max_step_rad
    cfg.scene.num_envs=3
    cfg.episode_length_s=seconds+1
    cfg.events={k:v for k,v in cfg.events.items() if bam_reference and k == 'expand_bam_friction_fields'}
    cfg.curriculum={}
    cfg.observations['actor'].enable_corruption=False
    for name,func in (('base_ang_vel',mdp.base_ang_vel),('projected_gravity',mdp.projected_gravity)):
        term=cfg.observations['actor'].terms[name]
        term.func=func
        term.params={}
        term.delay_min_lag=term.delay_max_lag=imu_age_steps
        term.delay_hold_prob=0.
    for name in ('joint_state', 'joint_pos', 'joint_vel'):
        if name not in cfg.observations['actor'].terms:
            continue
        term=cfg.observations['actor'].terms[name]
        term.delay_min_lag=term.delay_max_lag=joint_age_steps
        term.delay_hold_prob=0.
        if name == 'joint_state':
            term.params.update(biased=False, training_noise=False)
    motor=cfg.scene.entities['robot'].articulation.actuators[0]
    if bam_reference:
        if not 7.0 <= voltage <= 8.0:
            raise ValueError('M6 replay supports 7.0..8.0 V')
        motor.vin_range=(voltage,voltage)
        motor.vin_drop_gain_range=(0.,0.)
    else:
        motor.voltage_range=(voltage,voltage)
    motor.delay_min_lag=motor.delay_max_lag=delay_steps
    for command in cfg.commands.values():
        command.resampling_time_range=(1e6,1e6)
    cfg.commands['twist'].rel_standing_envs=0
    cfg.commands['twist'].rel_turn_in_place_envs=0
    cfg.commands['twist'].rel_world_envs=0
    cfg.commands['head_pose'].ranges=((0.,0.),)*4
    cfg.commands['body_pose'].ranges=((0.,0.),)*6
    return cfg


def policy_recipe(metadata, bam_reference):
    if not bam_reference:
        return None
    if metadata.get('command_semantics', 'twist_head_body') != 'twist_head_body':
        raise ValueError('velocity replay cannot execute phase/posture/roll commands')
    recipe=metadata.get('training_recipe', 'm6_reference_v1')
    if recipe == 'm6_reference_v1':
        return None
    if recipe.startswith('m6_repair_') and recipe.endswith('_v1'):
        return recipe[len('m6_repair_'):-len('_v1')]
    raise ValueError(f'unsupported BAM replay recipe: {recipe}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--bam-reference',action='store_true')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=20)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--extended',action='store_true')
    parser.add_argument('--voltage',type=float,default=7.4)
    parser.add_argument('--delay-steps',type=int,choices=(3,4,5,6),default=4)
    parser.add_argument('--joint-age-steps',type=int,choices=range(9),default=0)
    parser.add_argument('--imu-age-steps',type=int,choices=range(5),default=0)
    parser.add_argument('--initial-tilt-deg',type=float,default=0.)
    args=parser.parse_args()
    if not np.isfinite(args.seconds) or not .02<=args.seconds<=300:
        parser.error('seconds must be within 0.02..300')
    if not np.isfinite(args.initial_tilt_deg) or not 0 <= args.initial_tilt_deg <= 10:
        parser.error('initial tilt must be within 0..10 degrees')
    session=ort.InferenceSession(str(args.policy),providers=['CPUExecutionProvider'])
    metadata=session.get_modelmeta().custom_metadata_map
    semantics = metadata.get('action_semantics')
    variant=policy_recipe(metadata,args.bam_reference)
    cfg=make_replay_cfg(args.seconds, semantics == 'bounded_home_delta_v1', semantics == 'bounded_slew_home_delta_v2',
                        float(session.get_modelmeta().custom_metadata_map.get('max_action_step_rad','.10')),
                        args.voltage, args.delay_steps, args.bam_reference, variant,
                        args.joint_age_steps, args.imu_age_steps)
    cases=replay_cases(args.extended)
    cfg.scene.num_envs=len(cases)
    commands=torch.tensor([command for _,command in cases],device='cuda:0')
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
    try:
        env.reset(seed=args.seed)
        robot=env.scene['robot']
        digest=validate_metadata(session.get_modelmeta().custom_metadata_map,robot.joint_names,bam_reference=args.bam_reference)
        root=robot.data.default_root_state.clone()
        root[:,:3]=env.scene.env_origins+torch.tensor([0.,0.,.125],device=env.device)
        root[:,3:7]=torch.tensor([1.,0.,0.,0.],device=env.device)
        root[:,7:]=0
        rng=np.random.default_rng(args.seed)
        offsets=[]
        for i in range(len(cases)):
            if args.initial_tilt_deg:
                roll,pitch=np.radians(rng.uniform(-args.initial_tilt_deg,args.initial_tilt_deg,2))
                root[i,3:7]=torch.tensor([np.cos(roll/2)*np.cos(pitch/2),
                    np.sin(roll/2)*np.cos(pitch/2),np.cos(roll/2)*np.sin(pitch/2),
                    -np.sin(roll/2)*np.sin(pitch/2)],device=env.device)
            offsets.append(rng.uniform(-.005,.005,14))
        robot.write_root_state_to_sim(root)
        pos=robot.data.default_joint_pos+torch.tensor(np.asarray(offsets),device=env.device,dtype=torch.float32)
        robot.write_joint_state_to_sim(pos,torch.zeros_like(pos))
        env.command_manager.get_command('twist')[:]=commands
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.sim.sense()
        obs=env.observation_manager.compute(update_history=True)
        initial_observations=obs['actor'].cpu().tolist()
        falls=[None]*len(cases)
        maximum=np.zeros(len(cases))
        target_violations=np.zeros(len(cases),dtype=int)
        initial_target_jump=None
        previous_action=np.zeros((len(cases),14),dtype=np.float32)
        max_target_jump=np.zeros(len(cases))
        joint_limits=env.sim.mj_model.jnt_range[env.sim.mj_model.actuator_trnid[:,0]]
        home=robot.data.default_joint_pos.cpu().numpy()
        samples=[]
        head_errors=[]
        target_steps=[]
        geom_local=robot.find_geoms(['left_foot_collision','right_foot_collision'],preserve_order=True)[0]
        foot_geoms=[int(robot.data.indexing.geom_ids[i]) for i in geom_local]
        foot_vertices=[]
        for gid in foot_geoms:
            mesh=env.sim.mj_model.geom_dataid[gid]
            start=env.sim.mj_model.mesh_vertadr[mesh]
            count=env.sim.mj_model.mesh_vertnum[mesh]
            foot_vertices.append(torch.tensor(env.sim.mj_model.mesh_vert[start:start+count].copy(),device=env.device))
        swing_peaks=np.zeros((len(cases),2))
        airborne=np.zeros((len(cases),2),dtype=bool)
        completed_peaks=[[[],[]] for _ in cases]
        with torch.inference_mode():
            for step in range(round(args.seconds*50)):
                x=obs['actor'].cpu().numpy()
                actions=np.concatenate([session.run(None,{session.get_inputs()[0].name:row[None]})[0] for row in x])
                if not np.isfinite(actions).all():
                    raise ValueError('nonfinite policy output')
                targets=home+actions
                target_violations+=np.any((targets<joint_limits[:,0]-1e-6)|(targets>joint_limits[:,1]+1e-6),axis=1)
                if initial_target_jump is None:
                    initial_target_jump=np.max(np.abs(actions),axis=1)
                max_target_jump=np.maximum(max_target_jump,np.max(np.abs(actions-previous_action),axis=1))
                if step>=50:
                    target_steps.append((actions-previous_action).copy())
                previous_action=actions.copy()
                obs,_,terminated,_,_=env.step(torch.from_numpy(actions).to(env.device))
                tilt=torch.acos(torch.clamp(-robot.data.projected_gravity_b[:,2],-1,1)).cpu().numpy()*180/np.pi
                maximum=np.maximum(maximum,tilt)
                for i,ended in enumerate(terminated.cpu().tolist()):
                    if ended and falls[i] is None:
                        falls[i]=(step+1)/50
                if step>=50:
                    samples.append(torch.cat((robot.data.root_link_lin_vel_b[:,:2],robot.data.root_link_ang_vel_b[:,2:3]),dim=1).cpu().numpy())
                    head_errors.append((robot.data.joint_pos[:,5:9]-robot.data.default_joint_pos[:,5:9]
                                        -env.command_manager.get_command('head_pose')).cpu().numpy())
                    geometry=robot.data.data
                    clearance=torch.stack([
                        (geometry.geom_xmat[:,gid].reshape(-1,3,3)[:,2,:] @ vertices.T).min(dim=1).values
                        +geometry.geom_xpos[:,gid,2]-env.scene.env_origins[:,2]
                        for gid,vertices in zip(foot_geoms,foot_vertices)],dim=1).cpu().numpy()
                    contact=env.scene['feet_ground_contact'].data.found.cpu().numpy()!=0
                    for i in range(len(cases)):
                        if falls[i] is not None:
                            continue
                        for foot in range(2):
                            if not contact[i,foot]:
                                airborne[i,foot]=True
                                swing_peaks[i,foot]=max(swing_peaks[i,foot],clearance[i,foot])
                            elif airborne[i,foot]:
                                completed_peaks[i][foot].append(float(swing_peaks[i,foot]))
                                swing_peaks[i,foot]=0.
                                airborne[i,foot]=False
        rows=[]
        for i,(name,command) in enumerate(cases):
            velocity=np.asarray(samples)[:,i,:] if samples else None
            row=dict(case=name,command=command,first_fall_s=falls[i],max_tilt_deg=float(maximum[i]),
                     target_limit_violations=int(target_violations[i]),
                     initial_target_jump_rad=float(initial_target_jump[i]),
                     max_target_jump_rad=float(max_target_jump[i]),
                     completed=True,no_fall=falls[i] is None,
                     mean_body_velocity_after_1s=velocity.mean(axis=0).tolist() if samples else None,
                     rms_vx_error_after_1s=float(np.sqrt(np.mean((velocity[:,0]-command[0])**2))) if samples else None,
                     rms_yaw_error_after_1s=float(np.sqrt(np.mean((velocity[:,2]-command[2])**2))) if samples else None)
            row['foot_swing_peak_median_m']=[float(np.median(x)) if x else None for x in completed_peaks[i]]
            row['foot_landing_count']=[len(x) for x in completed_peaks[i]]
            row['baseline_check_passed']=baseline_check(row,args.seconds)
            row.update(head_center_metrics(np.asarray(head_errors)[:,i,:] if head_errors else []))
            row.update(target_chatter_metrics(np.asarray(target_steps)[:,i,:] if target_steps else [],
                float(session.get_modelmeta().custom_metadata_map['max_action_step_rad']) if semantics=='bounded_slew_home_delta_v2' else None))
            rows.append(row)
        result=dict(engine='mujoco_warp',profile_sha256=digest,policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                    action_semantics=session.get_modelmeta().custom_metadata_map.get('action_semantics','raw_home_delta'),
                    max_action_step_rad=float(session.get_modelmeta().custom_metadata_map['max_action_step_rad']) if semantics=='bounded_slew_home_delta_v2' else None,
                    seconds=args.seconds,seed=args.seed,hardware_tested=False,deployment_ready=False,gait_accepted=False,
                    voltage_v=args.voltage,delay_physics_steps=args.delay_steps,
                    actuator_delay_ms=args.delay_steps*5,initial_tilt_range_deg=args.initial_tilt_deg,
                    training_recipe=metadata.get('training_recipe'),
                    joint_observation_delay_ms=args.joint_age_steps*20,
                    imu_observation_delay_ms=args.imu_age_steps*20,
                    policy_hz=50,physics_hz=200,
                    extended_cases=args.extended,baseline_checks_passed=all(r['baseline_check_passed'] and r['motion_quality_check_passed'] for r in rows),
                    motion_quality_criteria=dict(saturated_step_ratio=.99,
                        max_saturated_reversal_fraction_per_joint=MAX_SATURATED_REVERSAL_FRACTION),
                    initial_observations=initial_observations,
                    final_commands=env.command_manager.get_command('twist').cpu().tolist(),
                    cases=rows)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        env.close()


if __name__=='__main__':
    main()
