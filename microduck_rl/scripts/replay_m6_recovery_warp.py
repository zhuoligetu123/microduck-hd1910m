#!/usr/bin/env python3
"""Training-engine cross-check of get-up, with all automatic resets disabled."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import mujoco
import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from replay_hd1910_warp import make_replay_cfg
from replay_hd1910 import load_replay_model, DEFAULT_POSE, validate_metadata
from replay_m6_recovery import floor_clearance, recovery_success


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=16.)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not math.isfinite(args.seconds) or not 12 <= args.seconds <= 120:
        p.error('duration must be within 12..120 seconds')
    session = ort.InferenceSession(str(args.policy), providers=['CPUExecutionProvider'])
    meta = session.get_modelmeta().custom_metadata_map
    if meta.get('policy_role') != 'recovery':
        raise ValueError('recovery policy required')
    cfg = make_replay_cfg(args.seconds, slew=True, bam_reference=True, delay_steps=6,
                           repair_variant='recovery_support')
    cfg.scene.num_envs = 5
    cfg.terminations = {}
    cfg.rewards = {}
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    cases = [('front',90),('back',-90),('front_partial',50),('back_partial',-50),('stand',0)]
    try:
        env.reset(seed=args.seed)
        robot = env.scene['robot']
        validate_metadata(meta, robot.joint_names, bam_reference=True)
        root = robot.data.default_root_state.clone()
        root[:, :3] = env.scene.env_origins
        root[:, 7:] = 0
        joints = np.tile(DEFAULT_POSE, (5,1)) + np.random.default_rng(args.seed).uniform(-.005,.005,(5,14))
        cpu, data, motor = load_replay_model(7.4, bam_reference=True, repair_variant='recovery')
        for i, (_, pitch) in enumerate(cases):
            a = math.radians(pitch)
            quat = [math.cos(a/2),0,math.sin(a/2),0]
            data.qpos[:7] = [0,0,.125,*quat]
            data.qpos[motor.qids] = joints[i]
            mujoco.mj_forward(cpu,data)
            root[i,2] += .125+.002-floor_clearance(cpu,data)
            root[i,3:7] = torch.tensor(quat,device=env.device)
        robot.write_root_state_to_sim(root)
        positions = torch.tensor(joints,device=env.device,dtype=torch.float32)
        robot.write_joint_state_to_sim(positions,torch.zeros_like(positions))
        for name in ('twist','head_pose','body_pose'):
            env.command_manager.get_command(name)[:] = 0
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.sim.sense()
        obs = env.observation_manager.compute(update_history=True)
        rows = [[] for _ in cases]
        with torch.inference_mode():
            for step in range(round(args.seconds*50)):
                x = obs['actor'].cpu().numpy()
                actions = np.concatenate([session.run(None,{'obs':v[None]})[0] for v in x])
                if not np.isfinite(actions).all():
                    raise ValueError('nonfinite output')
                obs,_,ended,timeout,_ = env.step(torch.from_numpy(actions).to(env.device))
                if ended.any() or timeout.any():
                    raise RuntimeError('Unexpected automatic reset in recovery evaluation')
                tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:,2],-1,1))*180/math.pi
                height = robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
                feet = env.scene['feet_ground_contact'].data.found.reshape(5,-1)
                other = env.scene['recovery_nonfeet_contact'].data.found.reshape(5,-1)
                values = torch.stack((tilt,height,(feet>0).sum(1),(other>0).sum(1)),dim=1).cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError('nonfinite simulation state')
                for i, row in enumerate(values):
                    rows[i].append([(step+1)/50,*map(float,row)])
        result = dict(engine='mujoco_warp',policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
            seconds=args.seconds,seed=args.seed,automatic_resets=0,hardware_tested=False,deployment_ready=False,
            voltage_v=7.4,actuator_delay_ms=30,collision_model='groundcontact',
            cases=[dict(case=name,initial_pitch_deg=pitch,standing_at_end=recovery_success(rows[i]),
                final_tilt_deg=rows[i][-1][1],final_height_m=rows[i][-1][2])
                for i,(name,pitch) in enumerate(cases)])
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
