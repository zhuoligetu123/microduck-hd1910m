#!/usr/bin/env python3
"""Compare CPU/Warp open-loop physics without policy feedback or hardware I/O."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from replay_hd1910 import load_replay_model
from replay_hd1910_warp import make_replay_cfg


def compare(height):
    cfg=make_replay_cfg(1)
    cfg.scene.num_envs=1
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
    try:
        env.reset(seed=42)
        robot=env.scene['robot']
        robot.reset()
        root=robot.data.default_root_state.clone()
        root[:,:3]=env.scene.env_origins
        root[:,2]=height
        root[:,3:7]=torch.tensor([1.,0.,0.,0.],device=env.device)
        root[:,7:]=0
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos,torch.zeros_like(robot.data.default_joint_pos))
        env.sim.forward()
        model,data,motor=load_replay_model(7.4)
        data.qpos[:]=env.sim.data.qpos[0].cpu().numpy()
        data.qvel[:]=env.sim.data.qvel[0].cpu().numpy()
        motor.reset(data.qpos)
        mujoco.mj_forward(model,data)
        rows=[]
        for step in range(40):
            action=torch.full((1,14),.05*np.sin(step*.1),device=env.device)
            env.action_manager.process_action(action)
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            motor.q_target[:]=robot.data.default_joint_pos[0].cpu().numpy()+action[0].cpu().numpy()
            motor.update()
            effort_error=float(np.abs(data.ctrl-env.sim.data.ctrl[0].cpu().numpy()).max())
            env.sim.step()
            env.scene.update(dt=.005)
            mujoco.mj_step(model,data)
            rows.append(dict(time_s=(step+1)*.005,effort_error_nm=effort_error,
                             qpos_max_error=float(np.abs(data.qpos-env.sim.data.qpos[0].cpu().numpy()).max()),
                             qvel_max_error=float(np.abs(data.qvel-env.sim.data.qvel[0].cpu().numpy()).max()),
                             cpu_contact_candidates=int(data.ncon),warp_contact_candidates=int(env.sim.data.nacon[0])))
        return dict(initial_height_m=height,rows=rows)
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    result=dict(hardware_tested=False,physics_hz=200,
                note='Qpos/qvel maxima mix root translation and joint angles; diagnostic, not gait acceptance.',
                cases=[compare(.125),compare(1.)])
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for case in result['cases']:
        print(case['initial_height_m'],case['rows'][-1])


if __name__=='__main__':
    main()
