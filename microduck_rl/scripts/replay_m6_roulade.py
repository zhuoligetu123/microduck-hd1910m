#!/usr/bin/env python3
"""CPU M6 rollout from upright: supported forward roll, head-top contact, landing.

No curriculum mid-roll initialization or automatic resets. Success is simulator
evidence only; contact loads are not a hardware qualification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from replay_hd1910 import DEFAULT_POSE, ReplayPolicy, load_replay_model, step_control_period, validate_metadata
from replay_m6_recovery import floor_clearance, recovery_success
from mjlab_microduck.tasks.mdp import _HEAD_TOP_AXIS, _HEAD_TOP_DOWN_MIN, _HEAD_LATCH_LO, _HEAD_LATCH_HI, _FLAT_FULL, _FLAT_ZERO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=12)
    args = parser.parse_args()
    if not 4 <= args.seconds <= 60:
        parser.error('seconds must be 4..60')
    model,data,motor = load_replay_model(7.4,bam_reference=True,repair_variant='recovery')
    policy = ReplayPolicy(model,data,walking_onnx_path=str(args.policy),bam_ctrl=motor,
                          new_cmd_obs=True,use_projected_gravity=True)
    validate_metadata(policy.ort_session.get_modelmeta().custom_metadata_map,
        [model.joint(int(i)).name for i in motor.joint_ids],bam_reference=True,roulade=True)
    floor=model.geom('floor').id
    feet={model.geom('left_foot_collision').id,model.geom('right_foot_collision').id}
    head=model.body('jaw_soft').id
    cases=[]
    for seed in (42,7,123):
        rng=np.random.default_rng(seed)
        mujoco.mj_resetData(model,data)
        data.qpos[:7]=[0,0,.125,1,0,0,0]
        data.qpos[motor.qids]=DEFAULT_POSE+rng.uniform(-.005,.005,14)
        mujoco.mj_forward(model,data)
        data.qpos[2]+=.002-floor_clearance(model,data)
        mujoco.mj_forward(model,data)
        motor.reset(data.qpos)
        motor.delay=4
        policy.last_action[:]=0
        policy.previous_velocity=None
        policy.set_vel_cmd(0,0,0)
        accumulated=maximum=0.
        latched=False
        rows=[]
        for step in range(round(args.seconds*50)):
            action=policy.infer()
            if not np.isfinite(action).all():
                raise ValueError('nonfinite action')
            policy.set_position_targets(policy.default_pose+action*policy.action_scale)
            step_control_period(model,data,motor)
            touching=set()
            head_contact=False
            for contact in data.contact:
                pair=set(map(int,contact.geom))
                if floor in pair and contact.dist<=.001:
                    touching.update(pair-{floor})
                    head_contact |= any(model.geom_bodyid[g]==head for g in pair-{floor})
            w,x,y,z=data.qpos[3:7]
            flat=np.clip((_FLAT_ZERO-abs(2*(y*z+w*x)))/(_FLAT_ZERO-_FLAT_FULL),0,1)
            accumulated+=float(data.qvel[4])*.02*bool(touching)*flat*flat*(3-2*flat)
            maximum=max(maximum,accumulated)
            top_down=float(data.xmat[head].reshape(3,3)[2]@np.asarray(_HEAD_TOP_AXIS)) < -_HEAD_TOP_DOWN_MIN
            latched |= bool(head_contact and top_down and _HEAD_LATCH_LO<accumulated<_HEAD_LATCH_HI)
            tilt=math.degrees(math.acos(float(np.clip(-policy.get_projected_gravity()[2],-1,1))))
            rows.append([(step+1)/50,tilt,float(data.qpos[2]),len(touching&feet),len(touching-feet)])
        landed=recovery_success(rows)
        cases.append(dict(seed=seed,supported_pitch_deg=math.degrees(maximum),head_top_contact=latched,
            landed=landed,success=bool(maximum>math.radians(330) and latched and landed),
            final_tilt_deg=rows[-1][1],final_height_m=rows[-1][2]))
    report=dict(policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),cases=cases,
        upright_starts=True,automatic_resets=0,hardware_tested=False,deployment_ready=False)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
