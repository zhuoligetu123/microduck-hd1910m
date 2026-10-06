#!/usr/bin/env python3
"""Measure low-obstacle exposure in MuJoCo Warp, without hardware or promotion.

Reports obstacle contact and survival, not plane-relative foot height masquerading
as clearance over uneven ground. Policy observations stay 61 dimensional.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains.terrain_generator import TerrainGenerator
from mjlab_microduck.tasks.xgoduck_bam import make_xgo_bam_env_cfg, configure_terrain_course
from replay_hd1910 import replay_cases, validate_metadata
from replay_hd1910_warp import make_replay_cfg, policy_recipe


def obstacle_approach(model, data, origin, direction):
    """Place moving trials before a real obstacle, not a random empty corridor."""
    tops = model.geom_pos[:, 2] + model.geom_size[:, 2]
    candidates = [i for i in np.flatnonzero(tops > 1e-8)
                  if np.max(np.abs(model.geom_pos[i, :2] - origin[:2])) < .75]
    candidates.sort(key=lambda i: np.linalg.norm(model.geom_pos[i, :2] - origin[:2]))
    for gid in candidates:
        xy = model.geom_pos[gid, :2].copy()
        xy[0] -= direction * (model.geom_size[gid, 0] + .12)
        clear = True
        for dx in (-.06, 0., .06):
            for dy in (-.08, 0., .08):
                start = np.array([xy[0] + dx, xy[1] + dy, .3])
                distance = mujoco.mj_ray(model, data, start, np.array([0., 0., -1.]),
                                         None, 1, -1, np.array([-1], dtype=np.int32))
                clear &= distance >= 0. and abs(.3 - distance) < 1e-7
        if clear:
            return xy, model.geom(gid).name
    raise ValueError('no clear obstacle approach on this terrain patch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--terrain-course', choices=('microblocks', 'microblocks12'), default='microblocks')
    args = parser.parse_args()
    if not 1. <= args.seconds <= 60.:
        parser.error('seconds must be within 1..60')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(args.policy), sess_options=options,
                                  providers=['CPUExecutionProvider'])
    metadata = session.get_modelmeta().custom_metadata_map
    variant = policy_recipe(metadata, True)
    template = make_xgo_bam_env_cfg(repair_variant=variant)
    configure_terrain_course(template, args.terrain_course)
    generator = TerrainGenerator(template.scene.terrain.terrain_generator)
    spec = mujoco.MjSpec()
    generator.compile(spec)
    terrain_model = spec.compile()
    terrain_data = mujoco.MjData(terrain_model)
    mujoco.mj_forward(terrain_model, terrain_data)
    tops = terrain_model.geom_pos[:, 2] + terrain_model.geom_size[:, 2]
    obstacles = tuple(terrain_model.geom(i).name for i in np.flatnonzero(tops > 1e-8))
    cfg = make_replay_cfg(args.seconds, slew=True, bam_reference=True,
                          repair_variant=variant, joint_age_steps=1, imu_age_steps=1)
    cfg.scene.terrain = template.scene.terrain
    cfg.scene.num_envs = 10
    # The vector engine requires resetting terminated worlds to advance peers.
    # Permanently exclude each world after its first fall from contact metrics.
    cfg.auto_reset = True
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name='low_obstacle_contact',
        primary=ContactMatch(mode='geom', pattern=obstacles),
        secondary=ContactMatch(mode='subtree', entity='robot', pattern='trunk_base'),
        secondary_policy='error', fields=('found',), reduce='netforce', num_slots=1))
    cases = replay_cases(True) * 2
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    try:
        env.reset(seed=args.seed)
        robot = env.scene['robot']
        validate_metadata(metadata, robot.joint_names, bam_reference=True)
        root = robot.data.default_root_state.clone()
        root[:, :3] = env.scene.env_origins + torch.tensor([0., 0., .125], device=env.device)
        root[:, 3:7] = torch.tensor([1., 0., 0., 0.], device=env.device)
        root[:, 7:] = 0.
        types = env.scene.terrain.terrain_types.cpu().tolist()
        approaches = [None] * len(cases)
        for i, (_, command) in enumerate(cases):
            if types[i] == 1 and command[0] != 0.:
                xy, name = obstacle_approach(terrain_model, terrain_data,
                    env.scene.env_origins[i].cpu().numpy(), np.sign(command[0]))
                root[i, :2] = torch.tensor(xy, device=env.device)
                approaches[i] = dict(obstacle=name, start_xy=xy.tolist())
        robot.write_root_state_to_sim(root)
        rng = np.random.default_rng(args.seed)
        position = robot.data.default_joint_pos + torch.tensor(
            rng.uniform(-.005, .005, (len(cases), 14)), device=env.device, dtype=torch.float32)
        robot.write_joint_state_to_sim(position,
                                       torch.zeros_like(robot.data.default_joint_pos))
        commands = torch.tensor([c for _, c in cases], device=env.device)
        env.command_manager.get_command('twist')[:] = commands
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.sim.sense()
        obs = env.observation_manager.compute(update_history=True)
        start = robot.data.root_link_pos_w.clone()
        hits = np.zeros(10, dtype=int)
        stopped = np.zeros(10, dtype=bool)
        first_fall = [None] * 10
        first_end = [None] * 10
        with torch.inference_mode():
            for step in range(round(args.seconds * 50)):
                actions = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0]
                                          for row in obs['actor'].cpu().numpy()])
                if not np.isfinite(actions).all():
                    raise ValueError('nonfinite policy output')
                obs, _, ended, truncated, _ = env.step(torch.from_numpy(actions).to(env.device))
                tilt = -robot.data.projected_gravity_b[:, 2]
                falling = (ended | (tilt < .5)).cpu().numpy()
                truncation = truncated.cpu().numpy()
                now = falling | truncation
                for i in np.flatnonzero(now & ~stopped):
                    if falling[i]:
                        first_fall[i] = (step + 1) / 50.
                    first_end[i] = dict(time_s=(step + 1) / 50.,
                                        reason='fall' if falling[i] else 'timeout_or_bounds')
                stopped |= now
                contact = (env.scene['low_obstacle_contact'].data.found > 0).any(dim=1).cpu().numpy()
                hits += contact & ~stopped
        displacement = (robot.data.root_link_pos_w - start).cpu().numpy()
        result = dict(policy=str(args.policy), sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                      seconds=args.seconds, seed=args.seed, hardware_tested=False, deployment_ready=False,
                      contact_definition='positive-height obstacle vs robot subtree; excludes flat floor',
                      terrain_course=args.terrain_course,
                      approach_version=2,
                      obstacle_count=len(obstacles), cases=[dict(case=name, terrain_type=types[i],
                          approach=approaches[i],
                          obstacle_contact_frames=int(hits[i]), first_fall_s=first_fall[i],
                          first_episode_end=first_end[i],
                          final_displacement_m=(None if stopped[i] else displacement[i].tolist()))
                          for i, (name, _) in enumerate(cases)])
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
