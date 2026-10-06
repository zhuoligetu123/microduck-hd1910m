"""Static sole/support constraints; a solution is not dynamic gait validation."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from replay_hd1910 import DEFAULT_POSE, load_replay_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    model, data, motor = load_replay_model(7.4, bam_reference=True)
    legs = np.r_[0:5, 9:14]
    sites = [model.site(name).id for name in ('left_foot', 'right_foot')]
    feet = []
    for name in ('left_foot_collision', 'right_foot_collision'):
        gid = model.geom(name).id
        mesh = model.geom_dataid[gid]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]].copy()
        feet.append((gid, vertices))
    data.qpos[:7] = [0., 0., .125, 1., 0., 0., 0.]
    data.qpos[motor.qids] = DEFAULT_POSE
    mujoco.mj_forward(model, data)
    nominal_sites = data.site_xpos[sites].copy()
    low = model.jnt_range[motor.joint_ids[legs], 0]
    high = model.jnt_range[motor.joint_ids[legs], 1]
    rows = []
    for swing in range(2):
        stance = 1 - swing
        for lift in (.008, .015, .025):
            for allow_roll in (False, True):
                # Foot sites lie on the sole plane; horizontal feet make the
                # convex support polygon valid, unlike an arbitrary tilted box.
                target = nominal_sites.copy()
                target[:, 2] = [lift if i == swing else 0. for i in range(2)]
                x0 = np.r_[0., 0., .12, 0., DEFAULT_POSE[legs]]
                bounds = (np.r_[-.08, -.08, .08, -.35 if allow_roll else -1e-10, low],
                          np.r_[.08, .08, .17, .35 if allow_roll else 1e-10, high])

                def update(x):
                    data.qpos[:3] = x[:3]
                    data.qpos[3:7] = [np.cos(x[3]/2), np.sin(x[3]/2), 0., 0.]
                    data.qpos[motor.qids] = DEFAULT_POSE
                    data.qpos[motor.qids[legs]] = x[4:]
                    mujoco.mj_forward(model, data)
                    world = [v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                             for g, v in feet]
                    hull = ConvexHull(world[stance][:, :2])
                    com = data.subtree_com[model.body('trunk_base').id, :2]
                    outside = np.maximum(hull.equations[:, :2] @ com + hull.equations[:, 2] + .002, 0.)
                    rotations = [Rotation.from_matrix(data.site_xmat[s].reshape(3, 3)).as_rotvec()
                                 for s in sites]
                    return world, outside, np.asarray(rotations)

                def residual(x):
                    world, outside, rotations = update(x)
                    # A fixed residual dimension is needed across hull changes.
                    return np.r_[(data.site_xpos[sites]-target).ravel()/.001,
                                 rotations.ravel()/.02, outside.max(initial=0.)/.001,
                                 .01*(x[4:]-DEFAULT_POSE[legs]), .01*x[3]]

                result = least_squares(residual, x0, bounds=bounds, max_nfev=500)
                world, outside, rotations = update(result.x)
                position_error = float(np.max(np.abs(data.site_xpos[sites]-target)))
                rotation_error = float(np.max(np.linalg.norm(rotations, axis=1)))
                clearance = np.array([np.min(v[:, 2]) for v in world])
                feasible = (position_error < .0005 and rotation_error < np.radians(.5)
                            and outside.max(initial=0.) < .001
                            and clearance[stance] >= -.0005 and clearance[swing] >= lift-.0005)
                rows.append(dict(swing=('left', 'right')[swing], requested_mm=lift*1000,
                    allow_trunk_roll=allow_roll, optimizer_success=bool(result.success),
                    static_constraints_satisfied=bool(feasible),
                    position_error_mm=position_error*1000,
                    orientation_error_deg=float(np.degrees(rotation_error)),
                    support_margin_shortfall_mm=float(outside.max(initial=0.)*1000),
                    sole_clearance_mm=(clearance*1000).tolist(),
                    trunk_roll_deg=float(np.degrees(result.x[3])),
                    leg_delta_deg=np.degrees(result.x[4:]-DEFAULT_POSE[legs]).tolist(),
                    qpos=data.qpos.tolist()))
    report = dict(method='static IK: horizontal soles, fixed foot XY, COM inside sole polygon with 2mm margin',
                  dynamic_feasibility_tested=False, hardware_tested=False,
                  interpretation='Optimizer failure is not proof of geometric impossibility; success is not gait stability.',
                  rows=rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for row in rows:
        print({k: v for k, v in row.items() if k not in ('qpos', 'leg_delta_deg')})


if __name__ == '__main__':
    main()
