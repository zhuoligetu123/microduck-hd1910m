#!/usr/bin/env python3
"""Render static WALK metadata HOME and model zero; never access hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'radxa'))
from reference_policy import JOINTS, LocalCalibration, ReferencePolicy
from replay_hd1910 import load_replay_model


def visual_vertices(model, data):
    result = []
    for g in range(model.ngeom):
        if model.geom_group[g] != 2 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        m = model.geom_dataid[g]
        start = model.mesh_vertadr[m]
        local = model.mesh_vert[start:start + model.mesh_vertnum[m]]
        result.append(local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    return np.concatenate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    policy = ReferencePolicy('walk')
    calibration = LocalCalibration()
    model, data, motor = load_replay_model(7.4, bam_reference=True, repair_variant='recovery')
    assert list(JOINTS) == [model.joint(int(j)).name for j in motor.joint_ids]
    assert np.allclose(model.qpos0[motor.qids], 0)
    floor = model.geom('floor').id
    model.geom_rgba[floor] = [.64, .79, .87, 1]
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = .55
    model.vis.headlight.diffuse[:] = .65
    model.vis.global_.offwidth = 760
    model.vis.global_.offheight = 850
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[2] = 1
    option.geomgroup[0] = 1
    # Collision meshes are diagnostic geometry, not the visible robot shell.
    for g in range(model.ngeom):
        if g != floor and model.geom_group[g] != 2:
            model.geom_rgba[g, 3] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.orthographic = 1
    camera.distance = .50
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 30)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
    records = {}
    with mujoco.Renderer(model, height=850, width=760) as renderer:
        for name, q in [('home', policy.home), ('urdf_zero', np.zeros(14))]:
            mujoco.mj_resetData(model, data)
            data.qpos[:7] = [0, 0, .2, 1, 0, 0, 0]
            data.qpos[motor.qids] = q
            mujoco.mj_forward(model, data)
            vertices = visual_vertices(model, data)
            data.qpos[2] += .001 - vertices[:, 2].min()
            mujoco.mj_forward(model, data)
            vertices = visual_vertices(model, data)
            center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            camera.lookat[:] = center
            title = 'WALK HOME | ONNX default_joint_pos' if name == 'home' else 'URDF ZERO | all revolute joints q = 0'
            sheet = Image.new('RGB', (2280, 980), '#edf4f7')
            draw = ImageDraw.Draw(sheet)
            draw.text((28, 16), title, font=font, fill='#153748')
            draw.text((28, 60), 'Static reference pose, not RL motion or a balance test. X forward / Y left / Z up.', font=small, fill='#385966')
            records[name] = {'joint_degrees': dict(zip(JOINTS, np.degrees(q).tolist())), 'views': []}
            for index, (view, azimuth, elevation, label) in enumerate([
                ('front', 180, 0, 'FRONT | from +X'),
                ('side', 270, 0, 'LEFT SIDE | from +Y'),
                ('top', 180, -90, 'TOP | from +Z')]):
                camera.azimuth, camera.elevation = azimuth, elevation
                renderer.update_scene(data, camera, scene_option=option)
                pixels = renderer.render().copy()
                assert np.std(pixels.astype(float)) > 5, 'blank image'
                picture = Image.fromarray(pixels)
                picture.save(args.output / f'{name}_{view}.png')
                sheet.paste(picture, (760 * index, 130))
                draw.text((760 * index + 26, 97), label, font=small, fill='#153748')
                records[name]['views'].append({'name': view, 'azimuth': azimuth, 'elevation': elevation})
            sheet.save(args.output / f'{name}_three_views.png')
    rows = []
    for name, row in sorted(calibration.joints.items(), key=lambda item: item[1]['id']):
        index = JOINTS.index(name) if name in JOINTS else None
        rows.append({'id': row['id'], 'joint': name, 'home_deg': float(np.degrees(policy.home[index])) if index is not None else 0., 'zero_deg': 0})
    report = {'policy': str(policy.path), 'policy_sha256': policy.sha256,
              'geometry_loader': "load_replay_model(7.4, bam_reference=True, repair_variant='recovery')",
              'urdf': str(ROOT / 'microduck_app/web/public/model/microduck.urdf'),
              'urdf_sha256': hashlib.sha256((ROOT / 'microduck_app/web/public/model/microduck.urdf').read_bytes()).hexdigest(),
              'mujoco': mujoco.__version__, 'physics_stepped': False, 'joint_table': rows, 'poses': records}
    (args.output / 'poses.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'joint_table': rows}, indent=2))


if __name__ == '__main__':
    main()
