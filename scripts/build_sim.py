#!/usr/bin/env python3
"""Build the CPU MuJoCo physics bundle from the included RL geometry."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'microduck_rl/src'))
sys.path.insert(0, str(ROOT / 'radxa'))


def main():
    import mujoco
    import mjlab
    from reference_policy import ReferencePolicy
    from mjlab_microduck.tasks.hd1910_bam import make_xgo_bam_env_cfg
    from mjlab_microduck.actuator.cpu_hd1910_bam import PROFILE_PATH
    dest = ROOT / 'sim'
    dest.mkdir(exist_ok=True)
    cfg = make_xgo_bam_env_cfg(play=True, repair_variant='recovery')
    spec = cfg.scene.entities['robot'].build().spec
    spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[.14,.23,.33], rgb2=[.14,.23,.33], width=512, height=3072)
    spec.add_texture(name='grid', type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[.75,.84,.90],
                     rgb1=[.10,.20,.30], rgb2=[.19,.30,.41], width=512, height=512)
    material = spec.add_material(name='floor_mat', texuniform=True, texrepeat=[4,4],
                                 reflectance=.18, shininess=.3)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'grid'
    spec.worldbody.add_light(pos=[0,0,3], dir=[0,0,-1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                           size=[0,0,.1], material='floor_mat')
    model = spec.compile()
    cfg.sim.mujoco.apply(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1920, 1080
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = .4
    model.vis.headlight.diffuse[:] = .6
    mujoco.mj_saveModel(model, str(dest / 'hd1910.mjb'))
    policy = ReferencePolicy('walk')
    shutil.copy2(policy.path, dest / 'policy.onnx')
    shutil.copy2(PROFILE_PATH, dest / 'motor_calibration.json')
    (dest / 'home.json').write_text(json.dumps(policy.home.tolist()))
    def digest(name):
        return hashlib.sha256((dest / name).read_bytes()).hexdigest()
    meta = dict(bundle_schema=2, profile_file='motor_calibration.json', policy_file='policy.onnx',
                policy_sha256=digest('policy.onnx'), sha256=digest('hd1910.mjb'),
                profile_sha256=digest('motor_calibration.json'), mujoco=mujoco.__version__,
                physics_hz=200, delay_steps=4, voltage=7.4, kp_fw=6., actuator_backend='hd1910_bam_m6',
                calibrated=False, hardware_tested=False)
    (dest / 'physics.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(dest)


if __name__ == '__main__':
    main()
