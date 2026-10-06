import math
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks.mdp import hd_head_gaze_envelope_cost
from mjlab_microduck.tasks.xgoduck_bam import make_xgo_bam_env_cfg


def gaze_cost(pitch, neck=0., head=0., tilt=0.):
    angle = math.radians(pitch) / 2
    robot = SimpleNamespace(data=SimpleNamespace(
        site_quat_w=torch.tensor([[[math.cos(angle), 0., math.sin(angle), 0.]]]),
        projected_gravity_b=torch.tensor([[math.sin(tilt), 0., -math.cos(tilt)]])))
    class Commands:
        def get_command(self, name):
            return torch.tensor([[math.radians(neck), math.radians(head), 0., 0.]])
    env = SimpleNamespace(scene={'robot': robot}, command_manager=Commands(), _hd_head_camera_site_id=0)
    return hd_head_gaze_envelope_cost(env).item()


def test_v5_preserves_head_contract_and_only_prices_locomotion():
    parent = make_xgo_bam_env_cfg(repair_variant='gait_head_dc_stride_v4')
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_head_lift_v5')
    assert cfg.actions == parent.actions and cfg.observations == parent.observations
    assert cfg.scene.entities['robot'].articulation == parent.scene.entities['robot'].articulation
    assert cfg.scene.terrain.terrain_type == parent.scene.terrain.terrain_type
    assert cfg.sim == parent.sim and cfg.events == parent.events
    assert cfg.rewards['head_pose_bias'].params == parent.rewards['head_pose_bias'].params
    assert cfg.rewards['head_pose_bias'].weight == 5.
    assert 'head_pose_bias_weight' not in cfg.curriculum
    assert cfg.rewards['foot_swing_height'].params['target_height'] == .025
    for name, weight in (('track_linear_velocity', 6.), ('foot_swing_height', -16.),
                         ('hd_sole_progress', 16.)):
        stages = cfg.curriculum[name + '_weight'].params['weight_stages']
        assert stages[-1] == {'step': 120*24, 'weight': weight}
        assert all(s['weight'] * weight > 0 for s in stages)


def test_envelope_honors_opposite_pitch_axes_without_rewarding_droop():
    assert gaze_cost(0.) == 0.
    assert gaze_cost(-25.) > 0.
    assert gaze_cost(-25., neck=20.) == 0.
    assert gaze_cost(-25., head=-20.) == 0.
    assert gaze_cost(25., head=20.) == 0.
    assert gaze_cost(25., neck=20.) > 0.
    assert gaze_cost(-25., tilt=math.radians(45.)) == 0.


def test_recipe_covers_head_commands_without_changing_action_contract():
    base = make_xgo_bam_env_cfg(repair_variant='gait_coherent_age')
    for name in ('gait_head_commands', 'gait_head_balance'):
        cfg = make_xgo_bam_env_cfg(repair_variant=name)
        assert cfg.actions == base.actions
        assert cfg.commands['head_pose'].zero_command_prob == .3
        assert cfg.commands['head_pose'].ranges[:2] == ((-.35, .35),)*2
        assert 'hd_head_upward_excess' not in cfg.rewards
        term = cfg.observations['actor'].terms['joint_state']
        assert term.delay_min_lag == 0 and term.delay_max_lag == 4
        play = make_xgo_bam_env_cfg(play=True, repair_variant=name)
        assert list(play.observations['actor'].terms) == list(
            make_xgo_bam_env_cfg(play=True).observations['actor'].terms)
    assert cfg.rewards['hd_head_gaze_envelope'].weight < 0
    assert cfg.rewards['head_pose_bias'].params['gate_tilt_zero_deg'] == 22.


def test_snapshot_export_restores_names_without_changing_graph(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    from mjlab_microduck.tasks.xgoduck_bam import restore_joint_snapshot_metadata
    graph = helper.make_graph([helper.make_node('Identity', ['obs'], ['out'])], 'test',
        [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info('out', TensorProto.FLOAT, [1, 61])])
    model = helper.make_model(graph)
    helper.set_model_props(model, {'joint_snapshot_training':'coherent_pos_vel_delay_v1',
                                  'observation_names':'joint_state'})
    duplicate = model.metadata_props.add()
    duplicate.key, duplicate.value = 'observation_names', 'joint_state'
    path = tmp_path/'policy.onnx'
    onnx.save(model, path)
    restore_joint_snapshot_metadata(path)
    restored = onnx.load(path)
    assert restored.graph.SerializeToString() == model.graph.SerializeToString()
    assert 'joint_pos,joint_vel' in {p.key:p.value for p in restored.metadata_props}['observation_names']


def test_targeted_head_buckets_keep_neutral_and_opposite_pitch_signs():
    from mjlab_microduck.tasks.mdp import LocomotionHeadCommand, LocomotionHeadCommandCfg
    torch.manual_seed(2026)
    term = LocomotionHeadCommand.__new__(LocomotionHeadCommand)
    term._env = SimpleNamespace(device='cpu', num_envs=600)
    term.cfg = LocomotionHeadCommandCfg(ranges=((-0.35,.35),)*4,
                                       resampling_time_range=(3.,6.))
    term.dim, term._command = 4, torch.zeros(600,4)
    term._resample_command(torch.arange(600))
    q = term.command
    assert torch.isfinite(q).all() and (q.abs() <= .35).all()
    assert ((q == 0).all(dim=1)).sum() > 50
    assert ((q[:,0] == 0) & (q[:,1] < -.15)).sum() > 50
    assert ((q[:,0] > .15) & (q[:,1] == 0)).sum() > 50
    assert ((q[:,0] > .15) & (q[:,1] < -.15)).sum() > 50


def test_v2_keeps_parent_control_and_measures_sole_not_site():
    from mjlab_microduck.tasks.mdp import hd_sole_swing_height
    parent = make_xgo_bam_env_cfg(repair_variant='gait_head_balance')
    for name in ('gait_head_follow_v2', 'gait_head_stride_v2'):
        cfg = make_xgo_bam_env_cfg(repair_variant=name)
        assert cfg.actions == parent.actions and cfg.observations == parent.observations
        assert cfg.rewards['head_pose_tracking'].params['fine_std'] == .15
    assert cfg.rewards['foot_swing_height'].func is hd_sole_swing_height
    assert cfg.rewards['foot_swing_height'].params['target_height'] == .025
    assert cfg.rewards['foot_swing_height'].weight < 0
    assert cfg.rewards['hd_sole_progress'].weight > 0


def test_v3_impulses_are_training_only_and_keep_velocity_contract():
    parent = make_xgo_bam_env_cfg(repair_variant='gait_head_follow_v2')
    for name in ('gait_head_force_v3', 'gait_head_force_sole_v3'):
        cfg = make_xgo_bam_env_cfg(repair_variant=name)
        assert cfg.actions == parent.actions and cfg.observations == parent.observations
        assert cfg.rewards['hd_velocity_error'] == parent.rewards['hd_velocity_error']
        assert cfg.events['head_impulse'].params['duration_s'] == (.1, .2)
        assert cfg.events['head_impulse'].params['asset_cfg'].body_names == ('jaw_soft',)
        assert cfg.rewards['head_pose_bias'].params['gate_height_high'] == .10
        assert cfg.rewards['head_pose_bias'].params['gate_tilt_zero_deg'] == 22.
        assert 'head_impulse' not in make_xgo_bam_env_cfg(play=True, repair_variant=name).events


def test_v4_dc_ablation_preserves_physics_and_does_not_freeze_the_head():
    parent = make_xgo_bam_env_cfg(repair_variant='gait_head_force_sole_v3')
    for name in ('gait_head_dc_v4', 'gait_head_dc_stride_v4'):
        cfg = make_xgo_bam_env_cfg(repair_variant=name)
        assert cfg.actions == parent.actions
        assert cfg.observations == parent.observations
        assert cfg.scene.entities['robot'].articulation == parent.scene.entities['robot'].articulation
        assert cfg.events == parent.events
        assert 'fine_std' not in cfg.rewards['head_pose_tracking'].params
        assert cfg.rewards['head_pose_bias'].params['axis_weights'][1] == 4.
        assert cfg.rewards['head_pose_bias'].params['gate_tilt_zero_deg'] == 35.
        assert cfg.rewards['foot_swing_height'] == parent.rewards['foot_swing_height']
        stages = cfg.curriculum['head_pose_bias_weight'].params['weight_stages']
        assert stages == [{'step':0,'weight':3.}, {'step':960,'weight':4.}, {'step':2400,'weight':5.}]
        assert 'head_impulse' not in make_xgo_bam_env_cfg(play=True, repair_variant=name).events
    dc = make_xgo_bam_env_cfg(repair_variant='gait_head_dc_v4')
    assert dc.rewards['hd_slew_demand'] == parent.rewards['hd_slew_demand']
    assert dc.rewards['hd_velocity_error'] == parent.rewards['hd_velocity_error']
    assert cfg.rewards['hd_slew_demand'].weight == -4.
    assert cfg.rewards['action_rate_l2'] == parent.rewards['action_rate_l2']
