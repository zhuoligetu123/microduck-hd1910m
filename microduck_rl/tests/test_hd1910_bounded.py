"""Bounded action/export/history parity, without physical hardware."""
from types import SimpleNamespace
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto
from mjlab_microduck.actuator.bounded_position import BoundedPositionAction, bound_export, CONTRACT
from mjlab_microduck.tasks.microduck_hd1910_env_cfg import make_bounded_hd1910_velocity_env_cfg, make_reference_hd1910_velocity_env_cfg
from mjlab_microduck.tasks.mdp import hd_target_saturation_cost


def term():
    action = BoundedPositionAction.__new__(BoundedPositionAction)
    action._raw_actions = torch.zeros((2, 14))
    action._offset = torch.linspace(-.2, .2, 14).repeat(2, 1)
    action._scale = 1.
    action.cfg = SimpleNamespace(clip={'all': (-.4, .5)})
    action._clip = torch.tensor([[[-.4, .5]]*14])
    return action


def test_bounded_action_history_is_applied_delta_and_input_unchanged():
    action = term()
    x = torch.stack((torch.full((14,), -10.), torch.full((14,), 10.)))
    original = x.clone()
    action.process_actions(x)
    torch.testing.assert_close(x, original)
    torch.testing.assert_close(action._raw_actions + action._offset, action._processed_actions)
    assert action._processed_actions.min() >= -.4
    assert action._processed_actions.max() <= .5
    action.reset(torch.tensor([0]))
    assert torch.count_nonzero(action.raw_action[0]) == 0


def test_contact_reset_keeps_seated_target_instead_of_home():
    action = term()
    action.cfg.max_step_rad = .1
    action.cfg.reset_from_joint_state = True
    action._target_ids = torch.arange(14)
    action._entity = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.full((2,14), .3)))
    action.process_actions(torch.zeros((2,14)))
    previous_other = action.raw_action[1].clone()
    action.reset(torch.tensor([0]))
    torch.testing.assert_close(action.raw_action[0]+action._offset[0],torch.full((14,),.3))
    torch.testing.assert_close(action.raw_action[1],previous_other)
    action.process_actions(action.raw_action.clone())
    torch.testing.assert_close(action._processed_actions[0],torch.full((14,),.3))


def test_export_matches_training_transform_and_rejects_double_export(tmp_path):
    action = term()
    path = tmp_path/'identity.onnx'
    graph = helper.make_graph([helper.make_node('Identity', ['obs'], ['actions'])], 'test',
                              [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [2,14])],
                              [helper.make_tensor_value_info('actions', TensorProto.FLOAT, [2,14])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',18)], ir_version=10)
    onnx.save(model, path)
    bound_export(path, action)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    for scale in (0., .1, 10.):
        x = np.random.default_rng(42).normal(size=(2,14)).astype(np.float32)*scale
        action.process_actions(torch.tensor(x))
        np.testing.assert_allclose(session.run(None, {'obs': x})[0], action.raw_action.numpy(), atol=1e-7)
    assert session.get_modelmeta().custom_metadata_map['action_semantics'] == CONTRACT
    with pytest.raises(ValueError):
        bound_export(path, action)


def test_opt_in_bounds_preserve_old_task_and_nonpositive_penalty():
    base = make_reference_hd1910_velocity_env_cfg()
    bounded = make_bounded_hd1910_velocity_env_cfg()
    assert base.actions['joint_pos'].clip is None
    assert len(bounded.actions['joint_pos'].clip) == 14
    for group in ('actor','critic'):
        assert bounded.observations[group].terms['actions'].params == {'action_name':'joint_pos'}
        assert base.observations[group].terms['actions'].params == {}
    action = term()
    latent = torch.full((2,14), 3.)
    action.process_actions(latent)
    manager = SimpleNamespace(action=latent, get_term=lambda name: action)
    cost = hd_target_saturation_cost(SimpleNamespace(action_manager=manager))
    assert torch.all(cost >= 0)
    assert torch.all(cost * bounded.rewards['hd_target_saturation'].weight <= 0)


def test_replay_cannot_pass_with_out_of_range_targets():
    import importlib.util
    from pathlib import Path
    import sys
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('bounded_replay_test', scripts/'replay_hd1910.py')
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        row = dict(completed=True, no_fall=True, max_tilt_deg=0., case='stand',
                   command=[0.,0.,0.], mean_body_velocity_after_1s=[0.,0.,0.],
                   rms_vx_error_after_1s=0., rms_yaw_error_after_1s=0., target_limit_violations=0)
        assert replay.baseline_check(row,20)
        assert not replay.baseline_check(row,10)
        row['target_limit_violations'] = 1
        assert not replay.baseline_check(row,20)
    finally:
        sys.path.pop(0)


def test_slew_export_matches_sequential_training_and_reset(tmp_path):
    action = term()
    action.cfg.max_step_rad = .04
    path = tmp_path/'slew.onnx'
    # Raw actor selects the first 14 features; history lives separately at 34:48.
    graph = helper.make_graph([
        helper.make_node('Gather',['obs','ids'],['actions'],axis=1)],'slew',
        [helper.make_tensor_value_info('obs',TensorProto.FLOAT,[2,61])],
        [helper.make_tensor_value_info('actions',TensorProto.FLOAT,[2,14])],
        initializer=[onnx.numpy_helper.from_array(np.arange(14,dtype=np.int64),'ids')])
    onnx.save(helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=10),path)
    bound_export(path,action)
    session = ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    obs = np.zeros((2,61),dtype=np.float32)
    for i in range(50):
        if i == 25:
            action.reset(torch.tensor([0]))
            obs[0,34:48]=0
        obs[:,:14] = 10. if i < 25 else -10.
        previous = obs[:,34:48].copy()
        actual = session.run(None,{'obs':obs})[0]
        action.process_actions(torch.from_numpy(obs[:,:14]))
        np.testing.assert_allclose(actual,action.raw_action.numpy(),atol=2e-7)
        assert np.abs(actual-previous).max() <= .040001
        assert np.all(actual+action._offset.numpy() >= -.400001)
        assert np.all(actual+action._offset.numpy() <= .500001)
        obs[:,34:48]=actual


def test_slew_is_opt_in_and_covers_low_speed_commands():
    from mjlab_microduck.tasks.microduck_hd1910_env_cfg import make_slew_hd1910_velocity_env_cfg
    from mjlab_microduck.actuator.bounded_position import BoundedPositionActionCfg
    cfg=make_slew_hd1910_velocity_env_cfg()
    assert make_bounded_hd1910_velocity_env_cfg().actions['joint_pos'].max_step_rad is None
    assert cfg.actions['joint_pos'].max_step_rad == .10
    assert cfg.commands['twist'].ranges.ang_vel_z == (-.5,.5)
    assert cfg.commands['twist'].rel_turn_in_place_envs == .35
    assert 'standing_envs' not in cfg.curriculum
    for value in (0.,-.01,float('nan'),float('inf'),.13):
        with pytest.raises(ValueError):
            BoundedPositionActionCfg(entity_name='robot',actuator_names=('.*',),max_step_rad=value)


def test_slew_lag_is_not_range_saturation_or_latent_action_rate():
    from mjlab_microduck.tasks.mdp import hd_applied_action_rate_cost
    action=term()
    action.cfg.max_step_rad=.04
    latent=torch.full((2,14),.2)
    action.process_actions(latent)
    env=SimpleNamespace(action_manager=SimpleNamespace(action=latent,get_term=lambda name: action))
    assert torch.all(hd_target_saturation_cost(env)<1e-12)
    torch.testing.assert_close(hd_applied_action_rate_cost(env),torch.full((2,),14*.04**2))


def test_slew_demand_distinguishes_the_applied_rate_plateau():
    from mjlab_microduck.tasks.mdp import hd_applied_action_rate_cost, hd_slew_demand_cost
    from mjlab_microduck.tasks.microduck_hd1910_env_cfg import make_refined_hd1910_velocity_env_cfg
    costs = []
    for request in (.02, .04, .10, .20):
        action = term()
        action._offset.zero_()
        action.cfg.max_step_rad = .04
        action.process_actions(torch.full((2, 14), request))
        manager = SimpleNamespace(get_term=lambda name: action)
        env = SimpleNamespace(action_manager=manager)
        costs.append((hd_applied_action_rate_cost(env)[0], hd_slew_demand_cost(env)[0]))
    assert costs[0][1] == costs[1][1] == 0
    assert costs[1][0] == costs[2][0] == costs[3][0]
    assert 0 < costs[2][1] < costs[3][1]
    for previous, request in ((-.15, -.25), (.15, .25), (.15, -.15)):
        action = term()
        action._offset.zero_()
        action.cfg.max_step_rad = .04
        action._raw_actions.fill_(previous)
        action.process_actions(torch.full((2, 14), request))
        env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: action))
        expected = 14 * (abs(request-previous)-.04)**2
        assert hd_slew_demand_cost(env)[0].item() == pytest.approx(expected)
    cfg = make_refined_hd1910_velocity_env_cfg()
    assert cfg.rewards['hd_slew_demand'].weight == 0
    assert cfg.actions['joint_pos'].max_step_rad == .10


def test_discovery_only_changes_learning_not_physics_or_safety_contract():
    from mjlab_microduck.tasks.microduck_hd1910_env_cfg import (
        make_discovery_hd1910_velocity_env_cfg, make_slew_hd1910_velocity_env_cfg,
    )
    old = make_slew_hd1910_velocity_env_cfg()
    cfg = make_discovery_hd1910_velocity_env_cfg()
    assert cfg.actions == old.actions
    assert cfg.observations == old.observations
    assert cfg.terminations == old.terminations
    assert cfg.events == old.events
    assert cfg.scene.entities['robot'] == old.scene.entities['robot']
    assert cfg.rewards['hd_target_saturation'] == old.rewards['hd_target_saturation']
    assert old.rewards['pose'].params['std_walking'][r'.*hip_roll.*'] == .05
    assert cfg.rewards['pose'].params['std_walking'][r'.*hip_roll.*'] == .15
    stages = cfg.curriculum['action_rate_weight'].params['weight_stages']
    assert stages[0]['weight'] == -.02
    assert stages[1]['step'] == 2500 * 24
    assert cfg.commands['twist'].rel_standing_envs > 0
    assert cfg.commands['twist'].rel_turn_in_place_envs > 0


def test_hd_factory_does_not_mutate_stock_or_double_apply_motor_mass():
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
    from mjlab_microduck.tasks.hd1910_suite import adapt_task
    stock = make_microduck_velocity_env_cfg()
    original_robot = stock.scene.entities['robot']
    original_factory = original_robot.spec_fn
    original_motor = original_robot.articulation.actuators[0]
    mass = original_factory().compile().body_mass.sum()
    hd = make_reference_hd1910_velocity_env_cfg()
    twin = adapt_task(stock)
    assert hd.scene.entities['robot'] is not original_robot
    assert twin.scene.entities['robot'] is not original_robot
    assert original_robot.spec_fn is original_factory
    assert original_robot.articulation.actuators[0] is original_motor
    assert type(original_motor).__name__ == 'FrictionDRBamActuatorCfg'
    for cfg in (hd, twin):
        assert cfg.scene.entities['robot'].spec_fn().compile().body_mass.sum() == pytest.approx(mass + .045)
