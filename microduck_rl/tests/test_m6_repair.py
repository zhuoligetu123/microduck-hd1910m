from types import SimpleNamespace
import sys
from pathlib import Path
import pytest
import torch
from test_hd1910_bounded import term
from mjlab_microduck.tasks.mdp import hd_action_reversal_cost
from mjlab_microduck.tasks.hd1910_bam import make_xgo_bam_env_cfg

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from run_m6_repair_campaign import assessment
from run_m6_transfer_campaign import CASES


def test_reversal_cost_and_selective_reset():
    action = term()
    action.cfg.max_step_rad = .1
    manager = SimpleNamespace()
    def get_term(name):
        assert name == 'joint_pos'
        return action
    manager.get_term = get_term
    env = SimpleNamespace(action_manager=manager)
    action.process_actions(torch.full((2,14), .4))
    assert not hd_action_reversal_cost(env).any()
    action.process_actions(torch.full((2,14), .4))
    assert not hd_action_reversal_cost(env).any()
    action.process_actions(torch.full((2,14), -.4))
    torch.testing.assert_close(hd_action_reversal_cost(env), torch.ones(2))
    action.reset(torch.tensor([0]))
    assert not action.applied_step[0].any()
    assert not action.previous_step[0].any()
    assert action.applied_step[1].abs().sum() > 0


@pytest.mark.parametrize('variant', ['control', 'reversal', 'yaw'])
def test_repair_preserves_contract_and_does_not_restart_curriculum(variant):
    base = make_xgo_bam_env_cfg(transfer_refine=True)
    cfg = make_xgo_bam_env_cfg(repair_variant=variant)
    assert cfg.actions == base.actions
    assert cfg.observations == base.observations
    assert cfg.scene.entities['robot'].articulation.actuators == base.scene.entities['robot'].articulation.actuators
    for name, weight in [('action_rate_l2', -3.), ('hd_slew_demand', -10.), ('head_pose_bias', 2.)]:
        assert cfg.rewards[name].weight == weight
        assert name + '_weight' not in cfg.curriculum
    assert ('hd_action_reversal' in cfg.rewards) == (variant != 'control')
    assert cfg.rewards['hd_velocity_error'].params['yaw_square_weight'] == (.75 if variant == 'yaw' else .25)


def test_failed_smoothness_cannot_be_selected_by_tracking_count():
    reports = [{'cases':[dict(case=c, completed=True, no_fall=True,
                 baseline_check_passed=True, motion_quality_check_passed=True,
                 head_center_check_passed=True) for c in CASES]} for _ in range(2)]
    assert assessment(reports)['simulation_screen_passed']
    assert not assessment(reports)['deployment_ready']
    reports[0]['cases'][0]['motion_quality_check_passed'] = False
    assert not assessment(reports)['simulation_screen_passed']
    with pytest.raises(ValueError):
        assessment([])
