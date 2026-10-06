"""Incomplete, mixed-policy or discontinuous evidence must not qualify."""
from pathlib import Path
import sys
from copy import deepcopy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from qualify_hd1910 import check_replay, check_stress
from replay_hd1910 import transition_cases, replay_cases, target_chatter_metrics


def valid_report():
    return dict(policy_sha256='abc',action_semantics='bounded_slew_home_delta_v2',
                seconds=20,continuous_transitions=True,max_action_step_rad=.04,
                cases=[dict(case=n,command=list(c),completed=True,no_fall=True,max_tilt_deg=1.,
                            mean_body_velocity_after_1s=list(c),rms_vx_error_after_1s=0.,
                            rms_yaw_error_after_1s=0.,target_limit_violations=0,
                            saturated_reversal_fraction_max_joint=0.,motion_quality_check_passed=True,
                            max_target_jump_rad=.04,initial_target_jump_rad=.04)
                       for n,c in transition_cases()])


def test_evidence_requires_real_measurements_and_consistent_policy():
    good=valid_report()
    assert not check_replay(good,'abc',transition_cases(),True)
    for key,value in [('policy_sha256','other'),('seconds',1),('continuous_transitions',False),('cases',[])]:
        bad=deepcopy(good);bad[key]=value
        assert check_replay(bad,'abc',transition_cases(),True)
    for key,value in [('max_target_jump_rad',.13),('target_limit_violations',1),
                      ('no_fall',False),('max_tilt_deg',float('nan')),
                      ('mean_body_velocity_after_1s',[0.,0.,.3]),('mean_body_velocity_after_1s',None)]:
        bad=deepcopy(good);bad['cases'][0][key]=value
        assert check_replay(bad,'abc',transition_cases(),True)


def test_saturated_reversal_cannot_pass_as_a_stationary_balanced_gait():
    import numpy as np
    smooth = np.full((100,14),.001)
    assert target_chatter_metrics(smooth,.1)['motion_quality_check_passed']
    # A single chattering joint must not disappear in the 14-joint average.
    smooth[:,0] = np.tile([.1,-.1],50)
    result = target_chatter_metrics(smooth,.1)
    assert result['saturated_reversal_fraction_max_joint'] == 1.
    assert not result['motion_quality_check_passed']
    assert not target_chatter_metrics([],.1)['motion_quality_check_passed']
    good = valid_report()
    for value in (None, float('nan'), -.01, .3):
        bad = deepcopy(good)
        bad['cases'][0]['saturated_reversal_fraction_max_joint'] = value
        assert check_replay(bad,'abc',transition_cases(),True)


def test_stress_cannot_reuse_nominal_or_short_reports():
    good=valid_report()
    good['cases']=good['cases'][:5]
    for row,(name,command) in zip(good['cases'],replay_cases(True)):
        row.update(case=name,command=list(command),mean_body_velocity_after_1s=list(command))
    good.update(profile_sha256='profile',voltage_v=6.5,delay_physics_steps=6,
                actuator_delay_ms=30,initial_tilt_range_deg=5.,seed=42,
                policy_hz=50,physics_hz=200)
    assert not check_stress(good,'abc','profile',6.5,6)
    for key,value in (('voltage_v',7.4),('delay_physics_steps',4),
                      ('actuator_delay_ms',20),('initial_tilt_range_deg',0.),
                      ('profile_sha256','other'),('seconds',1),('seed',7),
                      ('policy_hz',100),('physics_hz',100)):
        bad=deepcopy(good);bad[key]=value
        assert check_stress(bad,'abc','profile',6.5,6)
    assert check_stress({},'abc','profile',6.5,6)
