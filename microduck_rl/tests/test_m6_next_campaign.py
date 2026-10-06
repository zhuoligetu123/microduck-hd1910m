from mjlab_microduck.tasks.xgoduck_bam import make_xgo_bam_env_cfg


def test_lateral_quiet_preserves_gait_and_only_adds_targeted_cost():
    base = make_xgo_bam_env_cfg(repair_variant='pitch_retention')
    cfg = make_xgo_bam_env_cfg(repair_variant='head_lateral_quiet')
    assert cfg.actions == base.actions
    assert cfg.observations == base.observations
    assert cfg.commands == base.commands
    extra = cfg.rewards.pop('hd_head_lateral')
    assert extra.weight == -.5
    assert cfg.rewards == base.rewards


def test_yaw_hold_keeps_pitch_training_and_runtime_contract():
    base = make_xgo_bam_env_cfg(repair_variant='pitch_retention')
    train = make_xgo_bam_env_cfg(repair_variant='gait_yaw_hold')
    yaw_only = make_xgo_bam_env_cfg(repair_variant='gait_yaw_only')
    play = make_xgo_bam_env_cfg(play=True, repair_variant='gait_yaw_hold')
    assert train.observations == base.observations
    assert train.commands == base.commands
    assert train.rewards['hd_velocity_error'].params['yaw_square_weight'] == 1.5
    assert train.events['push_robot'] == base.events['push_robot']
    assert yaw_only.actions == base.actions
    assert yaw_only.events == base.events
    assert yaw_only.rewards['hd_velocity_error'].params['yaw_square_weight'] == 1.5
    assert train.actions['joint_pos'].command_loss_probability_range == (.02, .08)
    assert train.scene.entities['robot'].articulation.actuators[0].delay_max_lag == 8
    assert play.actions['joint_pos'].command_loss_probability_range == (0., 0.)


def test_head_limit_adds_only_one_sided_penalty():
    base = make_xgo_bam_env_cfg(repair_variant='pitch_retention')
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_head_limit')
    assert cfg.actions == base.actions
    assert cfg.observations == base.observations
    assert cfg.commands == base.commands
    assert cfg.events == base.events
    assert cfg.rewards.pop('hd_head_upward_excess').weight == -.1
    assert cfg.rewards == base.rewards


def test_yaw_hold_head_keeps_delivery_contract_and_prices_upward_gaze():
    base = make_xgo_bam_env_cfg(repair_variant='gait_yaw_hold')
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_yaw_hold_head')
    play = make_xgo_bam_env_cfg(play=True, repair_variant='gait_yaw_hold_head')
    assert cfg.actions == base.actions
    assert cfg.observations == base.observations
    assert cfg.commands == base.commands
    assert cfg.events == base.events
    assert cfg.rewards['hd_velocity_error'].params['yaw_square_weight'] == 2.
    assert cfg.rewards.pop('hd_head_upward_excess').weight == -.2
    base.rewards['hd_velocity_error'].params['yaw_square_weight'] = 2.
    assert cfg.rewards == base.rewards
    assert play.actions['joint_pos'].command_loss_probability_range == (0., 0.)


def test_step_is_m6_phase_task_not_zero_velocity_walk():
    cfg = make_xgo_bam_env_cfg(repair_variant='step')
    assert cfg.commands['twist'].period == 1.
    assert cfg.commands['head_pose'].ranges == ((0., 0.),)*4
    assert cfg.rewards['step_heights'].params['lift'] == .025
    assert cfg.rewards['step_contacts'].params['lift'] == .025
    assert cfg.actions['joint_pos'].max_step_rad == .10
    assert 'XgoBam' in type(cfg.scene.entities['robot'].articulation.actuators[0]).__name__


def test_balanced_step_keeps_policy_contract_and_rewards_both_soles():
    from mjlab_microduck.tasks import mdp
    base = make_xgo_bam_env_cfg(repair_variant='step')
    cfg = make_xgo_bam_env_cfg(repair_variant='step_balanced')
    assert cfg.actions == base.actions and cfg.observations == base.observations
    assert cfg.commands == base.commands
    assert cfg.rewards['step_heights'].func is mdp.hd_step_sole_phase_error
    assert cfg.rewards['step_heights'].params['lift'] == .025


def test_anchored_step_prices_drift_without_changing_phase_contract():
    base = make_xgo_bam_env_cfg(repair_variant='step_right_lift')
    cfg = make_xgo_bam_env_cfg(repair_variant='step_anchored')
    assert cfg.observations == base.observations
    assert cfg.actions['joint_pos'].clip == base.actions['joint_pos'].clip
    assert cfg.actions['joint_pos'].max_step_rad == base.actions['joint_pos'].max_step_rad
    assert cfg.commands == base.commands
    assert cfg.rewards['step_heights'] == base.rewards['step_heights']
    assert cfg.rewards['step_drift'].weight == -40.
    assert cfg.rewards['step_velocity'].weight == -5.
    assert cfg.actions['joint_pos'].command_loss_probability_range == (.05, .15)
    assert cfg.scene.entities['robot'].articulation.actuators[0].delay_max_lag == 10
    assert make_xgo_bam_env_cfg(play=True, repair_variant='step_anchored').actions['joint_pos'].command_loss_probability_range == (0., 0.)


def test_step_delay_curriculum_preserves_nominal_step_contract():
    base = make_xgo_bam_env_cfg(repair_variant='step_mild_right')
    train = make_xgo_bam_env_cfg(repair_variant='step_delay_curriculum')
    play = make_xgo_bam_env_cfg(play=True, repair_variant='step_delay_curriculum')
    assert train.observations == base.observations
    assert train.commands == base.commands
    assert train.rewards == base.rewards
    assert train.actions['joint_pos'].command_loss_probability_range == (.01, .08)
    assert train.actions['joint_pos'].command_hold_max_steps == 2
    assert train.scene.entities['robot'].articulation.actuators[0].delay_max_lag == base.scene.entities['robot'].articulation.actuators[0].delay_max_lag
    assert play.actions['joint_pos'].command_loss_probability_range == (0., 0.)


def test_joint_age_courses_only_delay_actor_joint_feedback():
    for variant, base in (('gait_joint_age', 'gait_yaw_hold_head'),
                          ('gait_age_only', 'gait_yaw_hold'),
                          ('step_joint_age', 'step_delay_curriculum')):
        train = make_xgo_bam_env_cfg(repair_variant=variant)
        parent = make_xgo_bam_env_cfg(repair_variant=base)
        play = make_xgo_bam_env_cfg(play=True, repair_variant=variant)
        assert train.actions == parent.actions
        assert train.rewards == parent.rewards
        assert train.commands == parent.commands
        for term_name in ('joint_pos', 'joint_vel'):
            term = train.observations['actor'].terms[term_name]
            assert (term.delay_min_lag, term.delay_max_lag) == (1, 4)
            assert term.delay_hold_prob == .8
            expected_play_lag = 0 if term_name == 'joint_pos' else 1
            assert play.observations['actor'].terms[term_name].delay_max_lag == expected_play_lag


def test_gait_mixed_age_covers_fresh_and_captured_feedback():
    base = make_xgo_bam_env_cfg(repair_variant='gait_joint_age')
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_age_mixed')
    assert cfg.actions == base.actions
    assert cfg.rewards == base.rewards
    assert cfg.commands == base.commands
    assert cfg.observations['actor'].terms['joint_pos'].delay_min_lag == 0
    assert cfg.observations['actor'].terms['joint_pos'].delay_max_lag == 4
    assert cfg.observations['actor'].terms['joint_vel'].delay_min_lag == 1


def test_gait_stress_course_only_changes_targeted_training_terms():
    base = make_xgo_bam_env_cfg(repair_variant='gait_joint_age')
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_joint_age_stress')
    assert cfg.observations == base.observations
    assert cfg.commands == base.commands
    assert cfg.rewards['hd_velocity_error'].params['yaw_square_weight'] == 5.
    assert cfg.rewards['hd_head_upward_excess'].weight == -.4
    assert cfg.actions['joint_pos'].command_loss_probability_range == (.05, .25)
    play = make_xgo_bam_env_cfg(play=True, repair_variant='gait_joint_age_stress')
    assert play.actions['joint_pos'].command_loss_probability_range == (0., 0.)


def test_coherent_joint_age_uses_one_delay_for_position_and_velocity():
    from types import SimpleNamespace
    import torch
    from mjlab_microduck.tasks import mdp
    for variant in ('gait_coherent_age', 'step_coherent_age'):
        cfg = make_xgo_bam_env_cfg(repair_variant=variant)
        terms = cfg.observations['actor'].terms
        names = list(terms)
        assert 'joint_vel' not in names
        assert names[names.index('joint_state') + 1] == 'actions'
        joint = terms['joint_state']
        assert joint.func is mdp.hd_joint_state_rel
        assert (joint.delay_min_lag, joint.delay_max_lag, joint.delay_hold_prob) == (1, 4, .8)
        assert joint.noise is None
        assert joint.params['training_noise'] is True
        play = make_xgo_bam_env_cfg(play=True, repair_variant=variant)
        assert 'joint_state' not in play.observations['actor'].terms
    data = SimpleNamespace(default_joint_pos=torch.zeros(1, 2), default_joint_vel=torch.zeros(1, 2),
        joint_pos=torch.tensor([[3., 4.]]), joint_pos_biased=torch.tensor([[1., 2.]]),
        joint_vel=torch.tensor([[5., 6.]]))
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=data)})
    torch.testing.assert_close(mdp.hd_joint_state_rel(env, joint.params['asset_cfg']),
                               torch.tensor([[1., 2., 5., 6.]]))


def test_step_age_drift_adds_small_error_gradient_without_new_observation():
    from mjlab_microduck.tasks import mdp
    import torch
    from types import SimpleNamespace
    base = make_xgo_bam_env_cfg(repair_variant='step_joint_age')
    cfg = make_xgo_bam_env_cfg(repair_variant='step_age_drift')
    assert cfg.actions == base.actions
    assert cfg.observations == base.observations
    assert cfg.rewards.pop('step_radial_drift').weight == -10.
    assert cfg.rewards == base.rewards
    robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor([[0., 0., 0.], [.1, 0., 0.]])))
    env = SimpleNamespace(scene={'robot': robot}, episode_length_buf=torch.tensor([0, 0]))
    torch.testing.assert_close(mdp.step_radial_drift_cost(env), torch.zeros(2))
    env.episode_length_buf[:] = 2
    robot.data.root_link_pos_w[1, 1] = .1
    assert mdp.step_radial_drift_cost(env)[1] > 0


def test_step_lift35_age_changes_only_height_target():
    base = make_xgo_bam_env_cfg(repair_variant='step_joint_age')
    cfg = make_xgo_bam_env_cfg(repair_variant='step_lift35_age')
    assert cfg.actions == base.actions
    assert cfg.observations == base.observations
    assert cfg.commands == base.commands
    assert cfg.rewards['step_heights'].params['lift'] == .035
    assert cfg.rewards['step_contacts'].params['lift'] == .035
    base.rewards['step_heights'].params['lift'] = .035
    base.rewards['step_contacts'].params['lift'] = .035
    assert cfg.rewards == base.rewards


def test_step_lift35_is_nominal_ablation():
    base = make_xgo_bam_env_cfg(repair_variant='step_mild_right')
    cfg = make_xgo_bam_env_cfg(repair_variant='step_lift35')
    assert cfg.observations == base.observations
    assert cfg.actions == base.actions
    assert cfg.commands == base.commands
    assert cfg.rewards['step_heights'].params['lift'] == .035
    assert cfg.rewards['step_contacts'].params['lift'] == .035
    base.rewards['step_heights'].params['lift'] = .035
    base.rewards['step_contacts'].params['lift'] = .035
    assert cfg.rewards == base.rewards


def test_mild_step_balances_right_lift_and_drift():
    base = make_xgo_bam_env_cfg(repair_variant='step_balanced')
    cfg = make_xgo_bam_env_cfg(repair_variant='step_mild_right')
    assert cfg.actions == base.actions
    assert cfg.commands == base.commands
    assert cfg.rewards['step_heights'].params['side_weights'] == (1., 1.5)
    assert cfg.rewards['step_heights'].weight == -6.
    assert cfg.rewards['step_drift'].weight == -24.
    assert cfg.rewards['step_velocity'].weight == -4.


def test_hold_step_only_randomizes_delivered_targets():
    base = make_xgo_bam_env_cfg(repair_variant='step_mild_right')
    cfg = make_xgo_bam_env_cfg(repair_variant='step_hold_robust')
    assert cfg.rewards == base.rewards
    assert cfg.observations == base.observations
    assert cfg.commands == base.commands
    assert cfg.scene.entities['robot'].articulation.actuators[0].delay_max_lag == base.scene.entities['robot'].articulation.actuators[0].delay_max_lag
    assert cfg.actions['joint_pos'].command_loss_probability_range == (.05, .20)
    assert cfg.actions['joint_pos'].command_hold_max_steps == 3


def test_balanced_step_prices_a_grounded_swing_foot_independently():
    from types import SimpleNamespace
    import torch
    from mjlab_microduck.tasks.mdp import hd_step_sole_phase_error
    term = hd_step_sole_phase_error.__new__(hd_step_sole_phase_error)
    term.geom_ids = [0, 1]
    term.vertices = [torch.zeros(1, 3), torch.zeros(1, 3)]
    positions = torch.tensor([[[0., 0., .025], [0., 0., 0.]],
                              [[0., 0., 0.], [0., 0., 0.]]])
    data = SimpleNamespace(geom_xmat=torch.eye(3).repeat(2, 2, 1, 1), geom_xpos=positions)
    command = SimpleNamespace(get_command=lambda _:torch.tensor([[0., 1., 0.], [0., 1., 0.]]))
    class FakeScene(dict):
        terrain = SimpleNamespace(env_origins=torch.zeros(2, 3))
    env = SimpleNamespace(scene=FakeScene(robot=SimpleNamespace(data=SimpleNamespace(
        data=data, projected_gravity_b=torch.tensor([[0., 0., -1.]]*2)))), command_manager=command)
    score = term(env)
    assert torch.allclose(score, torch.tensor([0., .5]))
    assert torch.allclose(term(env, side_weights=(1., 1.75)), torch.tensor([0., 1./2.75]))


def test_delivery_hold_preserves_last_accepted_action():
    from types import SimpleNamespace
    import torch
    from mjlab_microduck.actuator.bounded_position import BoundedPositionAction
    term = BoundedPositionAction.__new__(BoundedPositionAction)
    term.cfg = SimpleNamespace(max_step_rad=.1, clip=None, command_hold_max_steps=1)
    term._offset = torch.zeros(2, 1)
    term._scale = 1.
    term._raw_actions = torch.zeros(2, 1)
    term._processed_actions = torch.zeros(2, 1)
    term._hold_left = torch.zeros(2, dtype=torch.long)
    term._hold_rate = torch.tensor([1., 0.])
    term.process_actions(torch.ones(2, 1))
    assert torch.allclose(term.raw_action, torch.tensor([[0.], [.1]]))
    assert torch.allclose(term._processed_actions, term.raw_action)
    assert torch.allclose(term.applied_step, torch.tensor([[0.], [.1]]))


def test_upward_head_penalty_is_one_sided_and_honors_command():
    import math
    from types import SimpleNamespace
    import torch
    from mjlab_microduck.tasks import mdp
    pitch = math.radians(30)
    q = torch.tensor([[[math.cos(pitch/2), 0., math.sin(pitch/2), 0.]]])
    robot = SimpleNamespace(data=SimpleNamespace(site_quat_w=q,
        projected_gravity_b=torch.tensor([[0., 0., -1.]])))
    cmd = SimpleNamespace(get_command=lambda _:torch.zeros(1, 4))
    env = SimpleNamespace(scene={'robot':robot}, command_manager=cmd, _hd_head_camera_site_id=0)
    assert mdp.hd_head_upward_excess_cost(env).item() > 0
    cmd.get_command = lambda _:torch.tensor([[.6, 0., 0., 0.]])
    assert mdp.hd_head_upward_excess_cost(env).item() == 0


def test_next_variants_preserve_observation_and_action_contract():
    base = make_xgo_bam_env_cfg(repair_variant='head_omni')
    for variant in ('head_balance', 'delay_robust', 'pitch_retention', 'pitch_retention_delay'):
        cfg = make_xgo_bam_env_cfg(repair_variant=variant)
        assert cfg.actions == base.actions
        assert cfg.observations == base.observations
        assert cfg.commands == base.commands
        assert cfg.scene.entities['robot'].articulation.actuators[0].json_path == base.scene.entities['robot'].articulation.actuators[0].json_path
    cfg = make_xgo_bam_env_cfg(repair_variant='head_balance')
    assert cfg.rewards['head_pose_bias'].params['gate_tilt_zero_deg'] == 12.
    assert cfg.rewards['head_pose_bias'].params['gate_height_low'] == .10
    assert make_xgo_bam_env_cfg(repair_variant='delay_robust').scene.entities['robot'].articulation.actuators[0].delay_max_lag == 10


def test_pitch_retention_preserves_parent_rewards_and_covers_lateral_pushes():
    base = make_xgo_bam_env_cfg(repair_variant='head_omni')
    for variant,delay in [('pitch_retention',6),('pitch_retention_delay',10)]:
        cfg = make_xgo_bam_env_cfg(repair_variant=variant)
        assert cfg.rewards == base.rewards
        assert cfg.events['reset_base'] == base.events['reset_base']
        assert cfg.events['push_robot'].params['lateral_probability'] == .2
        assert cfg.scene.entities['robot'].articulation.actuators[0].delay_max_lag == delay


def test_roulade_has_contact_model_and_bounded_m6_actions():
    cfg = make_xgo_bam_env_cfg(repair_variant='roulade')
    assert cfg.actions['joint_pos'].max_step_rad == .10
    assert cfg.actions['joint_pos'].reset_from_joint_state
    assert cfg.episode_length_s == 5.
    assert 'roulade_progress' in cfg.rewards
    assert 'roulade_landing_composite' in cfg.rewards
    assert cfg.scene.entities['robot'].build().spec.compile().nu == 14
