from types import SimpleNamespace
import torch
from mjlab_microduck.tasks.mdp import hd_head_motion_cost, hd_sole_swing_height
from mjlab_microduck.tasks.hd1910_bam import make_xgo_bam_env_cfg


def test_ray_swing_peak_resets_per_environment_without_changing_landing_cost():
    from mjlab_microduck.tasks.mdp import hd_feet_swing_height

    class Contact:
        data = SimpleNamespace(found=torch.tensor([[0, 1], [0, 1]]))
        landing = torch.zeros(2, 2, dtype=torch.bool)

        def compute_first_contact(self, dt):
            return self.landing

    class Commands:
        def get_command(self, name):
            return torch.tensor([[.1, 0., 0.], [.1, 0., 0.]])

    term = hd_feet_swing_height.__new__(hd_feet_swing_height)
    term.peak_heights = torch.tensor([[.02, 0.], [.02, 0.]])
    term.step_dt = .02
    contact = Contact()
    sensor = SimpleNamespace(data=SimpleNamespace(heights=torch.tensor([[.008, 0.], [.008, 0.]])))
    env = SimpleNamespace(scene={'feet': contact, 'height': sensor},
                          command_manager=Commands(), extras={'log': {}})
    term.reset(torch.tensor([0]))
    term(env, 'feet', 'height', .02, 'twist', .01)
    contact.data.found[:] = 1
    contact.landing[:, 0] = True
    cost = term(env, 'feet', 'height', .02, 'twist', .01)
    assert torch.allclose(cost, torch.tensor([.36, 0.]))
    term.peak_heights[:] = .02
    term.reset()
    assert torch.count_nonzero(term.peak_heights) == 0
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_reference_linear_only_v22')
    assert cfg.rewards['foot_swing_height'].func is hd_feet_swing_height


def test_cycle_yaw_keeps_persistent_error_and_resets_commands_and_episodes():
    from mjlab_microduck.tasks.mdp import hd_velocity_error_cost
    class Commands:
        value = torch.tensor([[.1, 0., 0.]])
        def get_command(self, name):
            return self.value
    data = SimpleNamespace(root_link_lin_vel_b=torch.tensor([[.1,0.,0.]]),
                           root_link_ang_vel_b=torch.zeros(1,3))
    command = Commands()
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=data)}, command_manager=command,
                          episode_length_buf=torch.tensor([10]), step_dt=.02)
    for i in range(100):
        data.root_link_ang_vel_b[0, 2] = .4 if i % 2 else -.4
        cost = hd_velocity_error_cost(env, 2.5, .2)
    assert cost.item() < .1
    data.root_link_ang_vel_b[0, 2] = .4
    for i in range(100):
        cost = hd_velocity_error_cost(env, 2.5, .2)
    assert abs(cost.item()-3.5) < .001
    command.value[0, 2] = .4
    assert hd_velocity_error_cost(env, 2.5, .2).item() == 0.
    data.root_link_ang_vel_b[0, 2] = 0.
    env.episode_length_buf[:] = 1
    assert hd_velocity_error_cost(env, 2.5, .2).item() == 3.5
    parent = make_xgo_bam_env_cfg(repair_variant='gait_bilateral_v11')
    candidate = make_xgo_bam_env_cfg(repair_variant='gait_cycle_yaw_v15')
    assert parent.actions == candidate.actions and parent.commands == candidate.commands
    assert parent.observations == candidate.observations and parent.events == candidate.events
    assert candidate.rewards['hd_velocity_error'].params.pop('yaw_average_s') == .2
    assert candidate.rewards == parent.rewards
    release = make_xgo_bam_env_cfg(repair_variant='gait_lift_release_v16')
    assert release.actions == parent.actions and release.observations == parent.observations
    assert release.commands == parent.commands and release.events == parent.events
    assert release.rewards['hd_velocity_error'] == parent.rewards['hd_velocity_error']
    assert release.rewards['hd_sole_progress'] == parent.rewards['hd_sole_progress']
    assert release.rewards['action_rate_l2'].weight == -.1
    assert release.rewards['hd_slew_demand'].weight == -.5


def test_head_cost_is_nonnegative_and_has_a_deadband():
    data = SimpleNamespace(body_link_ang_vel_w=torch.tensor([[[0.,0.,0.]], [[1.,0.,0.]], [[3.,0.,0.]]]))
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=data)}, _hd_head_body_id=0)
    assert hd_head_motion_cost(env).tolist() == [0.,0.,1.]


def test_new_variants_keep_commands_actions_and_physics():
    base = make_xgo_bam_env_cfg(repair_variant='pitch_body')
    for variant in ('head_quiet', 'head_sole'):
        cfg = make_xgo_bam_env_cfg(repair_variant=variant)
        assert cfg.actions == base.actions and cfg.observations == base.observations
        assert cfg.commands == base.commands and cfg.events == base.events
        robot, parent = cfg.scene.entities['robot'], base.scene.entities['robot']
        assert robot.articulation == parent.articulation
        assert robot.collisions == parent.collisions and robot.init_state == parent.init_state
        assert robot.spec_fn.func == parent.spec_fn.func
        assert robot.spec_fn.args == parent.spec_fn.args and robot.spec_fn.keywords == parent.spec_fn.keywords
        assert cfg.rewards['head_pose_tracking'] == base.rewards['head_pose_tracking']
        assert cfg.rewards['head_pose_bias'] == base.rewards['head_pose_bias']
        assert cfg.curriculum['hd_head_motion_weight'].params['weight_stages'][-1]['weight'] == -.1
    assert cfg.rewards['foot_swing_height'].func is hd_sole_swing_height
    assert cfg.rewards['foot_swing_height'].params['target_height'] == .025


def test_sole_peak_penalizes_low_landings_and_drops_old_episode():
    class Scene(dict):
        terrain = SimpleNamespace(env_origins=torch.zeros(2,3))
    class Contact:
        data = SimpleNamespace(found=torch.ones(2,2))
        def compute_first_contact(self, dt):
            return torch.ones(2,2,dtype=torch.bool)
    class Commands:
        def get_command(self, name):
            return torch.tensor([[.1,0.,0.],[.1,0.,0.]])
    term = hd_sole_swing_height.__new__(hd_sole_swing_height)
    term.geom_ids = [0,1]
    term.vertices = [torch.zeros(1,3)]*2
    term.peak = torch.tensor([[.025,.025],[.005,.005]])
    data = SimpleNamespace(geom_xmat=torch.eye(3).repeat(2,2,1,1), geom_xpos=torch.zeros(2,2,3))
    env = SimpleNamespace(scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(data=data)), feet=Contact()),
        command_manager=Commands(), episode_length_buf=torch.tensor([10,10]), step_dt=.02, extras={'log':{}})
    cost = term(env, 'feet', 'unused', .025, 'twist', .01)
    assert cost[0] == 0 and cost[1] > 0
    assert torch.count_nonzero(term.peak) == 0
    # Course scoring must not punish clearing the former 20 mm target.
    term.peak[:] = torch.tensor([[.035, .035], [.010, .010]])
    cost = term(env, 'feet', 'unused', .025, 'twist', .01,
                tolerance=0., shortfall_only=True)
    assert cost[0] == 0 and torch.allclose(cost[1], torch.tensor(.72))
    term.peak[:] = .025
    env.episode_length_buf[:] = 0
    cost = term(env, 'feet', 'unused', .025, 'twist', .01)
    assert (cost > 0).all()
    term.peak[:] = 1.
    term.reset(torch.tensor([0]))
    assert term.peak[0].sum() == 0 and term.peak[1].sum() == 2


def test_lift_progress_requires_new_height_and_other_foot_support():
    class Contact:
        def __init__(self):
            self.data = SimpleNamespace(found=torch.tensor([[0,1],[0,0]]))
        def compute_first_contact(self, dt):
            return torch.zeros(2,2,dtype=torch.bool)
    class Scene(dict):
        terrain = SimpleNamespace(env_origins=torch.zeros(2,3))
    class Commands:
        def get_command(self, name):
            return torch.tensor([[.1,0.,0.],[.1,0.,0.]])
    term = hd_sole_swing_height.__new__(hd_sole_swing_height)
    term.geom_ids = [0,1]
    term.vertices = [torch.zeros(1,3)]*2
    term.peak = torch.tensor([[.005,0.],[.005,0.]])
    data = SimpleNamespace(geom_xmat=torch.eye(3).repeat(2,2,1,1),
        geom_xpos=torch.tensor([[[0.,0.,.010],[0.,0.,0.]],[[0.,0.,.010],[0.,0.,0.]]]))
    contact = Contact()
    env = SimpleNamespace(scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(data=data)), feet=contact),
        command_manager=Commands(), episode_length_buf=torch.tensor([10,10]), step_dt=.02, extras={'log':{}})
    progress = term(env, 'feet', 'unused', .025, 'twist', .01, reward_progress=True)
    assert torch.isclose(progress[0],torch.tensor(.2)) and progress[1] == 0
    assert term(env, 'feet', 'unused', .025, 'twist', .01, reward_progress=True).sum() == 0
    data.geom_xpos[:, 0, 2] = .05
    progress = term(env, 'feet', 'unused', .025, 'twist', .01, reward_progress=True)
    assert torch.isclose(progress[0], torch.tensor(.6))
    data.geom_xpos[:, 0, 2] = .10
    assert term(env, 'feet', 'unused', .025, 'twist', .01, reward_progress=True).sum() == 0


def test_air_time_prices_real_clearance_and_rejects_flight_fall_and_idle():
    class Scene(dict):
        terrain = SimpleNamespace(env_origins=torch.zeros(6, 3))
    class Contact:
        data = SimpleNamespace(found=torch.tensor([[0,1], [0,1], [0,0], [0,1], [0,1], [0,1]]),
                               current_air_time=torch.tensor([[.2, 0.], [.2, 0.], [.2, .2],
                                                              [.2, 0.], [.2, 0.], [.2, 0.]]))
        def compute_first_contact(self, dt):
            return torch.zeros(6, 2, dtype=torch.bool)
    class Commands:
        def get_command(self, name):
            return torch.tensor([[.1,0,0]]*5 + [[0.,0,0]])
    term = hd_sole_swing_height.__new__(hd_sole_swing_height)
    term.geom_ids, term.vertices = [0, 1], [torch.zeros(1, 3)]*2
    term.peak = torch.zeros(6, 2)
    geom = SimpleNamespace(geom_xmat=torch.eye(3).repeat(6,2,1,1), geom_xpos=torch.zeros(6,2,3))
    geom.geom_xpos[:, 0, 2] = torch.tensor([.001, .025, .025, .025, .025, .025])
    gravity = torch.tensor([[0.,0.,-1.]]*6)
    gravity[3, 2] = 0.
    position = torch.tensor([[0.,0.,.11]]*6)
    position[4, 2] = .05
    robot = SimpleNamespace(data=SimpleNamespace(data=geom, projected_gravity_b=gravity,
                                                 root_link_pos_w=position))
    env = SimpleNamespace(scene=Scene(robot=robot, feet=Contact()), command_manager=Commands(),
        episode_length_buf=torch.full((6,), 10), step_dt=.02, extras={'log':{}})
    reward = term(env, 'feet', 'unused', .025, 'twist', .01, reward_air_time=True)
    assert torch.allclose(reward, torch.tensor([.04, 1., 0., 0., 0., 0.]))
    # A real high swing stays credited while descending inside the time window.
    geom.geom_xpos[1, 0, 2] = .002
    reward = term(env, 'feet', 'unused', .025, 'twist', .01, reward_air_time=True)
    assert reward[1] == 1.
    term.reset(torch.tensor([1]))
    reward = term(env, 'feet', 'unused', .025, 'twist', .01, reward_air_time=True)
    assert torch.isclose(reward[1], torch.tensor(.08))
    blended = term(env, 'feet', 'unused', .025, 'twist', .01,
                   reward_air_time=True, air_time_quality_fraction=.25)
    assert torch.isclose(blended[1], torch.tensor(.77))
    assert blended[5] == 0.
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_sole_support_v9')
    parent = make_xgo_bam_env_cfg(repair_variant='gait_payload_v8')
    assert cfg.actions == parent.actions and cfg.observations == parent.observations
    assert cfg.events == parent.events and cfg.commands == parent.commands
    assert cfg.scene.entities['robot'].articulation == parent.scene.entities['robot'].articulation
    assert 'foot_clearance' not in cfg.rewards
    assert cfg.rewards['air_time'].params['reward_air_time']
    demand = make_xgo_bam_env_cfg(repair_variant='gait_sole_demand_v10')
    assert demand.actions == cfg.actions
    assert demand.rewards['action_rate_l2'] == cfg.rewards['action_rate_l2']
    assert demand.rewards['hd_slew_demand'].weight == -.5


def test_bilateral_reward_cannot_trade_weak_foot_for_repeated_strong_steps():
    class Scene(dict):
        terrain = SimpleNamespace(env_origins=torch.zeros(1, 3))
    class Contact:
        data = SimpleNamespace(found=torch.ones(1,2), last_air_time=torch.full((1,2), .2))
        landing = torch.tensor([[True, False]])
        def compute_first_contact(self, dt):
            return self.landing
    class Commands:
        value = torch.tensor([[.1, 0., 0.]])
        def get_command(self, name):
            return self.value
    term = hd_sole_swing_height.__new__(hd_sole_swing_height)
    term.geom_ids, term.vertices = [0,1], [torch.zeros(1,3)]*2
    term.peak, term.completed_peak = torch.zeros(1,2), torch.zeros(1,2)
    term.last_landed = torch.full((1,), -1, dtype=torch.long)
    data = SimpleNamespace(geom_xmat=torch.eye(3).repeat(1,2,1,1), geom_xpos=torch.zeros(1,2,3))
    robot = SimpleNamespace(data=SimpleNamespace(data=data, projected_gravity_b=torch.tensor([[0.,0.,-1.]]),
                                                 root_link_pos_w=torch.tensor([[0.,0.,.11]])))
    contact, commands = Contact(), Commands()
    env = SimpleNamespace(scene=Scene(robot=robot, feet=contact), command_manager=commands,
                          episode_length_buf=torch.tensor([10]), step_dt=.02, extras={'log':{}})
    def score(foot, height):
        contact.landing[:] = False
        contact.landing[0,foot] = True
        term.peak[0,foot] = height
        return term(env, 'feet', 'unused', .025, 'twist', .01, reward_bilateral=True).item()
    assert score(0, .025) == 0
    assert abs(score(1, .005) - .2) < 1e-6
    assert abs(score(0, .025) - .2) < 1e-6
    assert score(0, .05) == 0  # Repeating the stronger foot cannot earn another pair.
    assert score(1, .025) == 1
    commands.value[:] = 0
    assert score(0, .025) == 0
    commands.value[0,0] = .1
    assert score(1, .025) == 0  # Idle discarded the old pair.
    term.reset()
    assert term.completed_peak.sum() == 0 and term.last_landed.item() == -1
    term.completed_peak[:] = torch.tensor([[.025, .005]])
    term.peak_age = torch.zeros(1,2)
    contact.landing[:] = False
    cost = term(env, 'feet', 'unused', .025, 'twist', .01,
                reward_bilateral=True, bilateral_deficit=True)
    assert abs(cost.item() - .64) < 1e-6
    quality = term(env, 'feet', 'unused', .025, 'twist', .01,
                   reward_bilateral=True, reward_bilateral_quality=True)
    assert abs(quality.item() - .04) < 1e-6
    term.completed_peak[:] = .025
    assert term(env, 'feet', 'unused', .025, 'twist', .01,
                reward_bilateral=True, reward_bilateral_quality=True).item() == 1
    term.peak_age[:] = .8
    assert term(env, 'feet', 'unused', .025, 'twist', .01,
                reward_bilateral=True, reward_bilateral_quality=True).item() == 0
    term.peak_age[:] = 0
    term.completed_peak[:] = .025
    assert term(env, 'feet', 'unused', .025, 'twist', .01,
                reward_bilateral=True, bilateral_deficit=True).item() == 0
    commands.value[0,0] = .1
    robot.data.root_link_pos_w[0,2] = .08
    term.completed_peak[:] = .025
    term.peak_age[:] = 0
    assert term(env, 'feet', 'unused', .025, 'twist', .01,
                reward_bilateral=True, bilateral_deficit=True).item() == 1
    assert term(env, 'feet', 'unused', .025, 'twist', .01,
                reward_bilateral=True, reward_bilateral_quality=True).item() == 0
    robot.data.root_link_pos_w[0,2] = .11
    term.completed_peak[:] = .025
    term.peak_age[:] = .8
    assert term(env, 'feet', 'unused', .025, 'twist', .01,
                reward_bilateral=True, bilateral_deficit=True).item() == 1
    commands.value[:] = 0
    assert term(env, 'feet', 'unused', .025, 'twist', .01,
                reward_bilateral=True, bilateral_deficit=True).item() == 0
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_bilateral_v11')
    parent = make_xgo_bam_env_cfg(repair_variant='gait_sole_support_v9')
    assert cfg.actions == parent.actions and cfg.observations == parent.observations
    assert cfg.events == parent.events and cfg.commands == parent.commands
    assert cfg.rewards['hd_slew_demand'] == parent.rewards['hd_slew_demand']
    assert cfg.rewards['hd_sole_progress'].params['reward_bilateral']
    dense = make_xgo_bam_env_cfg(repair_variant='gait_weak_quality_v17')
    release = make_xgo_bam_env_cfg(repair_variant='gait_lift_release_v16')
    assert dense.actions == release.actions and dense.observations == release.observations
    assert dense.rewards['hd_weak_sole_quality'].weight == 4.
    assert dense.rewards['hd_weak_sole_quality'].params['reward_bilateral_quality']


def test_mirror_preserves_contract_and_is_an_involution():
    from tensordict import TensorDict
    from mjlab_microduck.tasks.symmetry import microduck_vel_symmetry, SYMMETRY_CFG
    obs = TensorDict({'actor': torch.randn(3,61), 'critic': torch.randn(3,76)}, batch_size=[3])
    action = torch.randn(3,14)
    mirrored, mirrored_action = microduck_vel_symmetry(None, obs, action)
    restored, restored_action = microduck_vel_symmetry(None, mirrored[3:], mirrored_action[3:])
    assert torch.equal(restored['actor'][3:], obs['actor'])
    assert torch.equal(restored_action[3:], action)
    assert torch.equal(mirrored['actor'][3:,48], obs['actor'][:,48])
    assert torch.equal(mirrored['actor'][3:,50], -obs['actor'][:,50])
    assert torch.equal(mirrored_action[3:,0:5], -action[:,9:14])
    assert not SYMMETRY_CFG['use_data_augmentation']  # Privileged critic is not mirrored.
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_bilateral_mirror_v12')
    parent = make_xgo_bam_env_cfg(repair_variant='gait_bilateral_v11')
    assert cfg.rewards == parent.rewards and cfg.actions == parent.actions
    assert cfg.observations == parent.observations and cfg.events == parent.events
    lift = make_xgo_bam_env_cfg(repair_variant='gait_bilateral_lift_v13')
    assert lift.actions == cfg.actions and lift.observations == cfg.observations
    assert lift.rewards['hd_weak_sole_deficit'].weight < 0
    assert lift.rewards['hd_weak_sole_deficit'].params['bilateral_deficit']
    stage = make_xgo_bam_env_cfg(repair_variant='gait_bilateral_stage20_v14')
    assert stage.actions == lift.actions and stage.commands == lift.commands
    assert stage.rewards['hd_weak_sole_deficit'].weight == -8.
    assert stage.rewards['air_time'].params['threshold_max'] == .5
    for name in ('air_time', 'foot_swing_height', 'hd_sole_progress', 'hd_weak_sole_deficit'):
        assert stage.rewards[name].params['target_height'] == .020
