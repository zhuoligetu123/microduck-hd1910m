import math

import mujoco
import numpy as np
import pytest

from mjlab_microduck.actuator.payload_uncertainty import (
    PROFILE, SCENARIOS, apply_mass_scenario, mass_contract)
from mjlab_microduck.actuator.reference_hd1910 import make_hd1910_spec
from mjlab_microduck.tasks.hd1910_bam import make_xgo_bam_env_cfg


def test_inventory_not_added_again():
    items = PROFILE['confirmed_components']
    total = sum(items[name]['count'] * items[name]['each_g']
                for name in ('large_bearings', 'small_bearings'))
    total += items['head_board']['mass_g'] + items['back_battery']['mass_g']
    assert total == pytest.approx(170.6)
    assert total == PROFILE['listed_component_total_g']
    assert PROFILE['cad_component_masses_kg'] is None
    assert items['large_bearings']['head_roll_count'] == 2
    assert items['small_bearings']['count'] == 2
    assert len(mass_contract()['profile_sha256']) == 64


def test_original_bearings_and_electronics_are_already_present():
    model = make_hd1910_spec(clear_actuators=False).compile()
    for mesh_name, expected in (
        ('seeed_bearing__configuration__22x16x4', 11),
        ('seeed_bearing__configuration_default', 3),
        ('pcb__raspberry_pi_zero_2_w', 1),
        ('elec_rpi_robot_hat_pcb', 1),
        ('np_f970', 1)):
        mesh_id = model.mesh(mesh_name).id
        instances = {(int(model.geom_bodyid[g]), *np.round(model.geom_pos[g], 10),
                      *np.round(model.geom_quat[g], 10)) for g in range(model.ngeom)
                     if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                     and model.geom_dataid[g] == mesh_id}
        assert len(instances) == expected


def test_payload_training_preserves_policy_semantics_and_nominal_cad():
    parent = make_xgo_bam_env_cfg(repair_variant='gait_head_dc_stride_v4')
    cfg = make_xgo_bam_env_cfg(repair_variant='gait_payload_v8')
    assert cfg.actions == parent.actions
    assert cfg.observations == parent.observations
    assert cfg.commands == parent.commands
    robot = cfg.scene.entities['robot']
    original = parent.scene.entities['robot']
    assert robot.articulation == original.articulation
    assert robot.spec_fn.func is original.spec_fn.func
    assert robot.spec_fn.keywords == original.spec_fn.keywords
    np.testing.assert_array_equal(robot.spec_fn().compile().body_mass,
                                  original.spec_fn().compile().body_mass)
    assert 'randomize_mass_inertia' not in cfg.events
    assert 'head_com_range' not in cfg.curriculum
    for name, bounds in PROFILE['body_mass_scale_ranges'].items():
        event = cfg.events['payload_mass_' + name]
        assert event.mode == 'startup'
        np.testing.assert_allclose(np.exp(2*np.array(event.params['alpha_range'])), bounds)
    cfg_play = make_xgo_bam_env_cfg(play=True, repair_variant='gait_payload_v8')
    assert not any(name.startswith('payload_mass_') for name in cfg_play.events)


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_cpu_scenario_is_physical_and_deterministic(scenario):
    spec = make_hd1910_spec(clear_actuators=False)
    original = spec.compile()
    model = spec.compile()
    for name in PROFILE['body_mass_scale_ranges']:
        assert model.body(name).mass[0] > 0
    report = apply_mass_scenario(model, mujoco.MjData(model), scenario)
    for name, scale in report['link_scales'].items():
        body = model.body(name).id
        np.testing.assert_allclose(model.body_inertia[body], original.body_inertia[body]*scale)
        assert model.body_mass[body] == pytest.approx(original.body_mass[body]*scale)
    principal = model.body_inertia[model.body_mass > 0]
    assert np.all(principal > 0)
    assert np.all(2*principal.max(axis=1) <= principal.sum(axis=1) + 1e-12)
    repeat = spec.compile()
    assert apply_mass_scenario(repeat, mujoco.MjData(repeat), scenario) == report
    if scenario == 'cad_nominal':
        np.testing.assert_array_equal(model.body_mass, original.body_mass)
        np.testing.assert_array_equal(model.body_ipos, original.body_ipos)
    assert math.isfinite(report['scenario_mass_kg'])
