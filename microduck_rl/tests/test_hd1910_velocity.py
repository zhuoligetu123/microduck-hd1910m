"""Factory invariants only, synthetic fixture is not hardware calibration."""
from pathlib import Path
import pytest
from mjlab_microduck.tasks.microduck_hd1910_env_cfg import make_hd1910_velocity_env_cfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

ROOT = Path(__file__).resolve().parents[1]


def test_velocity_preserves_open_source_contract():
    cfg=make_hd1910_velocity_env_cfg(ROOT/'config/hd1910_smoke_fixture.json')
    stock=make_microduck_velocity_env_cfg()
    assert cfg.observations == stock.observations
    assert cfg.actions == stock.actions
    assert cfg.rewards == stock.rewards
    assert cfg.decimation*cfg.sim.mujoco.timestep == pytest.approx(.02)
    assert cfg.scene.entities['robot'].articulation.actuators != stock.scene.entities['robot'].articulation.actuators
    assert 'expand_bam_friction_fields' not in cfg.events
    entity = cfg.scene.entities['robot'].build()
    assert len(entity.spec.actuators) == 14
    with pytest.raises(ValueError):
        make_hd1910_velocity_env_cfg(ROOT/'config/hd1910_identification.json')
