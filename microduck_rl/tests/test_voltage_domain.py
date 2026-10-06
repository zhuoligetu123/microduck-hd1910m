from copy import deepcopy
from dataclasses import asdict

import pytest
from mjlab_microduck.tasks.xgoduck_bam import make_xgo_bam_env_cfg, configure_voltage_domain


def test_only_voltage_changes_and_baseline_is_unchanged():
    base = make_xgo_bam_env_cfg(repair_variant='gait_luwu_curriculum_scaled_v21')
    candidate = deepcopy(base)
    configure_voltage_domain(candidate, 'static_home')
    before = asdict(base)
    after = asdict(candidate)
    motor = candidate.scene.entities['robot'].articulation.actuators[0]
    assert motor.vin_range == (6.6, 7.4)
    assert motor.vin_min == 6.6
    motor.vin_range = base.scene.entities['robot'].articulation.actuators[0].vin_range
    motor.vin_min = base.scene.entities['robot'].articulation.actuators[0].vin_min
    assert repr(asdict(candidate)) == repr(before)
    assert repr(after) != repr(before)
    configure_voltage_domain(base, 'nominal')
    assert repr(asdict(base)) == repr(before)
    with pytest.raises(ValueError):
        configure_voltage_domain(base, 'invalid')
