import json
import math
import pytest
from mjlab_microduck.actuator.radxa_alignment import JOINTS, alignment_contract, geometry_contract
from mjlab_microduck.tasks.xgoduck_bam import make_xgo_bam_env_cfg


def installation():
    return dict(calibration_verified=False, joints=[dict(name=n, id=i+1, zero_ticks=2048,
        direction=(-1 if i % 2 else 1)) for i, n in enumerate(JOINTS)])


def test_native_zero_roundtrip_and_no_double_offset(tmp_path):
    path = tmp_path / 'installation.json'
    data = installation()
    data['joints'][0]['zero_ticks'] = 2000
    path.write_text(json.dumps(data))
    contract = alignment_contract(path)
    assert contract['joints'][0]['model_offset_from_midpoint_rad'] == pytest.approx(48*math.tau/4096)
    assert contract['roundtrip_max_error_ticks'] == 0
    assert contract['bam_q_offset_rad'] == 0
    assert len(contract['policy_joint_names']) == 14
    assert not contract['calibration_verified']
    assert not contract['physical_alignment_tested']


@pytest.mark.parametrize('field,value', [('zero_ticks', 4096), ('direction', 0), ('id', True)])
def test_invalid_calibration_rejected(tmp_path, field, value):
    path = tmp_path / 'installation.json'
    data = installation()
    data['joints'][0][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        alignment_contract(path)


def test_geometry_and_uncertainty_preserved():
    result = geometry_contract(make_xgo_bam_env_cfg())
    assert result['adapted_total_mass_kg']-result['source_total_mass_kg'] == pytest.approx(.045)
    assert result['delay_steps'] == [3, 6]
    assert result['physics_dt_s'] == .005
    assert result['voltage_range_v'] == [7.4, 8.0]
    assert result['encoder_bias_range_rad'] == [-.015, .015]
    assert not result['physical_mass_measured']
