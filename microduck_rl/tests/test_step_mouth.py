import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_step import MouthPreview, ROOT, mouth_target


def test_mouth_runtime_range_and_period():
    assert mouth_target(0) == 0.
    assert math.isclose(mouth_target(.5), math.radians(15))
    assert math.isclose(mouth_target(1), math.radians(30))
    assert math.isclose(mouth_target(2), mouth_target(0))
    assert all(mouth_target(i/100) >= 0 for i in range(401))


def test_visual_jaw_does_not_change_dynamics_or_policy_contract():
    m = mujoco.MjModel.from_xml_path(str(ROOT/'src/mjlab_microduck/robot/microduck/scene_walk.xml'))
    mass = m.body_mass.copy()
    inertia = m.body_inertia.copy()
    preview = MouthPreview(m)
    original = m.geom_pos.copy()
    preview.update(mouth_target(1))
    assert not np.allclose(m.geom_pos[preview.ids], original[preview.ids])
    np.testing.assert_array_equal(m.body_mass, mass)
    np.testing.assert_array_equal(m.body_inertia, inertia)
    assert m.nu == 14
    assert m.njnt == 15  # Free base plus 14 policy joints, no invented mouth actuator.
    preview.update(mouth_target(0))
    np.testing.assert_allclose(m.geom_pos, original, atol=1e-12)
