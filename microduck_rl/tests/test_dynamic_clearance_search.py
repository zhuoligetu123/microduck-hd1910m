from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from search_dynamic_clearance import IDENTITY, shape_target


def test_identity_preserves_valid_policy_output():
    home = np.zeros(14)
    limits = np.tile([-1., 1.], (14, 1))
    action = np.linspace(-.09, .09, 14)
    np.testing.assert_allclose(shape_target(action, IDENTITY, home, home, limits), action)


def test_reference_never_bypasses_ranges_or_slew():
    home = np.zeros(14)
    limits = np.tile([-.08, .08], (14, 1))
    result = shape_target(np.ones(14)*3, IDENTITY, home, home, limits)
    assert np.max(np.abs(result)) <= .08
    limits[:] = [-1., 1.]
    result = shape_target(np.ones(14)*3, IDENTITY, home, home, limits)
    assert np.max(np.abs(result)) <= .1


def test_symmetric_shape_commutes_with_leg_reflection():
    perm = [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4]
    sign = np.array([-1]*5+[1, 1, -1, -1]+[-1]*5)
    rng = np.random.default_rng(4)
    action, previous = rng.normal(size=(2, 14))*.1
    params = np.array([1.3, .8, 1.5, .7, .1, -.2, .15, .2])
    limits = np.tile([-1., 1.], (14, 1))
    a = shape_target(action, params, previous, np.zeros(14), limits)
    b = shape_target(action[perm]*sign, params, previous[perm]*sign, np.zeros(14), limits)
    np.testing.assert_allclose(a[perm]*sign, b)


def test_gain_does_not_scale_slew_history_when_latent_requests_home():
    previous = np.zeros(14)
    previous[3] = 1.4
    params = IDENTITY.copy()
    params[2] = 1.5
    result = shape_target(np.zeros(14), params, previous, np.zeros(14),
                          np.tile([-1.6, 1.6], (14, 1)))
    assert result[3] < previous[3]
    np.testing.assert_allclose(result[3], 1.3)
