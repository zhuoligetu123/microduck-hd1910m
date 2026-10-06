import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_step import transition_action


def test_transition_endpoints_and_midpoint():
    action = np.array([1., -2.])
    np.testing.assert_array_equal(transition_action(action, 0., 2.), [0., 0.])
    np.testing.assert_allclose(transition_action(action, 1., 2.), action*.5)
    np.testing.assert_array_equal(transition_action(action, 2., 2.), action)
    np.testing.assert_array_equal(transition_action(action, 0., 0.), action)
    np.testing.assert_array_equal(action, [1., -2.])


def test_no_rate_limit_after_transition():
    for action in (np.array([2., -2.]), np.array([-2., 2.])):
        np.testing.assert_array_equal(transition_action(action, 3., 2.), action)


def test_smooth_monotone_weight():
    weights = np.array([transition_action(np.ones(1), t, 2.)[0]
                        for t in np.linspace(0, 2, 101)])
    assert np.all(np.diff(weights) >= 0)
    assert weights[1]-weights[0] < .0001
    assert weights[-1]-weights[-2] < .0001


@pytest.mark.parametrize('duration', [-1., float('nan'), float('inf')])
def test_invalid_duration(duration):
    with pytest.raises(ValueError):
        transition_action(np.ones(1), 0., duration)
