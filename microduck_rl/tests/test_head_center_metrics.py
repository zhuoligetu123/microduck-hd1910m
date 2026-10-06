import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from replay_hd1910 import head_center_metrics, head_optical_pitch_deg


def test_sustained_roll_fails_but_zero_mean_shake_passes():
    errors = np.zeros((100,4))
    errors[:,3] = np.radians(22)
    assert not head_center_metrics(errors)['head_center_check_passed']
    errors[::2,3] *= -1
    assert head_center_metrics(errors)['head_center_check_passed']
    assert not head_center_metrics([])['head_center_check_passed']


def test_head_optical_pitch_uses_camera_forward_axis():
    neutral = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    assert head_optical_pitch_deg(neutral) == 0.0
    raised = np.diag([-1., 1., -1.])
    assert head_optical_pitch_deg(raised) == 90.0
