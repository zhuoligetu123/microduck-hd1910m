"""Intermediate screening must not silently accept a chattering policy."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'scripts/screen_hd1910_checkpoint.py'
spec = importlib.util.spec_from_file_location('checkpoint_screen', path)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def test_intermediate_screen_requires_motion_quality_and_all_cases():
    rows = [dict(case=name, baseline_check_passed=True, motion_quality_check_passed=True)
            for name in ('stand', 'forward', 'turn', 'backward', 'turn_right')]
    assert screen.qualified_case_count({'cases': rows}) == 5
    rows[0]['motion_quality_check_passed'] = False
    assert screen.qualified_case_count({'cases': rows}) == 4
    del rows[1]['motion_quality_check_passed']
    assert screen.qualified_case_count({'cases': rows}) == 3
    assert screen.qualified_case_count({'cases': rows[:4]}) == 0
    assert screen.qualified_case_count({'cases': rows[::-1]}) == 0
    assert screen.qualified_case_count({}) == 0
