import importlib.util
from pathlib import Path


def test_stress_matrix_covers_both_engines_and_all_seeds():
    path = Path(__file__).parents[1] / 'scripts/iterate_hd1910.py'
    spec = importlib.util.spec_from_file_location('hd_iteration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = list(module.velocity_stress_cases((42, 7, 123)))
    assert len(rows) == 18
    assert len({name for name, _, _ in rows}) == 18
    for seed in (42, 7, 123):
        selected = [row for row in rows if row[2][1] == str(seed)]
        assert len(selected) == 6
        assert {script for _, script, _ in selected} == {
            'replay_hd1910.py', 'replay_hd1910_warp.py'}
    assert len(list(module.velocity_stress_cases((42,)))) == 6
    assert 'stress_v8.4_delay6' in {row[0] for row in rows}
