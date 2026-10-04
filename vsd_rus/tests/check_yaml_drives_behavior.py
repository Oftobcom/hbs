"""
Проверяет, что YAML — единственный источник параметров:
изменение physiology.yaml (через overrides) меняет
результат extract_simulation_config и build_model_from_params.
"""
import sys
from pathlib import Path

# --- ROOT: родительская директория тестов, откуда импортируются модули ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physio_config import load_physiology, load_all_patients
from sim_builder import extract_simulation_config, build_model_from_params


def test_t_span_from_yaml():
    phys = load_physiology()
    pats = load_all_patients(base_physiology=phys)
    p_healthy = pats['Здоровый']
    p_eisen   = pats['Эйзенменгер, декомпенсированный']

    sc_h = extract_simulation_config(p_healthy)
    sc_e = extract_simulation_config(p_eisen)

    assert sc_h['t_span'] == (0.0, 1800.0), sc_h['t_span']
    assert sc_h['n_samples_t'] == 12001
    assert sc_h['t_calib'] == 400.0    # adaptive for healthy

    assert sc_e['t_span'] == (0.0, 1500.0), sc_e['t_span']
    assert sc_e['n_samples_t'] == 30000
    assert sc_e['t_calib'] == 600.0    # no adaptive


def test_overrides_change_behavior():
    """Если поменять t_span через overrides, extract_simulation_config
    должен вернуть новое значение (для healthy без pressure_remodel)."""
    phys = load_physiology(overrides={
        'simulation': {'t_span': [0.0, 42.0], 'n_samples_t': 100}
    })
    pats = load_all_patients(base_physiology=phys)
    sc = extract_simulation_config(pats['Здоровый'])
    assert sc['t_span'] == (0.0, 42.0)
    assert sc['n_samples_t'] == 100


def test_build_model_reads_yaml():
    """Изменение E_max_lv в YAML должно отразиться в модели."""
    phys = load_physiology()
    pats = load_all_patients(base_physiology=phys)
    p = pats['Здоровый']
    m1 = build_model_from_params(p)
    assert m1.heart.E_max_base['LV'] == 3.5

    # override
    phys2 = load_physiology(overrides={'heart': {'E_max_lv': 5.0}})
    pats2 = load_all_patients(base_physiology=phys2)
    m2 = build_model_from_params(pats2['Здоровый'])
    assert m2.heart.E_max_base['LV'] == 5.0


if __name__ == "__main__":
    test_t_span_from_yaml()
    test_overrides_change_behavior()
    test_build_model_reads_yaml()
    print("YAML drives behavior: OK")