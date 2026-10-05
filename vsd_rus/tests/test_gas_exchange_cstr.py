import sys
from pathlib import Path

# --- ROOT: родительская директория тестов, откуда импортируются модули ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from gas_exchange import GasExchange


def _make_ge(V_mix=200.0):
    return GasExchange(
        P_alv_O2=100.0, P_alv_CO2=40.0, Hb=15.0,
        P50=26.8, n_hill=2.7, alpha_O2=0.003,
        C_CO2_offset=0.22, k_CO2_slope=0.0065,
        V_mix=V_mix,
    )


def test_no_shunt_steady():
    """Без шунта стационарное C_a = C_pv."""
    ge = _make_ge()
    C_pv = ge._C_O2_from_P(100.0)
    state = np.array([C_pv, ge._C_CO2_from_P(40.0)])
    deriv = ge.get_derivatives(0.0, state, {
        'C_v_O2': 0.148, 'C_v_CO2': 0.52,
        'Q_p': 60.0, 'Q_shunt': 0.0,
    })
    assert abs(deriv[0]) < 1e-9


def test_LR_shunt_Ca_does_not_exceed_Cpv():
    """L→R: C_a стабилизируется на C_pv, НЕ выше."""
    ge = _make_ge()
    C_pv = ge._C_O2_from_P(100.0)
    state = np.array([C_pv + 0.01, ge._C_CO2_from_P(40.0)])
    deriv = ge.get_derivatives(0.0, state, {
        'C_v_O2': 0.148, 'C_v_CO2': 0.52,
        'Q_p': 80.0, 'Q_shunt': +20.0,       # L→R 25%
    })
    # C_a выше C_pv — обязана падать (CSTR не имеет источника > C_pv)
    assert deriv[0] < 0


def test_LR_shunt_equilibrium_is_Cpv():
    """L→R: стационар C_a = C_pv (не 1.25·C_pv)."""
    ge = _make_ge()
    C_pv = ge._C_O2_from_P(100.0)
    state = np.array([C_pv, ge._C_CO2_from_P(40.0)])
    deriv = ge.get_derivatives(0.0, state, {
        'C_v_O2': 0.148, 'C_v_CO2': 0.52,
        'Q_p': 80.0, 'Q_shunt': +20.0,
    })
    assert abs(deriv[0]) < 1e-9


def test_RL_shunt_equilibrium():
    """R→L 25%: C_a = 0.75·C_pv + 0.25·C_v."""
    ge = _make_ge()
    C_pv = ge._C_O2_from_P(100.0)
    C_v  = 0.148
    Ca_eq = 0.75 * C_pv + 0.25 * C_v
    state = np.array([Ca_eq, ge._C_CO2_from_P(40.0)])
    deriv = ge.get_derivatives(0.0, state, {
        'C_v_O2': C_v, 'C_v_CO2': 0.52,
        'Q_p': 60.0, 'Q_shunt': -20.0,       # R→L, Q_s=80
    })
    assert abs(deriv[0]) < 1e-9


def test_diastolic_RL_relaxes_to_Cv():
    """В диастоле Q_p = 0, Q_rl > 0 → C_a → C_v."""
    ge = _make_ge()
    state = np.array([0.20, ge._C_CO2_from_P(40.0)])
    deriv = ge.get_derivatives(0.0, state, {
        'C_v_O2': 0.10, 'C_v_CO2': 0.52,
        'Q_p': 0.0, 'Q_shunt': -20.0,
    })
    assert deriv[0] < 0


def test_readout_reads_state():
    """compute_effects НЕ пересчитывает C_a — читает из state."""
    ge = _make_ge()
    state = np.array([0.10, 0.55])
    out = ge.compute_effects(
        state=state, C_v_O2=0.15, C_v_CO2=0.52,
        Q_p=60.0, Q_shunt=-20.0,
    )
    assert abs(out['C_a_O2'] - 0.10) < 1e-12
    assert out['SaO2'] < 0.70


def test_V_mix_does_not_shift_equilibrium():
    """Стационар CSTR не зависит от V_mix."""
    args = {'C_v_O2': 0.148, 'C_v_CO2': 0.52, 'Q_p': 60.0, 'Q_shunt': -20.0}
    eq_small = _make_ge(V_mix=100.0)._equilibrium_state(**args)
    eq_large = _make_ge(V_mix=400.0)._equilibrium_state(**args)
    assert abs(eq_small[0] - eq_large[0]) < 1e-12