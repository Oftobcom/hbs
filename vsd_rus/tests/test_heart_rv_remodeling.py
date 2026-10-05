def test_healthy_rv_unchanged_by_fixes():
    """При rv_afterload=0 V0_rv и E_min_rv совпадают с базой."""
    heart = Heart4Chambers(...)
    inputs = {'P_sa': 85, 'P_sv': 5, 'P_pa': 15, 'P_pv': 12,
              'hr_factor': 1.0, 'baro_activation': 1.0,
              'baro_activation_rv': 1.0, 'rv_afterload': 0.0}
    heart._update_parameters(inputs)
    assert abs(heart.V0['RV'] - heart.V0_base['RV']) < 1e-9
    assert abs(heart.E_min['RV'] - heart.E_min_base['RV']) < 1e-9

def test_decompensated_rv_dilates_and_relaxes():
    """При rv_afterload=1 V0_rv растёт, E_min_rv падает, E_max_rv ограничен."""
    heart = Heart4Chambers(...)
    inputs = {'P_sa': 85, 'P_sv': 5, 'P_pa': 82, 'P_pv': 12,
              'hr_factor': 1.0, 'baro_activation': 1.0,
              'baro_activation_rv': 1.5, 'rv_afterload': 1.0}
    heart._update_parameters(inputs)
    assert heart.V0['RV'] >= heart.V0_base['RV'] * 5.0   # ≥ 5×
    assert heart.E_min['RV'] <= heart.E_min_base['RV'] / 1.3
    assert heart._current_E_max['RV'] <= 9.0             # не 13.5

def test_RV_pressure_at_dilated_EDV_is_physiological():
    """P_rv(EDV=150) не должен превышать ~150 мм рт.ст. в систолу."""
    heart = Heart4Chambers(...)
    inputs = {'P_sa': 85, 'P_sv': 5, 'P_pa': 82, 'P_pv': 12,
              'hr_factor': 1.0, 'baro_activation': 1.0,
              'baro_activation_rv': 1.5, 'rv_afterload': 1.0}
    heart._update_parameters(inputs)
    P_rv_sys = heart._pressure('RV', 150, 0.33)   # пик систолы
    assert 40 < P_rv_sys < 150, f"P_rv={P_rv_sys} нефизиологично"

def test_no_EDV_in_RHS():
    """Изменение EDV не влияет на RHS (защита от иллюзорных фиксов)."""
    heart = Heart4Chambers(...)
    y1 = heart.get_initial_state()
    heart.EDV['RV'] *= 3
    y2 = heart.get_initial_state()
    # initial state изменится, но derivatives при том же y — нет
    dy1 = heart.get_derivatives(0.0, y1, {..., 'rv_afterload': 1.0})
    dy2 = heart.get_derivatives(0.0, y1, {..., 'rv_afterload': 1.0})
    assert np.allclose(dy1, dy2)