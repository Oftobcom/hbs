def test_V_mix_not_in_V_blood():
    """V_mix не увеличивает V_blood_total."""
    from physio_config import load_physiology, load_patient
    from sim_builder import build_model_from_params
    p = load_patient('config/patient_001.yaml', base=load_physiology())
    model = build_model_from_params(p)
    y0 = model.get_initial_state()
    out = model.compute_outputs(0.0, y0)
    # V_blood = сумма физических компартментов, V_mix=200 в неё НЕ входит
    assert 5000 < out['V_blood'] < 6500, \
        f"V_blood={out['V_blood']:.0f} — подозрение на +V_mix"

def test_derivatives_shape_and_finite():
    """derivatives() возвращает корректный вектор."""
    import numpy as np
    from physio_config import load_physiology, load_patient
    from sim_builder import build_model_from_params
    p = load_patient('config/patient_001.yaml', base=load_physiology())
    model = build_model_from_params(p)
    y0 = model.get_initial_state()
    dy = model.derivatives(0.0, y0)
    assert dy.shape == y0.shape, f"{dy.shape} vs {y0.shape}"
    assert np.all(np.isfinite(dy)), f"non-finite: {np.argwhere(~np.isfinite(dy))}"

def test_derivatives_gas_exchange_slot():
    """Проверка, что слот gas_exchange получает производные именно C_a."""
    import numpy as np
    from physio_config import load_physiology, load_patient
    from sim_builder import build_model_from_params
    p = load_patient('config/patient_001.yaml', base=load_physiology())
    model = build_model_from_params(p)
    y0 = model.get_initial_state()
    sl = model.idx['gas_exchange']
    dy = model.derivatives(0.0, y0)
    # Если слоты перепутаны, dy[sl] содержит d_sys_art, d_sys_ven —
    # их величина не имеет отношения к C_a_O2 порядка 0.2.
    # Смотрим, что dC_a/dt по модулю разумно мал (state = equilibrium).
    assert abs(dy[sl][0]) < 0.1, f"dC_a_O2/dt={dy[sl][0]} — вероятно, слот перепутан"