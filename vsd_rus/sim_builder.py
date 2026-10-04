# sim_builder.py
"""
Общая сборка WholeBodyModel из merged-params (physiology + patient).
Используется run_simulation_parallel.simulate_scenario и
tests/debug_whole_body.run_scenario, чтобы избежать дублирования
распаковки секций YAML.
"""
from whole_body import WholeBodyModel

_BARO_KEYS = ('k_hr', 'k_inotropy', 'k_vasomotor',
              'tau_hr', 'tau_inotropy', 'tau_vaso')


def build_model_from_params(params: dict) -> WholeBodyModel:
    """Строит WholeBodyModel строго из merged-конфига."""
    heart_cfg    = dict(params['heart'])
    lungs_cfg    = dict(params['lungs'])
    baro_cfg     = dict(params['baroreflex'])
    blood_cfg    = dict(params['blood'])
    periph_cfg   = dict(params['peripheral'])
    liver_cfg    = dict(params['liver'])
    kidney_cfg   = dict(params['kidney'])
    brain_cfg    = dict(params['brain'])
    gitract_cfg  = dict(params['gitract'])
    ge_cfg       = dict(params['gas_exchange'])
    systemic_cfg = dict(params['systemic'])
    jugular_cfg  = dict(params['jugular_vein'])

    # --- top-level overrides из patient_*.yaml ---
    heart_cfg['R_vsd'] = params['vsd_resistance']
    heart_cfg['hr']    = params.get('HR_base', heart_cfg['hr'])
    for key in ('E_max_rv', 'E_max_lv', 'EDV_rv',
                'R_venous_sys', 'R_venous_pulm', 'R_tricuspid',
                'rv_hypertrophy_sensitivity'):
        if key in params:
            heart_cfg[key] = params[key]

    lungs_cfg['flow_dependent_resistance'] = bool(params['flow_dependent_lungs'])
    lungs_cfg['pressure_remodel']          = bool(params['pressure_remodel'])
    for key in ('P_pa_threshold', 'pressure_sensitivity',
                'R_remodel_max', 'tau_remodel', 'flow_sensitivity'):
        if key in params:
            lungs_cfg[key] = params[key]

    baro_cfg['HR_base'] = params.get('HR_base', baro_cfg['HR_base'])
    for key in _BARO_KEYS:
        if key in params:
            baro_cfg[key] = params[key]
    if 'k_inotropy_pulm' in params:
        baro_cfg['k_inotropy_pulm'] = params['k_inotropy_pulm']

    return WholeBodyModel(
        heart_params=heart_cfg,
        lungs_params=lungs_cfg,
        baroreflex_params=baro_cfg,
        blood_params=blood_cfg,
        peripheral_params=periph_cfg,
        liver_params=liver_cfg,
        kidney_params=kidney_cfg,
        brain_params=brain_cfg,
        gitract_params=gitract_cfg,
        gas_exchange_params=ge_cfg,
        jugular_params=jugular_cfg,
        flow_dependent_lungs=bool(params['flow_dependent_lungs']),
        target_MAP=systemic_cfg['target_MAP'],
        target_CO=systemic_cfg['target_CO'],
        C_sys_art=systemic_cfg['C_sys_art'],
        C_pul_ven=systemic_cfg['C_pul_ven'],
        P_sa0=systemic_cfg['P_sa0'],
        P_sv0=systemic_cfg['P_sv0'],
        P_pv0=systemic_cfg['P_pv0'],
        SYS_VEN_FRACTION=systemic_cfg['SYS_VEN_FRACTION'],
        C_sys_ven_eff=systemic_cfg['C_sys_ven_eff'],
        VO2_rest=systemic_cfg['VO2_rest'],
        RQ=systemic_cfg['RQ'],
        occlusion_factor=systemic_cfg['occlusion_factor'],
        method=params['simulation']['method'],
        fluid_intake_rate=systemic_cfg['fluid_intake_rate'],
        insensible_loss_rate=systemic_cfg['insensible_loss_rate'],
    )


def extract_simulation_config(params: dict) -> dict:
    """
    Собирает симуляционные константы из params['simulation'].
    Возвращает словарь с ключами:
        t_span, n_samples_t, t_calib, max_step, rtol, atol, method, steady_frac    
    """
    sim = dict(params['simulation'])
    pressure_remodel = bool(params.get('pressure_remodel', False))

    # t_span / n_samples — разные для ремоделирования
    if pressure_remodel and 't_span_remodel' in sim:
        t_span    = tuple(sim['t_span_remodel'])
        n_samples = int(sim['n_samples_remodel'])
    else:
        t_span    = tuple(sim['t_span'])
        n_samples = int(sim['n_samples_t'])

    # t_calib — адаптивный для healthy (короткая калибровка)
    t_calib = float(sim['t_calib'])
    if not pressure_remodel and t_calib >= 600:
        t_calib = float(sim.get('t_calib_healthy', 400.0))

    return {
        't_span':       t_span,
        'n_samples_t':  n_samples,
        't_calib':      t_calib,     # ← уже эффективный
        'max_step':     float(sim['max_step']),
        'rtol':         float(sim['rtol']),
        'atol':         float(sim['atol']),
        'method':       str(sim['method']),
        'steady_frac':  float(sim['steady_frac']),
    }