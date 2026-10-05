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

_REQUIRED_PERIPHERAL_KEYS = (
    'R_base', 'C_tissue', 'P_tissue0', 'O2_norm',
    'k_O2_autoreg', 'R_min_factor', 'R_myogenic_min_factor', 
    'tau_autoreg', 'k_P_myogenic', 'P_sa_norm', 'R_max_factor',
    'P_myogenic_deadband', 'VO2_base', 'VO2_basal_frac',
    'V_tissue_eff', 'C_a_O2_norm', 'C_O2_lactate_threshold',
    'k_lactate_prod', 'k_lactate_clear', 'k_lactate_release',
    'C_lactate0', 'C_O2_local0',
)

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
                'R_remodel_max', 'tau_remodel', 'flow_sensitivity', 'k_rarefaction'):
        if key in params:
            lungs_cfg[key] = params[key]

    baro_cfg['HR_base'] = params.get('HR_base', baro_cfg['HR_base'])
    for key in _BARO_KEYS:
        if key in params:
            baro_cfg[key] = params[key]
    if 'k_inotropy_pulm' in params:
        baro_cfg['k_inotropy_pulm'] = params['k_inotropy_pulm']

    # --- R_sys_peripheral: явный override или авто-калибровка ---
    R_sys_peripheral = systemic_cfg.get('R_sys_peripheral')
    if R_sys_peripheral is None:
        R_sys_peripheral = WholeBodyModel.auto_calibrate_R_sys_peripheral(
            target_MAP=systemic_cfg['target_MAP'],
            target_CO=systemic_cfg['target_CO'],
        )
    # --- peripheral.R_base: null → alias на R_sys_peripheral ---
    # Периферическое русло = systemic peripheral bed,
    # поэтому R_base по умолчанию равен R_sys_peripheral.
    # Задать независимое значение: peripheral.R_base: <число> в YAML.
    if periph_cfg.get('R_base') is None:
        periph_cfg['R_base'] = R_sys_peripheral

    missing = [k for k in _REQUIRED_PERIPHERAL_KEYS if k not in periph_cfg]
    if missing:
        raise ValueError(
            f"sim_builder: peripheral — отсутствуют ключи {missing}."
        )
    none_keys = [k for k in _REQUIRED_PERIPHERAL_KEYS if periph_cfg[k] is None]
    if none_keys:
        raise ValueError(
            f"sim_builder: peripheral — ключи не должны быть None: {none_keys}."
        )

    # --- substance_names: источник — blood.initial_concentrations ---
    if 'initial_concentrations' not in blood_cfg:
        raise ValueError(
            "sim_builder: blood.initial_concentrations обязателен "
            "(WholeBodyModel больше не выводит substance_names "
            "автоматически)."
        )
    substance_names = list(blood_cfg['initial_concentrations'].keys())

    return WholeBodyModel(
        heart_params=heart_cfg,
        lungs_params=lungs_cfg,
        liver_params=liver_cfg,
        kidney_params=kidney_cfg,
        blood_params=blood_cfg,
        gitract_params=gitract_cfg,
        brain_params=brain_cfg,
        baroreflex_params=baro_cfg,
        gas_exchange_params=ge_cfg,
        peripheral_params=periph_cfg,
        jugular_params=jugular_cfg,
        target_MAP=systemic_cfg['target_MAP'],
        target_CO=systemic_cfg['target_CO'],
        C_sys_art=systemic_cfg['C_sys_art'],
        C_pul_ven=systemic_cfg['C_pul_ven'],
        P_sa0=systemic_cfg['P_sa0'],
        P_sv0=systemic_cfg['P_sv0'],
        P_pv0=systemic_cfg['P_pv0'],
        SYS_VEN_FRACTION=systemic_cfg['SYS_VEN_FRACTION'],
        C_sys_ven_eff=systemic_cfg['C_sys_ven_eff'],
        R_sys_peripheral=R_sys_peripheral,
        fluid_intake_rate=systemic_cfg['fluid_intake_rate'],
        insensible_loss_rate=systemic_cfg['insensible_loss_rate'],
        VO2_rest=systemic_cfg['VO2_rest'],
        RQ=systemic_cfg['RQ'],
        occlusion_factor=systemic_cfg['occlusion_factor'],
        substance_names=substance_names,
        method=params['simulation']['method'],
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