# sim_builder.py
"""
Общая сборка WholeBodyModel из merged-params (physiology + patient).
Используется run_simulation_parallel.simulate_scenario и
tests/debug_whole_body.run_scenario, чтобы избежать дублирования
распаковки секций YAML.
"""
from whole_body import WholeBodyModel
from kidney import KidneyHemodynamic

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

_REQUIRED_HEART_KEYS = (
    'hr', 'hr_min', 'hr_max',
    'E_max_la', 'E_min_la', 'E_max_ra', 'E_min_ra',
    'E_max_lv', 'E_min_lv', 'E_max_rv', 'E_min_rv',
    'V0_la', 'V0_lv', 'V0_ra', 'V0_rv',
    'EDV_la', 'EDV_lv', 'EDV_ra', 'EDV_rv',
    'R_mitral', 'R_aortic', 'R_tricuspid', 'R_pulmonary',
    'R_venous_sys', 'R_venous_pulm',
    'k_valve', 'rv_hypertrophy_sensitivity',
    'k_lv_sympathetic', 'k_lv_parasympathetic',
    'k_rv_sympathetic', 'k_rv_parasympathetic', 'k_atria_inotropy',
    # --- RV remodeling ---
    'rv_hypertrophy_cap', 'rv_dilation_gain', 'rv_compliance_gain',
    'baro_rv_cap', 'rv_emax_rel_lv_cap',
)

_REQUIRED_LIVER_KEYS = (
    'R_ha', 'R_pv_base', 'R_hv_base',
    'C', 'C_portal',
    'P_hv0', 'P_portal0', 'V_liver',
    'albumin_prod_base', 'bilirubin_clearance_base',
    'ammonia_clearance_base', 'PS_lac',
    'C_bilirubin0', 'C_ammonia0', 'C_albumin0',
    'k_uptake_bil', 'k_uptake_amm',
    'k_deg_alb', 'k_release_alb',
)

_REQUIRED_KIDNEY_KEYS = (
    'GFR_base', 'P_autoreg',
    'autoreg_amplitude', 'autoreg_slope',
    'toxin_clearance_frac', 'volume_reabsorption_frac',
    'renal_resistance', 'basal_urine_output', 'RBF_target',
)

_REQUIRED_GITRACT_KEYS = (
    'R_art', 'R_cap', 'R_venous',
    'C_art', 'C_cap',
    'k_absorption_water', 'k_absorption_nutrients',
    'portal_pressure_sensitivity',
    'P_art0', 'P_cap0',
)

_REQUIRED_GAS_EXCHANGE_KEYS = (
    'P_alv_O2', 'P_alv_CO2', 'V_mix',
    'Hb', 'P50', 'n_hill', 'alpha_O2',
    'C_CO2_offset', 'k_CO2_slope',
)

_REQUIRED_JUGULAR_VEIN_KEYS = (
    'C', 'P0', 'V0', 'R_out',
    'C_O2_init', 'C_CO2_init', 'Hb',
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
                'rv_hypertrophy_sensitivity',
                'rv_hypertrophy_cap', 'rv_dilation_gain',
                'rv_compliance_gain', 'baro_rv_cap',
                'rv_emax_rel_lv_cap'):
        if key in params:
            heart_cfg[key] = params[key]

    missing = [k for k in _REQUIRED_HEART_KEYS if k not in heart_cfg]
    if missing:
        raise ValueError(f"sim_builder: heart — отсутствуют ключи {missing}.")
    none_keys = [k for k in _REQUIRED_HEART_KEYS if heart_cfg[k] is None]
    if none_keys:
        raise ValueError(f"sim_builder: heart — ключи не должны быть None: {none_keys}.")

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

    missing = [k for k in _REQUIRED_LIVER_KEYS if k not in liver_cfg]
    if missing:
        raise ValueError(
            f"sim_builder: liver — отсутствуют ключи {missing}."
        )
    none_keys = [k for k in _REQUIRED_LIVER_KEYS if liver_cfg[k] is None]
    if none_keys:
        raise ValueError(
            f"sim_builder: liver — ключи не должны быть None: {none_keys}."
        )

    missing = [k for k in _REQUIRED_GITRACT_KEYS if k not in gitract_cfg]
    if missing:
        raise ValueError(
            f"sim_builder: gitract — отсутствуют ключи {missing}."
        )
    none_keys = [k for k in _REQUIRED_GITRACT_KEYS if gitract_cfg[k] is None]
    if none_keys:
        raise ValueError(
            f"sim_builder: gitract — ключи не должны быть None: {none_keys}."
        )    

    missing = [k for k in _REQUIRED_GAS_EXCHANGE_KEYS if k not in ge_cfg]
    if missing:
        raise ValueError(
            f"sim_builder: gas_exchange — отсутствуют ключи {missing}."
        )
    none_keys = [k for k in _REQUIRED_GAS_EXCHANGE_KEYS if ge_cfg[k] is None]
    if none_keys:
        raise ValueError(
            f"sim_builder: gas_exchange — ключи не должны быть None: {none_keys}."
        )

    missing = [k for k in _REQUIRED_JUGULAR_VEIN_KEYS if k not in jugular_cfg]
    if missing:
        raise ValueError(
            f"sim_builder: jugular_vein — отсутствуют ключи {missing}."
        )
    none_keys = [k for k in _REQUIRED_JUGULAR_VEIN_KEYS if jugular_cfg[k] is None]
    if none_keys:
        raise ValueError(
            f"sim_builder: jugular_vein — ключи не должны быть None: {none_keys}."
        )

    # --- substance_names: источник — blood.initial_concentrations ---
    if (('initial_concentrations' not in blood_cfg)
            or blood_cfg['initial_concentrations'] is None):
        raise ValueError(
            "sim_builder: blood.initial_concentrations обязателен "
            "и не может быть None."
        )
    substance_names = list(blood_cfg['initial_concentrations'].keys())

    # --- kidney.renal_resistance: null → авто-калибровка ---
    missing = [k for k in _REQUIRED_KIDNEY_KEYS if k not in kidney_cfg]
    if missing:
        raise ValueError(f"sim_builder: kidney — отсутствуют ключи {missing}.")
    if kidney_cfg.get('renal_resistance') is None:
        kidney_cfg['renal_resistance'] = \
            KidneyHemodynamic.auto_calibrate_renal_resistance(
                P_autoreg=kidney_cfg['P_autoreg'],
                RBF_target=kidney_cfg['RBF_target'],
            )
    none_keys = [k for k in _REQUIRED_KIDNEY_KEYS if kidney_cfg[k] is None]
    if none_keys:
        raise ValueError(
            f"sim_builder: kidney — ключи не должны быть None: {none_keys}."
        )    

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