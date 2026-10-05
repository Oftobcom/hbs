# whole_body.py
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from organ_base import OrganModel
from heart import Heart4Chambers
from lungs import Lungs2Chamber
from liver import Liver
from kidney import KidneyHemodynamic
from blood import BloodPool
from gitract import GITract
from brain import Brain
from baroreflex import Baroreflex
from gas_exchange import GasExchange
from peripheral_tissues import PeripheralTissues
from jugular_vein import JugularVein


# =====================================================================
# Вспомогательные проверки
# =====================================================================

def _check_range(name: str, v, lo: float, hi: float, typical: str = "") -> float:
    """Fail-fast проверка числового параметра на попадание в [lo, hi]."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"WholeBodyModel: {name}={v!r} не число.")
    if not np.isfinite(v) or not (lo <= v <= hi):
        raise ValueError(
            f"WholeBodyModel: {name}={v} вне [{lo}, {hi}]. {typical}"
        )
    return v


def _check_fraction(name: str, v, typical: str = "") -> float:
    """Проверка на долю строго внутри (0, 1)."""
    v = float(v)
    if not np.isfinite(v) or not (0.0 < v < 1.0):
        raise ValueError(
            f"WholeBodyModel: {name}={v} вне (0, 1). {typical}"
        )
    return v


# =====================================================================
# WindkesselVessel — двухэлементная модель сосуда
# =====================================================================

class WindkesselVessel(OrganModel):
    """
    Двухэлементная модель Windkessel.

    Два режима работы (выбирается через mode):
      mode='P'  — состояние это давление P, dP/dt = (Qin − Qout)/C.
                  Используется для артерий и лёгочных вен.
      mode='V'  — состояние это объём V, dV/dt = Qin − Qout + dV_external.
                  Давление выводится: P = P0 + (V − V0)/C.
                  Используется для системных вен — «буфера» крови.
    """

    _C_MIN, _C_MAX = 1e-3, 1e5
    _P0_MIN, _P0_MAX = 0.0, 300.0
    _V0_MIN, _V0_MAX = 0.0, 1e5

    def __init__(self, C, P0, mode='P', V0=None):
        if mode not in ('P', 'V'):
            raise ValueError(
                f"WindkesselVessel: mode={mode!r} должен быть 'P' или 'V'."
            )
        self.mode = mode

        self.C = _check_range(
            "Windkessel.C", C, self._C_MIN, self._C_MAX,
            "мл/мм рт.ст., типично 1.5–550."
        )
        self.P0 = _check_range(
            "Windkessel.P0", P0, self._P0_MIN, self._P0_MAX,
            "мм рт.ст., типично 5–90."
        )

        if V0 is not None:
            V0 = _check_range(
                "Windkessel.V0", V0, self._V0_MIN, self._V0_MAX,
                "мл, типично 5–4000."
            )
            self.V0 = float(V0)
        else:
            self.V0 = self.C * self.P0

        self._current_outputs = {}

    def get_state_size(self):
        return 1

    def get_initial_state(self):
        if self.mode == 'P':
            return np.array([self.P0])
        return np.array([self.V0])

    def get_derivatives(self, t, state, inputs):
        Q_in = inputs.get('Q_in', 0.0)
        Q_out = inputs.get('Q_out', 0.0)

        if self.mode == 'P':
            P = state[0]
            dP = (Q_in - Q_out) / self.C
            self._current_outputs = {'P': P}
            return np.array([dP])

        # mode == 'V'
        V = state[0]
        dV = Q_in - Q_out
        dV += float(inputs.get('dV_external', 0.0))

        P = self.P0 + (V - self.V0) / self.C
        P = max(P, 0.0)
        self._current_outputs = {'P': P, 'V': V}
        return np.array([dV])

    def get_outputs(self, state):
        return self._current_outputs.copy()


# =====================================================================
# WholeBodyModel
# =====================================================================

class WholeBodyModel:
    """
    Полная модель организма здорового человека или с ДМЖП.
    """

    # --- Санити-пороги для валидации конфигурации ---
    _TARGET_MAP_MIN, _TARGET_MAP_MAX = 40.0, 200.0
    _TARGET_CO_MIN, _TARGET_CO_MAX = 20.0, 200.0

    _C_SYS_ART_MIN, _C_SYS_ART_MAX = 0.1, 20.0
    _C_PUL_VEN_MIN, _C_PUL_VEN_MAX = 1.0, 100.0
    _C_SYS_VEN_MIN, _C_SYS_VEN_MAX = 10.0, 5000.0

    _P_SYS_MIN, _P_SYS_MAX = 0.0, 200.0
    _P_PV_MIN, _P_PV_MAX = 0.0, 100.0


    _FLUID_RATE_MIN, _FLUID_RATE_MAX = 0.0, 1.0
    _VO2_REST_MIN, _VO2_REST_MAX = 0.0, 20.0
    _RQ_MIN, _RQ_MAX = 0.5, 1.5
    _OCCLUSION_MIN, _OCCLUSION_MAX = 0.0, 1.0

    _R_SYS_PERIPH_MIN, _R_SYS_PERIPH_MAX = 0.1, 100.0
    _BLOOD_V0_MIN, _BLOOD_V0_MAX = 1000.0, 15000.0

    # --- Пол объёма крови для расчёта концентраций ---
    # Ниже этого порога модель выходит за пределы физиологической
    # применимости (массивная кровопотеря, ошибка интегрирования).
    # Пол задаётся как доля от BloodPool.V0 — номинального объёма крови.
    _VB_FLOOR_FRAC  = 0.5      # доля V0_blood
    _VB_SOFT_WIDTH  = 0.1      # ширина перехода soft-plus, доля V0_blood
    _VB_SOFT_CLIP   = 30.0     # ограничение аргумента exp для защиты от overflow

    _VALID_METHODS = frozenset({
        'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA',
    })

    def __init__(self,
        # --- Organ configs (11) ---
        heart_params: dict,
        lungs_params: dict,
        liver_params: dict,
        kidney_params: dict,
        blood_params: dict,
        gitract_params: dict,
        brain_params: dict,
        baroreflex_params: dict,
        gas_exchange_params: dict,
        peripheral_params: dict,
        jugular_params: dict,
        # --- Systemic vascular (10) ---
        target_MAP: float,
        target_CO: float,
        C_sys_art: float,
        C_pul_ven: float,
        P_sa0: float,
        P_sv0: float,
        P_pv0: float,
        SYS_VEN_FRACTION: float,
        C_sys_ven_eff: float,
        R_sys_peripheral: float,
        # --- Fluid balance (2) ---
        fluid_intake_rate: float,
        insensible_loss_rate: float,
        # --- Metabolism (3) ---
        VO2_rest: float,
        RQ: float,
        occlusion_factor: float,
        # --- Substances (1) ---
        substance_names: list[str],
        *,
        method: str,
    ):

        # =================================================================
        # Валидация конфигурации — fail-fast при инициализации.
        # =================================================================

        # --- Целевые показатели ---
        target_MAP = _check_range(
            "target_MAP", target_MAP,
            self._TARGET_MAP_MIN, self._TARGET_MAP_MAX,
            "мм рт.ст., типично 85."
        )
        target_CO = _check_range(
            "target_CO", target_CO,
            self._TARGET_CO_MIN, self._TARGET_CO_MAX,
            "мл/с, типично 83."
        )

        # --- Сосудистые компартменты ---
        C_sys_art = _check_range(
            "C_sys_art", C_sys_art,
            self._C_SYS_ART_MIN, self._C_SYS_ART_MAX,
            "мл/мм рт.ст., типично 1.5."
        )
        C_pul_ven = _check_range(
            "C_pul_ven", C_pul_ven,
            self._C_PUL_VEN_MIN, self._C_PUL_VEN_MAX,
            "мл/мм рт.ст., типично 15."
        )
        C_sys_ven_eff = _check_range(
            "C_sys_ven_eff", C_sys_ven_eff,
            self._C_SYS_VEN_MIN, self._C_SYS_VEN_MAX,
            "мл/мм рт.ст., типично 550."
        )

        # --- Начальные давления ---
        P_sa0 = _check_range(
            "P_sa0", P_sa0, self._P_SYS_MIN, self._P_SYS_MAX,
            "мм рт.ст., типично 85."
        )
        P_sv0 = _check_range(
            "P_sv0", P_sv0, self._P_SYS_MIN, self._P_SYS_MAX,
            "мм рт.ст."
        )
        P_pv0 = _check_range(
            "P_pv0", P_pv0, self._P_PV_MIN, self._P_PV_MAX,
            "мм рт.ст."
        )

        SYS_VEN_FRACTION = _check_fraction(
            "SYS_VEN_FRACTION", SYS_VEN_FRACTION,
            "типично 0.63; строго (0, 1)."
        )

        # --- Жидкостный баланс ---
        fluid_intake_rate = _check_range(
            "fluid_intake_rate", fluid_intake_rate,
            self._FLUID_RATE_MIN, self._FLUID_RATE_MAX,
            "мл/с, типично 0.02."
        )
        insensible_loss_rate = _check_range(
            "insensible_loss_rate", insensible_loss_rate,
            self._FLUID_RATE_MIN, self._FLUID_RATE_MAX,
            "мл/с, типично 0.0."
        )

        # --- Метаболизм ---
        VO2_rest = _check_range(
            "VO2_rest", VO2_rest,
            self._VO2_REST_MIN, self._VO2_REST_MAX,
            "мл O2/с, типично 1.9 (сердце+печень+почки+ЖКТ)."
        )
        RQ = _check_range(
            "RQ", RQ, self._RQ_MIN, self._RQ_MAX,
            "дыхательный коэффициент, типично 0.8."
        )
        occlusion_factor = _check_range(
            "occlusion_factor", occlusion_factor,
            self._OCCLUSION_MIN, self._OCCLUSION_MAX,
            "0 — полная окклюзия, 1 — норма."
        )

        if not isinstance(method, str) or method not in self._VALID_METHODS:
            raise ValueError(
                f"WholeBodyModel: method={method!r} не входит в "
                f"{sorted(self._VALID_METHODS)}. "
                f"Задаётся в physiology.yaml: simulation.method. "
                f"Через sim_builder пробрасывается автоматически."
            )
        self.method = method

        if not isinstance(substance_names, (list, tuple)):
            raise ValueError(
                f"WholeBodyModel: substance_names должен быть list/tuple, "
                f"получено {type(substance_names).__name__}."
            )
        if len(substance_names) == 0:
            raise ValueError(
                "WholeBodyModel: substance_names пуст."
            )
        if not all(isinstance(s, str) and s for s in substance_names):
            raise ValueError(
                "WholeBodyModel: substance_names должен содержать "
                "непустые строки."
            )
        if len(set(substance_names)) != len(substance_names):
            raise ValueError(
                f"WholeBodyModel: substance_names содержит дубликаты: "
                f"{substance_names}."
            )

        self.substance_names = list(substance_names)
        self._substance_idx = {name: i for i, name in enumerate(substance_names)}

        # =================================================================
        # Инициализация органов
        # =================================================================

        # --- BloodPool initial state ---
        if 'V0' not in blood_params:
            raise ValueError(
                "WholeBodyModel: blood_params['V0'] обязателен. "
                "Дефолт удалён; добавьте blood.V0 в physiology.yaml "
                "(сейчас уже есть)."
            )
        V0_blood = _check_range(
            "blood.V0", blood_params['V0'],
            self._BLOOD_V0_MIN, self._BLOOD_V0_MAX,
            "мл, типично 5000–6500."
        )

        if 'initial_concentrations' not in blood_params:
            raise ValueError(
                "WholeBodyModel: blood_params['initial_concentrations'] "
                "обязателен."
            )
        initial_concentrations = dict(blood_params['initial_concentrations'])

        missing = [s for s in substance_names
                   if s not in initial_concentrations]
        if missing:
            raise ValueError(
                f"WholeBodyModel: blood.initial_concentrations "
                f"не содержит {missing}. "
                f"Все вещества из substance_names должны быть заданы "
                f"в YAML — DEFAULT_CONC удалён."
            )

        # --- Heart ---
        self.heart = Heart4Chambers(**heart_params)

        # --- Lungs ---
        self.lungs = Lungs2Chamber(**lungs_params)

        # --- Liver, Kidney, Blood, GITract, Brain ---
        self.liver   = Liver(**liver_params)
        self.kidney  = KidneyHemodynamic(**kidney_params)
        self.gitract = GITract(**gitract_params)
        self.brain   = Brain(**brain_params)
        self.blood = BloodPool(
            substance_names=self.substance_names,
            V0=V0_blood,
            initial_concentrations=initial_concentrations,
        )
        
        # --- Периферическое сопротивление ---
        R_sys_peripheral = _check_range(
            "R_sys_peripheral", R_sys_peripheral,
            self._R_SYS_PERIPH_MIN, self._R_SYS_PERIPH_MAX,
            "мм рт.ст.·с/мл, типично 2–6."
        )

        # --- Периферические ткани ---
        # R_base разрешён в sim_builder (null → R_sys_peripheral),
        # здесь peripheral_params приходит уже с числовым R_base.
        self.peripheral = PeripheralTissues(**peripheral_params)

        self.target_MAP = target_MAP
        self.target_CO = target_CO

        # --- Барорефлекс ---
        self.baroreflex = Baroreflex(**baroreflex_params)
        # --- Валидация контракта Baroreflex — fail-fast ---
        if self.baroreflex.get_state_size() < 1:
            raise RuntimeError(
                "WholeBodyModel: Baroreflex.get_state_size() < 1 — "
                "барорефлекс обязан иметь хотя бы одно состояние."
            )
        required_out = {'HR', 'HR_target', 'hr_factor',
                        'baro_activation', 'baro_activation_rv',
                        'R_sys_scale'}
        # get_outputs уже имеет fallback на пустой кэш, поэтому
        # проверка безопасна и на первом вызове.
        _probe = self.baroreflex.get_outputs(
            self.baroreflex.get_initial_state()
        )
        missing = required_out - set(_probe.keys())
        if missing:
            raise RuntimeError(
                f"WholeBodyModel: Baroreflex не выдаёт обязательные "
                f"ключи {sorted(missing)}."
            )
        
        # --- Системные артерии и лёгочные вены (P-mode) ---
        self.sys_art = WindkesselVessel(C=C_sys_art, P0=P_sa0, mode='P')
        self.pul_ven = WindkesselVessel(C=C_pul_ven, P0=P_pv0, mode='P')

        # --- Системные вены (V-mode) ---
        V_sv0 = SYS_VEN_FRACTION * V0_blood
        self.sys_ven = WindkesselVessel(
            C=C_sys_ven_eff, P0=P_sv0, mode='V', V0=V_sv0,
        )

        # --- Яремная вена ---
        self.jugular_vein = JugularVein(**jugular_params)

        self.VO2_rest = VO2_rest
        self.RQ = RQ
        self._occlusion_factor = occlusion_factor

        self.R_sys_peripheral = R_sys_peripheral
        self.fluid_intake_rate = fluid_intake_rate
        self.insensible_loss_rate = insensible_loss_rate

        # --- Газообмен ---
        ge = dict(gas_exchange_params)
        ge.pop('VO2_base', None)     # legacy-ключи, не поддерживаются
        ge.pop('VCO2_base', None)
        ge.pop('Q_norm', None)
        self.gas_exchange = GasExchange(**ge)

        # =================================================================
        # Сборка organ_list и индекс срезов состояния
        # =================================================================
        self.organ_list = [
            self.heart, self.lungs, self.liver, self.blood, self.gitract,
            self.brain, self.peripheral, self.baroreflex,
            self.sys_art, self.sys_ven, self.pul_ven, self.jugular_vein,
            self.gas_exchange,
        ]
        ORGAN_NAMES = [
            'heart', 'lungs', 'liver', 'blood', 'gitract',
            'brain', 'peripheral', 'baroreflex',
            'sys_art', 'sys_ven', 'pul_ven', 'jugular_vein',
            'gas_exchange',
        ]

        # Проверка синхронизации (raise вместо assert — assert отключается -O)
        if len(ORGAN_NAMES) != len(self.organ_list):
            raise RuntimeError(
                f"WholeBodyModel: organ_list ({len(self.organ_list)}) и "
                f"ORGAN_NAMES ({len(ORGAN_NAMES)}) рассинхронизированы."
            )

        self.state_slices = []
        self.idx = {}
        start = 0
        for name, org in zip(ORGAN_NAMES, self.organ_list):
            size = org.get_state_size()
            if size < 0:
                raise RuntimeError(
                    f"WholeBodyModel: {name}.get_state_size()={size} < 0."
                )
            slc = slice(start, start + size)
            self.state_slices.append(slc)
            self.idx[name] = slc
            start += size
        self.total_states = start

        if self.total_states <= 0:
            raise RuntimeError(
                f"WholeBodyModel: total_states={self.total_states} ≤ 0."
            )

        # --- Кэш _compute_organ_flows ---
        self._flow_cache_t = None
        self._flow_cache_y = None
        self._flow_cache_result = None
        self._flow_cache_hits = 0
        self._flow_cache_misses = 0

        self._negative_vol_warned = False
        self._low_vol_warned = False


    @classmethod
    def auto_calibrate_R_sys_peripheral(cls,
                                        target_MAP: float,
                                        target_CO: float) -> float:
        """
        Авто-калибровка R_sys_peripheral из target_MAP и target_CO.

        Оценки сопротивлений органов заданы как константы — их замена
        требует калибровки, а не конфигурации. Если нужен полный
        контроль, задайте systemic.R_sys_peripheral в YAML явно
        (null → этот метод).
        """
        _R_RENAL_EST   = 4.5
        _R_BRAIN_EST   = 7.0
        _R_HA_EST      = 17.0
        _R_GITRACT_EST = 4.5

        R_total_target = target_MAP / target_CO
        sum_cond_other = (1/_R_RENAL_EST + 1/_R_BRAIN_EST
                          + 1/_R_HA_EST + 1/_R_GITRACT_EST)
        cond_per_needed = max(1/R_total_target - sum_cond_other, 0.1)
        R = 1.0 / cond_per_needed

        if not (cls._R_SYS_PERIPH_MIN <= R <= cls._R_SYS_PERIPH_MAX):
            raise ValueError(
                f"WholeBodyModel.auto_calibrate_R_sys_peripheral: "
                f"R={R:.3f} вне [{cls._R_SYS_PERIPH_MIN}, "
                f"{cls._R_SYS_PERIPH_MAX}]. Проверьте target_MAP="
                f"{target_MAP}, target_CO={target_CO}."
            )
        return R


    # ------------------------------------------------------------------
    # Калибровка начального состояния
    # ------------------------------------------------------------------
    def calibrate_initial_state(self, t_calib=600.0, t_eval=None,
                                rtol=1e-4, atol=1e-5,
                                p_sa_lo=50.0, p_sa_hi=150.0,
                                rel_tol_cycle=0.25):
        """
        Калибровка начального состояния. Валидирует свои входы.
        Возвращает либо y_steady, либо аналитическое y0.
        """
        # --- Валидация параметров калибровки ---
        t_calib = _check_range(
            "calibrate.t_calib", t_calib, 1.0, 1e5,
            "с, типично 400–600."
        )
        rtol = _check_range("calibrate.rtol", rtol, 1e-12, 1e-2, "типично 1e-4.")
        atol = _check_range("calibrate.atol", atol, 1e-15, 1e-2, "типично 1e-5.")
        p_sa_lo = _check_range(
            "calibrate.p_sa_lo", p_sa_lo, 0.0, 300.0, "мм рт.ст., типично 50."
        )
        p_sa_hi = _check_range(
            "calibrate.p_sa_hi", p_sa_hi, 0.0, 300.0, "мм рт.ст., типично 150."
        )
        if not (p_sa_lo < p_sa_hi):
            raise ValueError(
                f"WholeBodyModel.calibrate: p_sa_lo={p_sa_lo} должно быть < "
                f"p_sa_hi={p_sa_hi}."
            )
        rel_tol_cycle = _check_range(
            "calibrate.rel_tol_cycle", rel_tol_cycle, 1e-6, 10.0,
            "доля, типично 0.25."
        )

        y0 = self.get_initial_state()
        heart_slc = self.idx['heart']
        V0_arr = np.array([self.heart.V0[c] for c in ('LA', 'LV', 'RA', 'RV')])
        # heart-срез теперь длины 5: [V_la, V_lv, V_ra, V_rv, phi].
        # Маскируем только первые 4 объёмных слота; phi (phi0) не трогаем.
        vol_slc = slice(heart_slc.start, heart_slc.start + 4)
        y0[vol_slc] = np.maximum(y0[vol_slc], 1.5 * V0_arr)

        # Санитизация t_eval
        if t_eval is not None:
            t_eval = np.asarray(t_eval, dtype=float)
            t_eval = t_eval[(t_eval > 0.0) & (t_eval < t_calib)]
            if t_eval.size == 0:
                t_eval = None

        try:
            sol = solve_ivp(
                self.derivatives, (0.0, t_calib), y0,
                t_eval=t_eval, method=self.method,
                rtol=rtol, atol=atol, max_step=0.05,
            )
        except Exception as e:
            warnings.warn(f"calibrate: solver failed ({e}); using analytic y0")
            return y0

        if sol.y.shape[1] < 2 or not np.all(np.isfinite(sol.y[:, -1])):
            warnings.warn("calibrate: non-finite solution; using analytic y0")
            return y0

        y_steady = sol.y[:, -1].copy()

        P_sa_end = y_steady[self.idx['sys_art']][0]
        if not (p_sa_lo < P_sa_end < p_sa_hi):
            warnings.warn(
                f"calibrate: P_sa={P_sa_end:.1f} вне "
                f"[{p_sa_lo},{p_sa_hi}]; using analytic y0"
            )
            return y0

        C_a_O2_end = float(y_steady[self.idx['gas_exchange']][0])
        if not (0.05 < C_a_O2_end < 0.25):
            warnings.warn(
                f"calibrate: C_a_O2={C_a_O2_end:.3f} вне (0.05, 0.25); "
                f"using analytic y0"
            )
            return y0
        # Проверка: C_a_end должен совпасть с равновесием при конечных потоках
        out_end = self.compute_outputs(sol.t[-1], y_steady)
        C_eq = self.gas_exchange._equilibrium_state(
            C_v_O2=out_end['C_v_O2'],
            C_v_CO2=out_end['C_v_CO2'],
            Q_p=out_end['Q_pulmonary'],
            Q_shunt=out_end['Q_vsd'],
        )
        if abs(C_a_O2_end - C_eq[0]) > 0.01:
            warnings.warn(
                f"calibrate: C_a_O2={C_a_O2_end:.3f} ≠ равновесие "
                f"{C_eq[0]:.3f}; CSTR не сошёлся — using analytic y0"
            )
            return y0       

        if sol.t.size >= 3:
            P_sa_traj = sol.y[self.idx['sys_art'].start, :]
            br_out = self.baroreflex.get_outputs(y_steady[self.idx['baroreflex']])
            HR_end = float(br_out.get('HR', self.baroreflex.HR_base))
            T = 60.0 / max(HR_end, 1e-6)
            mask = sol.t >= (sol.t[-1] - T)
            if mask.sum() >= 3:
                ps_cycle = P_sa_traj[mask]
                mean_ps = np.mean(ps_cycle)
                if mean_ps > 0 and np.std(ps_cycle) / mean_ps > rel_tol_cycle:
                    warnings.warn(
                        f"calibrate: P_sa вариация за цикл "
                        f"{np.std(ps_cycle)/mean_ps:.2f} > {rel_tol_cycle}; "
                        f"using analytic y0"
                    )
                    return y0

        phi_idx = self.idx['heart'].start + 4
        y_steady[phi_idx] = y_steady[phi_idx] % 1.0

        return y_steady

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------
    def get_initial_state(self, calibrated=False):
        y0 = []
        for org in self.organ_list:
            y0.extend(org.get_initial_state())
        y0 = np.array(y0)
        if calibrated:
            return self.calibrate_initial_state()
        return y0

    def _effective_blood_volume(self, Vb: float) -> float:
        """
        Гладкий физиологический пол для объёма крови.

        При Vb >> V_floor возвращает Vb с экспоненциально малой поправкой.
        При Vb → 0 возвращает V_floor — конечное, физиологически осмысленное
        значение. Формула C∞-гладкая.

            soft_floor(V) = V_floor + w · log(1 + exp((V − V_floor)/w))
        """
        V0 = self.blood.V0
        V_floor = self._VB_FLOOR_FRAC * V0
        w       = self._VB_SOFT_WIDTH * V0
        x = (Vb - V_floor) / w
        x = float(np.clip(x, -self._VB_SOFT_CLIP, self._VB_SOFT_CLIP))
        return V_floor + w * float(np.log1p(np.exp(x)))

    def _compute_organ_flows(self, t, y):
        # --- Кэш ---
        if (self._flow_cache_t is not None
                and t == self._flow_cache_t
                and self._flow_cache_y is not None
                and self._flow_cache_y.shape == y.shape
                and np.array_equal(self._flow_cache_y, y)):
            self._flow_cache_hits += 1
            return self._flow_cache_result

        self._flow_cache_misses += 1
        sl = self.idx

        # --- Состояния ---
        V_heart      = y[sl['heart']]
        V_lungs      = y[sl['lungs']]
        V_liver      = y[sl['liver']]
        V_blood      = y[sl['blood']]
        V_gitract    = y[sl['gitract']]
        V_brain      = y[sl['brain']]
        V_periph     = y[sl['peripheral']]
        V_baroreflex = y[sl['baroreflex']]
        if not np.all(np.isfinite(V_baroreflex)):
            V_baroreflex = np.where(
                np.isfinite(V_baroreflex), V_baroreflex,
                self.baroreflex.get_initial_state(),
            )
        P_sa         = y[sl['sys_art']][0]
        V_sv         = y[sl['sys_ven']][0]
        P_pv         = y[sl['pul_ven']][0]
        V_jv_state   = y[sl['jugular_vein']]

        # --- Полный циркулирующий объём: сумма всех физических V ---
        # Все давления уже доступны из y (P-mode и pressure-states),
        # V_sv и V_jv — direct states. Ничего не зависит от order.
        # heart-состояние теперь 5: [V_la, V_lv, V_ra, V_rv, phi].
        # φ — безразмерная фаза (накопленные циклы), НЕ объём.
        _V_heart_full = y[sl['heart']]
        _V_heart_vol  = _V_heart_full[:4]      # только физические объёмы
        if np.any(_V_heart_vol < -1e-6) and not self._negative_vol_warned:
            warnings.warn(f"Negative heart volume: {_V_heart_vol}", RuntimeWarning)
            self._negative_vol_warned = True
        V_heart_phys = float(np.maximum(_V_heart_vol, 0.0).sum())

        _lungs_state = y[sl['lungs']]
        V_lungs_phys = (self.lungs.C1 * max(float(_lungs_state[0]), 0.0)
                    + self.lungs.C2 * max(float(_lungs_state[1]), 0.0))

        V_sv_phys = max(float(y[sl['sys_ven']][0]), 0.0)
        V_jv_phys = max(float(V_jv_state[0]), 0.0)
        V_sys_art_phys = self.sys_art.C * max(float(y[sl['sys_art']][0]), 0.0)
        V_pul_ven_phys = self.pul_ven.C * max(float(y[sl['pul_ven']][0]), 0.0)

        _liver_state = y[sl['liver']]
        V_liver_phys = (self.liver.C * max(float(_liver_state[0]), 0.0)
                    + self.liver.C_portal * max(float(_liver_state[5]), 0.0))

        _git_state = y[sl['gitract']]
        V_gitract_phys = (self.gitract.C_art * max(float(_git_state[0]), 0.0)
                        + self.gitract.C_cap * max(float(_git_state[1]), 0.0))

        V_brain_phys = self.brain.C * max(float(y[sl['brain']][0]), 0.0)

        # ВАЖНО: V_mix в GasExchange — ВИРТУАЛЬНЫЙ параметр (τ = V_mix/Q_s),
        # НЕ физический компартмент. НЕ добавлять его в V_blood_total.
        V_blood_total = (V_heart_phys + V_lungs_phys + V_sv_phys + V_jv_phys
                    + V_sys_art_phys + V_pul_ven_phys + V_liver_phys
                    + V_gitract_phys + V_brain_phys)

        P_sv = self.sys_ven.P0 + (V_sv - self.sys_ven.V0) / self.sys_ven.C
        P_sv = max(P_sv, 0.0)

        Vb           = V_blood_total
        C_blood      = V_blood
        conc         = dict(zip(self.substance_names, C_blood))
        P_pa         = V_lungs[0]

        # --- Барорефлекс ---
        R_remodel_now = float(V_lungs[2])
        baroreflex_inputs = {
            'P_sa': P_sa,
            'P_pa': P_pa,
            'R_remodel': R_remodel_now,
        }
        d_baroreflex = self.baroreflex.get_derivatives(
            t, V_baroreflex, baroreflex_inputs
        )
        baroreflex_out = self.baroreflex.get_outputs(V_baroreflex)
        HR = float(baroreflex_out['HR'])
        hr_factor = float(baroreflex_out.get(
            'hr_factor', HR / self.baroreflex.HR_base
        ))
        R_remodel_max = float(self.lungs.R_remodel_max)
        rv_afterload = float(np.clip(
            (R_remodel_now - 1.0) / max(R_remodel_max - 1.0, 1e-6),
            0.0, 5.0,
        ))

        # --- Сердце ---
        heart_inputs = {
            'P_sa': P_sa, 'P_sv': P_sv, 'P_pa': P_pa, 'P_pv': P_pv,
            'hr_factor': hr_factor,
            'baro_activation': baroreflex_out['baro_activation'],
            'baro_activation_rv': baroreflex_out.get('baro_activation_rv',
                                                    baroreflex_out['baro_activation']),
            'rv_afterload': rv_afterload,
            'V_sv':  V_sv,
            'V0_sv': self.sys_ven.V0,
        }
        d_heart = self.heart.get_derivatives(t, V_heart, heart_inputs)
        heart_out = self.heart.get_outputs(V_heart)

        # --- Лёгкие ---
        lungs_inputs = {'Q_pulmonary': heart_out['Q_pulmonary'], 'P_pv': P_pv}
        d_lungs = self.lungs.get_derivatives(t, V_lungs, lungs_inputs)
        lungs_out = self.lungs.get_outputs(V_lungs)

        # --- ЖКТ ---
        P_portal_state = V_liver[5] if len(V_liver) > 5 else 8.0
        gitract_inputs = {
            'P_sa': P_sa, 'P_sv': P_sv,
            'P_portal': P_portal_state,
            'intake_water': 0.0, 'intake_nutrients': 0.0,
        }
        d_gitract = self.gitract.get_derivatives(t, V_gitract, gitract_inputs)
        gitract_out = self.gitract.get_outputs(V_gitract)

        # --- Печень ---
        liver_inputs = {
            'P_sa': P_sa, 'P_sv': P_sv,
            'C_bilirubin_blood': conc.get('bilirubin', 0.0),
            'C_ammonia_blood':   conc.get('ammonia',   0.0),
            'C_albumin_blood':   conc.get('albumin',   0.0),
            'C_lactate_blood':   conc.get('lactate',   0.10),
            'V_blood': Vb,
            'Q_gut_out': gitract_out['Q_out'],
        }
        d_liver = self.liver.get_derivatives(t, V_liver, liver_inputs)
        liver_out = self.liver.get_outputs(V_liver)

        # --- Почки ---
        kidney_effects = self.kidney.compute_effects(
            P_sa, P_sv, conc.get('tox', 0.0), Vb,
        )

        # --- Газообмен (CSTR) ---
        V_gas_ex = y[sl['gas_exchange']]
        C_v_O2_bulk  = conc.get('oxygen', 0.15)
        C_v_CO2_bulk = conc.get('co2',   0.52)
        Q_p_now      = heart_out['Q_pulmonary']
        Q_shunt_now  = heart_out['Q_vsd']

        gas_inputs = {
            'C_v_O2':  C_v_O2_bulk,
            'C_v_CO2': C_v_CO2_bulk,
            'Q_p':     Q_p_now,
            'Q_shunt': Q_shunt_now,
        }
        d_gas_ex = self.gas_exchange.get_derivatives(t, V_gas_ex, gas_inputs)
        gas_ex   = self.gas_exchange.compute_effects(
            state=V_gas_ex,
            C_v_O2=C_v_O2_bulk,
            C_v_CO2=C_v_CO2_bulk,
            Q_p=Q_p_now,
            Q_shunt=Q_shunt_now,
        )

        # --- Яремная вена: извлечение P_jv ---
        # V_jv_phys уже посчитан выше (в блоке V_blood_total)
        P_jv = self.jugular_vein.P0 + (V_jv_phys - self.jugular_vein.V0) / self.jugular_vein.C
        P_jv = max(P_jv, 0.0)

        # --- Мозг ---
        brain_inputs = {
            'P_sa':            P_sa,
            'P_sv':            P_jv,
            'C_a_O2':          gas_ex['C_a_O2'],
            'C_a_CO2':         gas_ex['C_a_CO2'],
            'C_lactate_blood': conc.get('lactate', 0.10),
            'C_ammonia':       conc.get('ammonia', 0.0),
            'V_blood':         Vb,
            'occlusion_factor': self._occlusion_factor,
        }
        d_brain = self.brain.get_derivatives(t, V_brain, brain_inputs)
        brain_out = self.brain.get_outputs(V_brain)

        # --- Периферия ---
        peripheral_inputs = {
            'P_sa': P_sa, 'P_sv': P_sv,
            'C_a_O2': gas_ex['C_a_O2'],
            'C_v_lactate': conc.get('lactate', 0.10),
            'V_blood': Vb,
            'baro_sys_scale': baroreflex_out.get('R_sys_scale', 1.0),
        }
        d_peripheral = self.peripheral.get_derivatives(t, V_periph, peripheral_inputs)
        periph_out = self.peripheral.get_outputs(V_periph)

        # --- Яремная вена ---
        jugular_inputs = {
            'Q_in':     brain_out['Q_out'],
            'C_in_O2':  brain_out['C_v_O2_brain'],
            'C_in_CO2': brain_out['C_v_CO2_brain'],
            'P_sv':     P_sv,
        }
        d_jugular = self.jugular_vein.get_derivatives(t, V_jv_state, jugular_inputs)
        jugular_out = self.jugular_vein.get_outputs(V_jv_state)

        # --- Централизованный баланс O2/CO2 ---
        Q_s       = heart_out['Q_aortic']
        Q_br      = brain_out['Q_br']
        Q_other   = Q_s - Q_br
        C_a_O2    = gas_ex['C_a_O2']
        C_a_CO2   = gas_ex['C_a_CO2']
        C_bulk_O2 = conc.get('oxygen', 0.15)
        C_bulk_CO2= conc.get('co2', 0.52)
        C_jv_O2   = jugular_out['C_jv_O2']
        C_jv_CO2  = jugular_out['C_jv_CO2']
        VO2_brain  = brain_out['VO2_brain']
        VO2_periph = periph_out['O2_consumption_periph']
        # VO2_other сохраняется ТОЛЬКО для диагностики VO2_total.
        # В сам баланс O2 крови он больше не входит (см. ниже).
        VO2_other  = VO2_periph + self.VO2_rest
        VO2_total  = VO2_brain + VO2_other
        VCO2_brain  = brain_out['CO2_production']
        VCO2_periph = periph_out.get(
            'VCO2_production',
            periph_out['O2_consumption_periph'] * self.RQ
        )
        VCO2_total  = VCO2_brain + VCO2_periph + self.VO2_rest * self.RQ

        # Гладкий физиологический пол вместо max(Vb, 1e-6).
        # См. _effective_blood_volume — soft-plus с шириной 0.1·V0.
        Vb_safe = self._effective_blood_volume(Vb)

        # Однократное предупреждение при выходе за пределы применимости
        if Vb < self._VB_FLOOR_FRAC * self.blood.V0 and not self._low_vol_warned:
            warnings.warn(
                f"V_blood={Vb:.1f} mL ниже физиологического пола "
                f"{self._VB_FLOOR_FRAC * self.blood.V0:.0f} mL — модель вне "
                f"области применимости. Концентрации считаются по Vb_safe="
                f"{Vb_safe:.0f} mL.",
                RuntimeWarning,
            )
            self._low_vol_warned = True

        # --- Баланс O2 крови: двухкомпартментное сопряжение ---
        # Разделяем Q_other на периферию (имеет собственный C_O2_loc)
        # и остальные органы (печень, почки, ЖКТ, «покой» — без
        # тканевого O2-состояния, для них well-mixed приближение).
        #
        # Для периферии потребление O2 уже выражено через разницу
        # (C_v_O2_periph − C_bulk_O2) в её тканевом ОДУ. Отдельный
        # член «− VO2_periph» был бы двойным счётом и разрывал бы
        # масс-баланс в переходных процессах (см. O2_1.md §1.2).
        Q_periph_flow   = float(periph_out['Q_peripheral'])
        Q_other_non_per = max(Q_other - Q_periph_flow, 0.0)
        # Fallback на C_bulk_O2 — обратная совместимость со старыми
        # сборками peripheral_tissues без ключа 'C_v_O2_periph'.
        C_v_O2_periph   = float(periph_out.get('C_v_O2_periph', C_bulk_O2))

        dC_O2_blood = (
            Q_other_non_per * (C_a_O2 - C_bulk_O2)
            + Q_periph_flow * (C_v_O2_periph - C_bulk_O2)
            + Q_br          * (C_jv_O2 - C_bulk_O2)
            - self.VO2_rest
        ) / Vb_safe

        dC_CO2_blood = (
            Q_s * (C_a_CO2 - C_bulk_CO2)
            + VCO2_total
        ) / Vb_safe

        # --- Локальные потоки ---
        Q_peripheral = periph_out['Q_peripheral']
        Q_ha         = liver_out.get('Q_ha', 0.0)
        Q_renal      = kidney_effects['Q_renal']
        Q_gitract_in = gitract_out['Q_in']
        Q_brain      = brain_out['Q_br']
        Q_jv_out     = jugular_out.get('Q_jv_out', Q_brain)
        Q_art_out    = Q_peripheral + Q_ha + Q_renal + Q_gitract_in + Q_brain
        Q_ven_in     = Q_peripheral + liver_out['Q_liver_out'] + Q_renal + Q_jv_out
        Q_ven_out    = heart_out['Q_sv_to_ra']

        # --- Накопление dC_blood_arr ---
        dC_blood_arr = np.zeros(len(self.substance_names))
        idx = self._substance_idx
        if 'bilirubin' in idx: dC_blood_arr[idx['bilirubin']] += liver_out.get('dC_bilirubin', 0.0)
        if 'ammonia'   in idx: dC_blood_arr[idx['ammonia']]   += liver_out.get('dC_ammonia',   0.0)
        if 'albumin'   in idx: dC_blood_arr[idx['albumin']]   += liver_out.get('dC_albumin',   0.0)
        if 'lactate'   in idx: dC_blood_arr[idx['lactate']]   += liver_out.get('dC_lactate',   0.0)
        if 'tox'       in idx: dC_blood_arr[idx['tox']]       += kidney_effects['dC_tox']
        if 'lactate'   in idx: dC_blood_arr[idx['lactate']]   += periph_out['dC_lactate_blood']
        if 'lactate'   in idx: dC_blood_arr[idx['lactate']]   += brain_out.get('dC_lactate_blood', 0.0)
        if 'ammonia'   in idx: dC_blood_arr[idx['ammonia']]   += brain_out.get('dC_ammonia_blood', 0.0)
        if 'oxygen' in idx: dC_blood_arr[idx['oxygen']] += dC_O2_blood
        if 'co2'    in idx: dC_blood_arr[idx['co2']]    += dC_CO2_blood

        # --- Баланс жидкости ---
        dV_total = (
            gitract_out['absorption_water']
            + self.fluid_intake_rate
            - kidney_effects['urine_output']
            - self.insensible_loss_rate
        )

        result = {
            'V_heart': V_heart, 'V_lungs': V_lungs, 'V_liver': V_liver,
            'V_blood': V_blood, 'V_gitract': V_gitract, 'V_brain': V_brain,
            'V_periph': V_periph, 'V_baroreflex': V_baroreflex, 'V_jugular': V_jv_state,
            'P_sa': P_sa, 'P_sv': P_sv, 'P_pv': P_pv, 'P_pa': P_pa,
            'V_sv': V_sv, 'Vb': Vb, 'C_blood': C_blood, 'conc': conc,
            'V_sv_fraction': V_sv / max(Vb, 1e-6) if Vb > 0 else 0.0,
            'V_jv': V_jv_phys, 'C_jv_O2': V_jv_state[1], 'C_jv_CO2': V_jv_state[2],
            'HR': HR, 'hr_factor': hr_factor, 'baroreflex_out': baroreflex_out,
            'baro_activation': baroreflex_out['baro_activation'],
            'd_heart': d_heart, 'd_lungs': d_lungs, 'd_liver': d_liver,
            'd_gitract': d_gitract, 'd_brain': d_brain,
            'd_peripheral': d_peripheral, 'd_baroreflex': d_baroreflex, 'd_jugular': d_jugular,
            'heart_out': heart_out, 'lungs_out': lungs_out,
            'gitract_out': gitract_out, 'liver_out': liver_out,
            'kidney_effects': kidney_effects, 'brain_out': brain_out,
            'gas_ex': gas_ex, 'periph_out': periph_out, 'jugular_out': jugular_out,
            'Q_peripheral': Q_peripheral, 'Q_ha': Q_ha, 'Q_renal': Q_renal,
            'Q_gitract_in': Q_gitract_in, 'Q_brain': Q_brain, 'Q_jv_out': Q_jv_out,
            'Q_art_out': Q_art_out, 'Q_ven_in': Q_ven_in, 'Q_ven_out': Q_ven_out,
            'dC_blood_arr': dC_blood_arr, 'dV_total': dV_total,
            'VO2_total':    float(VO2_total),
            'VO2_brain':    float(VO2_brain),
            'VO2_periph':   float(VO2_periph),
            'VO2_rest':     float(self.VO2_rest),
            'VCO2_total':   float(VCO2_total),
            'dC_O2_blood':  float(dC_O2_blood),
            'dC_CO2_blood': float(dC_CO2_blood),
            'P_jv':         float(P_jv),
            'dV_external': dV_total,
            'd_gas_ex': d_gas_ex,
            'V_gas_ex': V_gas_ex,
        }
        self._flow_cache_t = t
        self._flow_cache_y = y.copy()
        self._flow_cache_result = result
        return result

    # ------------------------------------------------------------------
    # derivatives
    # ------------------------------------------------------------------
    def derivatives(self, t, y):
        f = self._compute_organ_flows(t, y)

        # Кровь: state = только концентрации, V/dV — извне
        blood_inputs = {
            'V_blood':  f['Vb'],          # derived total
            'dV_blood': f['dV_total'],    # = absorption + intake − urine − insensible
            'dC':       f['dC_blood_arr'],
        }
        d_blood = self.blood.get_derivatives(t, f['C_blood'], blood_inputs)

        # Системные артерии
        d_sys_art = self.sys_art.get_derivatives(
            t, np.array([f['P_sa']]),
            {'Q_in': f['heart_out']['Q_aortic'], 'Q_out': f['Q_art_out']},
        )

        # Системные вены: получают dV_external как сток
        d_sys_ven = self.sys_ven.get_derivatives(
            t, np.array([f['V_sv']]),
            {
                'Q_in':        f['Q_ven_in'],
                'Q_out':       f['Q_ven_out'],
                'dV_external': f['dV_total'],   # ← новый вход
            },
        )

        # Лёгочные вены
        Q_from_lungs = f['lungs_out']['Q_out']
        Q_pul_ven_out = f['heart_out']['Q_pv_to_la']
        d_pul_ven = self.pul_ven.get_derivatives(
            t, np.array([f['P_pv']]),
            {'Q_in': Q_from_lungs, 'Q_out': Q_pul_ven_out},
        )

        d_jugular_vein = f['d_jugular']

        return np.concatenate([
            f['d_heart'], f['d_lungs'], f['d_liver'], d_blood,
            f['d_gitract'], f['d_brain'], f['d_peripheral'], f['d_baroreflex'],
            d_sys_art, d_sys_ven, d_pul_ven, d_jugular_vein, f['d_gas_ex'],
        ])

    # ------------------------------------------------------------------
    # compute_outputs
    # ------------------------------------------------------------------
    def compute_outputs(self, t, y):
        f = self._compute_organ_flows(t, y)

        heart_out = f['heart_out']
        Qp = heart_out['Q_pulmonary']
        Qs = heart_out['Q_aortic']
        shunt_fraction_LR = max(heart_out['Q_vsd'], 0.0) / max(Qp, 1e-6)

        return {
            'P_sa': f['P_sa'], 'P_sv': f['P_sv'], 'P_pa': f['P_pa'], 'P_pv': f['P_pv'],
            'V_la': f['V_heart'][0], 'V_lv': f['V_heart'][1],
            'V_sv': f['V_sv'],
            'V_sv_fraction': f['V_sv_fraction'],
            'V_ra': f['V_heart'][2], 'V_rv': f['V_heart'][3],
            'Q_aortic': Qs, 'Q_pulmonary': Qp,
            'Q_vsd': heart_out['Q_vsd'],
            'P_lv': heart_out['P_lv'], 'P_rv': heart_out['P_rv'],
            'P_la': heart_out['P_la'], 'P_ra': heart_out['P_ra'],
            'Q_ven_in':   f['Q_ven_in'],
            'Q_ven_out':  f['Q_ven_out'],
            'Q_sv_to_ra': heart_out['Q_sv_to_ra'],
            'Q_pv_to_la': heart_out['Q_pv_to_la'],
            'shunt_fraction_LR': shunt_fraction_LR,
            'shunt_fraction_R2L': f['gas_ex']['shunt_fraction_R2L'],
            'Q_liver_out': f['liver_out']['Q_liver_out'],
            'Q_renal': f['kidney_effects']['Q_renal'],
            'urine_output': f['kidney_effects']['urine_output'],
            'Q_gitract_out': f['gitract_out']['Q_out'],
            'absorption_water': f['gitract_out']['absorption_water'],
            'Q_brain': f['brain_out']['Q_br'],
            'V_blood': f['Vb'],
            'GFR': f['kidney_effects']['GFR'],
            'C_bilirubin_blood': f['conc'].get('bilirubin', 0),
            'C_ammonia_blood':   f['conc'].get('ammonia',   0),
            'C_albumin_blood':   f['conc'].get('albumin',   0),
            'C_tox_blood':       f['conc'].get('tox',       0),
            'oxygenation_index': f['gas_ex']['SaO2'],
            'metabolic_inhibition': f['brain_out'].get('metabolic_inhibition', 1.0),
            'liver_functional':     f['liver_out'].get('functional', 1.0),
            'R1_lungs': f['lungs_out'].get('R1_eff', self.lungs.R1_base),
            'R2_lungs': f['lungs_out'].get('R2_eff', self.lungs.R2_base),
            'R_remodel':     f['lungs_out'].get('R_remodel',     1.0),
            'f_recruit':     f['lungs_out'].get('recruit_factor', 1.0),
            'f_recruit_eff': f['lungs_out'].get('f_recruit_eff', 1.0),
            'R_target':      f['lungs_out'].get('R_target',      1.0),
            'dR_remodel':    f['lungs_out'].get('dR_remodel',    0.0),
            'mode_remodeled': 1.0 if f['lungs_out'].get('mode') == 'remodeled' else 0.0,
            'HR': f['HR'],
            'HR_target': f['baroreflex_out']['HR_target'],
            'baro_activation': f['baroreflex_out']['baro_activation'],
            'baro_activation_rv': f['baroreflex_out'].get(
                'baro_activation_rv', f['baroreflex_out']['baro_activation']
            ),
            'baro_inotropy':         f['baroreflex_out'].get('baro_activation', 1.0),
            'baro_vasomotor':        f['baroreflex_out'].get('R_sys_scale', 1.0),
            'baro_inotropy_target':  f['baroreflex_out'].get('baro_inotropy_target', 1.0),
            'baro_vasomotor_target': f['baroreflex_out'].get('baro_vasomotor_target', 1.0),
            'suppress':              f['baroreflex_out'].get('suppress', 1.0),
            'SaO2':   f['gas_ex']['SaO2'],
            'C_a_O2': f['gas_ex']['C_a_O2'],
            'C_v_O2': f['gas_ex']['C_v_O2'],
            'P_a_O2': f['gas_ex']['P_a_O2'],
            'P_v_O2': f['gas_ex']['P_v_O2'],
            'P_v_CO2': f['gas_ex']['P_v_CO2'],
            'C_a_CO2': f['gas_ex']['C_a_CO2'],
            'C_v_CO2': f['gas_ex']['C_v_CO2'],
            'O2_uptake':   f['gas_ex']['O2_uptake'],
            'CO2_removal': f['gas_ex']['CO2_removal'],
            'Q_peripheral': f['Q_peripheral'],
            'Q_art_out':    f['Q_art_out'],
            'R_sys_peripheral': self.R_sys_peripheral,
            'R_eff_peripheral':      f['periph_out'].get('R_eff', None),
            'C_O2_local_periph':     f['periph_out'].get('C_O2_local', None),
            'C_lactate_local_periph': f['periph_out'].get('C_lactate_local', None),
            'C_lactate_blood':      f['conc'].get('lactate', None),
            'dC_lactate_liver':     f['liver_out'].get('dC_lactate', 0.0),
            'lactate_production':   f['periph_out'].get('lactate_production', 0.0),
            'O2_consumption_periph': f['periph_out'].get('O2_consumption_periph', 0.0),
            'f_O2_autoreg':         f['periph_out'].get('f_O2_autoreg', 1.0),
            'f_P_myogenic':         f['periph_out'].get('f_P_myogenic', 1.0),
            'dC_lactate_periph_to_blood': f['periph_out'].get('dC_lactate_blood', 0.0),
            'Q_mitral':    heart_out['Q_mitral'],
            'Q_tricuspid': heart_out['Q_tricuspid'],
            'V_jv': f['V_jv'],
            'P_jv': f['jugular_out']['P_jv'],
            'Q_jv_out': f['Q_jv_out'],
            'C_jv_O2': f['jugular_out']['C_jv_O2'],
            'C_jv_CO2': f['jugular_out']['C_jv_CO2'],
            'SjvO2': f['jugular_out']['SjvO2'],
            'P_jv_O2': f['jugular_out']['P_jv_O2'],
            'C_v_O2_brain': f['brain_out'].get('C_v_O2_brain', f['brain_out'].get('C_v_O2')),
            'C_v_O2_periph': f['periph_out'].get('C_v_O2_periph'),
            'C_v_CO2_brain': f['brain_out'].get('C_v_CO2_brain'),
            'C_v_lactate_brain': f['brain_out'].get('C_v_lactate_brain'),
            'C_v_ammonia_brain': f['brain_out'].get('C_v_ammonia_brain'),
            'f_CO2_autoreg_brain': f['brain_out'].get('f_CO2_autoreg', 1.0),
            'C_O2_tissue_brain': f['brain_out'].get('C_O2_tissue'),
            'C_CO2_tissue_brain': f['brain_out'].get('C_CO2_tissue'),
            'C_lactate_tissue_brain': f['brain_out'].get('C_lactate_tissue'),
            'C_ammonia_tissue_brain': f['brain_out'].get('C_ammonia_tissue'),
            'inhib_O2_brain': f['brain_out'].get('inhib_O2'),
            'inhib_amm_brain': f['brain_out'].get('inhib_amm'),
            'lactate_production_brain': f['brain_out'].get('lactate_production'),
            'VO2_total':       f['VO2_total'],
            'VO2_brain':       f['VO2_brain'],
            'VO2_periph':      f['VO2_periph'],
            'VO2_rest':        f['VO2_rest'],
            'VCO2_total':      f['VCO2_total'],
            'dC_CO2_blood':    f['dC_CO2_blood'],
            'dC_O2_blood':     f['dC_O2_blood'],
            'occlusion_factor': f['brain_out'].get('occlusion_factor', 1.0),
            'E_max_lv': heart_out.get('E_max_lv'),
            'E_max_rv': heart_out.get('E_max_rv'),
            'E_max_la': heart_out.get('E_max_la'),
            'E_max_ra': heart_out.get('E_max_ra'),
            'HR_current': heart_out.get('HR_current'),
            'T_current': heart_out.get('T_current'),
            # текущая фаза ∈ [0, 1)
            'phi_heart':     heart_out.get('phi'),
            # накопленное число циклов
            'phi_raw_heart': heart_out.get('phi_raw'),
            'baro_scale_periph': f['periph_out'].get('baro_scale_applied', 1.0),
            'R_target_periph':   f['periph_out'].get('R_target', None),
        }

    def cycle_averaged_flows(self, t_end, y_end, n_pts=60):
        """
        Усредняет Qp/Qs/Q_vsd за один РЕАЛЬНЫЙ кардиоцикл, интегрируя
        ODE от y_end вперёд на T.

        Здесь же объёмы камер, P_sa, P_pv, R_remodel эволюционируют
        через реальный RHS, и только после этого потоки усредняются.
        """
        # --- HR и период из конечного состояния ---
        _out = self.compute_outputs(t_end, y_end)
        HR = float(_out['HR'])
        T = 60.0 / max(HR, 1.0)

        # --- Равномерная сетка по одному циклу [t_end, t_end + T) ---
        # endpoint=False: t_end+T — это уже следующая фаза;
        # для среднего по [0, T) он не нужен, иначе получим дублирование.
        n_pts = max(int(n_pts), 20)
        t_eval = np.linspace(t_end, t_end + T, n_pts, endpoint=False)

        # --- Реальная интеграция ODE вперёд ---
        try:
            sol_cycle = solve_ivp(
                self.derivatives, (t_end, t_end + T), y_end,
                t_eval=t_eval,
                method=self.method,
                rtol=1e-4, atol=1e-5,
                max_step= 0.05 # T / max(n_pts, 40),   # достаточно точек на клапанные пики
            )
        except Exception as e:
            warnings.warn(f"cycle_averaged_flows: solver failed ({e}); "
                        f"returning NaN")
            nan = float('nan')
            return {
                'Qp_cycle_mean': nan, 'Qs_cycle_mean': nan,
                'Q_vsd_cycle_mean': nan, 'Qp_Qs_cycle': nan,
                'mass_balance_error': nan, 'SaO2_cycle_mean': nan,
                'shunt_fraction_R2L_mean': nan,
            }

        if sol_cycle.t.size < 2 or not np.any(np.isfinite(sol_cycle.y)):
            warnings.warn("cycle_averaged_flows: non-finite cycle solution; "
                        "returning NaN")
            nan = float('nan')
            return {
                'Qp_cycle_mean': nan, 'Qs_cycle_mean': nan,
                'Q_vsd_cycle_mean': nan, 'Qp_Qs_cycle': nan,
                'mass_balance_error': nan, 'SaO2_cycle_mean': nan,
                'shunt_fraction_R2L_mean': nan,
            }

        # Сбор только потоков — они пульсируют, но не сглажены состояниями
        Qp, Qs, Qv = [], [], []
        for i in range(sol_cycle.t.size):
            out = self.compute_outputs(float(sol_cycle.t[i]),
                                    sol_cycle.y[:, i])
            Qp.append(float(out['Q_pulmonary']))
            Qs.append(float(out['Q_aortic']))
            Qv.append(float(out['Q_vsd']))

        Qp_m = float(np.mean(Qp))
        Qs_m = float(np.mean(Qs))
        Qv_m = float(np.mean(Qv))

        # SaO2 — из состояния в конце цикла: CSTR уже интегральная величина
        out_end = self.compute_outputs(float(sol_cycle.t[-1]),
                                    sol_cycle.y[:, -1])
        SaO2_m   = float(out_end['SaO2'])
        C_a_O2_m = float(out_end['C_a_O2'])

        # Цикловое f_bypass
        Qrl_m = max(-Qv_m, 0.0)
        Qlr_m = max( Qv_m, 0.0)
        Qs_eff = max(Qs_m, 1e-6)
        f_bypass_cycle = Qrl_m / Qs_eff

        return {
            'Qp_cycle_mean':           Qp_m,
            'Qs_cycle_mean':           Qs_m,
            'Q_vsd_cycle_mean':        Qv_m,
            'Qp_Qs_cycle':             Qp_m / Qs_eff,
            'mass_balance_error':      Qp_m - Qs_m - Qv_m,
            'SaO2_cycle_mean':         SaO2_m,
            'C_a_O2_cycle':            C_a_O2_m,
            'f_bypass_cycle':          f_bypass_cycle,
            'shunt_fraction_R2L_mean': f_bypass_cycle,
        }

    # ------------------------------------------------------------------
    # simulate / set_occlusion
    # ------------------------------------------------------------------
    def simulate(self, t_span, t_eval=None, y0=None, method=None, **kwargs):
        if y0 is None:
            y0 = self.calibrate_initial_state()
        if method is None:
            method = self.method
        kwargs.setdefault('max_step', 0.05)
        return solve_ivp(
            self.derivatives, t_span, y0,
            t_eval=t_eval, method=method, **kwargs,
        )

    def set_occlusion(self, factor: float) -> None:
        """0 — полная окклюзия, 1 — норма."""
        factor = float(np.clip(factor, 0.0, 1.0))
        self._occlusion_factor = factor
        # Сброс кэша, чтобы следующий шаг увидел новое значение
        self._flow_cache_t = None