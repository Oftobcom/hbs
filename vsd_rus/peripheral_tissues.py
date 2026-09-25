# peripheral_tissues.py
"""
Модель периферических тканей (скелетные мышцы, кожа, соединительная ткань).

Состояние: [C_O2_local, C_lactate_local, R_eff]
    C_O2_local      — локальная концентрация O2 (мл O2 / мл крови-экв.)
    C_lactate_local — локальная концентрация лактата (мг/мл)
    R_eff           — эффективное периферическое сопротивление
                      (мм рт.ст.·с/мл)

Единицы:
    P          — мм рт. ст.
    Q          — мл/с
    R          — мм рт.ст.·с/мл
    C_O2       — мл O2 / мл крови
    C_lactate  — мг/мл  (≈ 0.09 мг/мл = 1 мМ)
"""

import numpy as np
from organ_base import OrganModel


class PeripheralTissues(OrganModel):
    """
    Модель периферических тканей.
    Состояние: [C_O2_local, C_lactate_local, R_eff].
    """

    # --- Санити-пороги для валидации конфигурации ---
    _R_BASE_MIN, _R_BASE_MAX = 0.1, 50.0
    _C_TISSUE_MIN, _C_TISSUE_MAX = 0.1, 1000.0
    _P_TISSUE0_MIN, _P_TISSUE0_MAX = 0.0, 100.0

    _O2_NORM_MIN, _O2_NORM_MAX = 0.05, 0.25
    _K_O2_MIN, _K_O2_MAX = 0.1, 3.0
    _R_MIN_FACTOR_MIN, _R_MIN_FACTOR_MAX = 0.1, 0.9
    _TAU_AUTOREG_MIN, _TAU_AUTOREG_MAX = 0.1, 60.0

    _K_P_MYOGENIC_MIN, _K_P_MYOGENIC_MAX = 0.0, 0.05
    _P_SA_NORM_MIN, _P_SA_NORM_MAX = 50.0, 120.0
    _R_MAX_FACTOR_MIN, _R_MAX_FACTOR_MAX = 1.0, 5.0
    _DEADBAND_MIN, _DEADBAND_MAX = 0.0, 30.0

    _VO2_BASE_MIN, _VO2_BASE_MAX = 0.1, 10.0
    _VO2_BASAL_FRAC_MIN, _VO2_BASAL_FRAC_MAX = 0.0, 0.5
    _V_TISSUE_MIN, _V_TISSUE_MAX = 10.0, 5000.0
    _C_A_O2_NORM_MIN, _C_A_O2_NORM_MAX = 0.05, 0.25

    _LAC_THRESH_MIN, _LAC_THRESH_MAX = 0.0, 0.15
    _K_LAC_PROD_MIN, _K_LAC_PROD_MAX = 0.0, 1.0
    _K_LAC_CLEAR_MIN, _K_LAC_CLEAR_MAX = 0.0, 1.0
    _K_LAC_REL_MIN, _K_LAC_REL_MAX = 0.0, 1.0
    _C_LAC0_MIN, _C_LAC0_MAX = 0.0, 1.0
    _C_O2_LOC0_MIN, _C_O2_LOC0_MAX = 0.0, 0.25

    # ------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------
    def __init__(self,
        # --- Гемодинамика ---
        R_base: float = 3.8,
        C_tissue: float = 8.0,             # не используется, оставлено для расширения
        P_tissue0: float = 15.0,           # диагностика

        # --- Метаболическая ауторегуляция ---
        O2_norm: float = 0.15,
        k_O2_autoreg: float = 1.0,         # экспонента в f_O2 (1.0 = линейно)
        R_min_factor: float = 0.75,        # макс. вазодилатация (f_O2_min)
        tau_autoreg: float = 3.0,

        # --- Миогенная ауторегуляция ---
        k_P_myogenic: float = 0.002,
        P_sa_norm: float = 90.0,
        R_max_factor: float = 2.5,
        P_myogenic_deadband: float = 10.0,

        # --- Потребление O2 ---
        VO2_base: float = 1.5,
        VO2_basal_frac: float = 0.05,      # доля VO2_base при Q=0 (анаэробный резерв)
        V_tissue_eff: float = 400.0,
        C_a_O2_norm: float = 0.20,

        # --- Лактат ---
        C_O2_lactate_threshold: float = 0.08,
        k_lactate_prod: float = 0.05,
        k_lactate_clear: float = 0.02,     # согласовано с physiology.yaml
        k_lactate_release: float = 0.05,
        C_lactate0: float = 0.10,
        C_O2_local0: float = 0.10):

        # =================================================================
        # Валидация конфигурации — fail-fast при инициализации.
        # Параметры приходят из YAML и не меняются во время симуляции;
        # ошибки ловятся один раз, а не в горячем пути RHS.
        # =================================================================
        def _check_range(name, v, lo, hi, typical=""):
            v = float(v)
            if not np.isfinite(v) or not (lo <= v <= hi):
                raise ValueError(
                    f"PeripheralTissues: {name}={v} вне [{lo}, {hi}]. {typical}"
                )
            return v

        # --- Гемодинамика ---
        self.R_base = _check_range(
            "R_base", R_base, self._R_BASE_MIN, self._R_BASE_MAX,
            "типично 2–6 мм рт.ст.·с/мл (после калибровки под target_MAP/CO)."
        )
        self.C_tissue = _check_range(
            "C_tissue", C_tissue, self._C_TISSUE_MIN, self._C_TISSUE_MAX,
            "не используется в текущей модели; оставлено для расширения."
        )
        self.P_tissue0 = _check_range(
            "P_tissue0", P_tissue0, self._P_TISSUE0_MIN, self._P_TISSUE0_MAX,
            "диагностическое значение, типично 15 мм рт.ст."
        )

        # --- Метаболическая ауторегуляция ---
        self.O2_norm = _check_range(
            "O2_norm", O2_norm, self._O2_NORM_MIN, self._O2_NORM_MAX,
            "типично 0.15 мл O2/мл."
        )
        self.k_O2_autoreg = _check_range(
            "k_O2_autoreg", k_O2_autoreg, self._K_O2_MIN, self._K_O2_MAX,
            "типично 1.0 (линейно). k>1 усиливает дилатацию при глубокой гипоксии, "
            "k<1 делает реакцию более вялой."
        )
        self.R_min_factor = _check_range(
            "R_min_factor", R_min_factor,
            self._R_MIN_FACTOR_MIN, self._R_MIN_FACTOR_MAX,
            "типично 0.4–0.5 (дилатация 2–2.5×). 0.75 — консервативно (1.33×)."
        )
        self.tau_autoreg = _check_range(
            "tau_autoreg", tau_autoreg,
            self._TAU_AUTOREG_MIN, self._TAU_AUTOREG_MAX,
            "типично 3 с."
        )

        # --- Миогенная ауторегуляция ---
        self.k_P_myogenic = _check_range(
            "k_P_myogenic", k_P_myogenic,
            self._K_P_MYOGENIC_MIN, self._K_P_MYOGENIC_MAX,
            "типично 0.002–0.005."
        )
        self.P_sa_norm = _check_range(
            "P_sa_norm", P_sa_norm, self._P_SA_NORM_MIN, self._P_SA_NORM_MAX,
            "типично 90 мм рт.ст."
        )
        self.R_max_factor = _check_range(
            "R_max_factor", R_max_factor,
            self._R_MAX_FACTOR_MIN, self._R_MAX_FACTOR_MAX,
            "типично 2.5 (макс. вазоконстрикция)."
        )
        self.P_myogenic_deadband = _check_range(
            "P_myogenic_deadband", P_myogenic_deadband,
            self._DEADBAND_MIN, self._DEADBAND_MAX,
            "типично 10 мм рт.ст."
        )

        # --- Потребление O2 ---
        self.VO2_base = _check_range(
            "VO2_base", VO2_base, self._VO2_BASE_MIN, self._VO2_BASE_MAX,
            "типично 1.5 мл/с (мышцы+кожа, ~90 мл/мин)."
        )
        self.VO2_basal_frac = _check_range(
            "VO2_basal_frac", VO2_basal_frac,
            self._VO2_BASAL_FRAC_MIN, self._VO2_BASAL_FRAC_MAX,
            "типично 0.05 — анаэробный резерв при Q→0."
        )
        self.V_tissue_eff = _check_range(
            "V_tissue_eff", V_tissue_eff,
            self._V_TISSUE_MIN, self._V_TISSUE_MAX,
            "типично 400 мл."
        )
        self.C_a_O2_norm = _check_range(
            "C_a_O2_norm", C_a_O2_norm,
            self._C_A_O2_NORM_MIN, self._C_A_O2_NORM_MAX,
            "типично 0.20 мл O2/мл."
        )

        # --- Лактат ---
        self.C_O2_lactate_threshold = _check_range(
            "C_O2_lactate_threshold", C_O2_lactate_threshold,
            self._LAC_THRESH_MIN, self._LAC_THRESH_MAX,
            "типично 0.08 мл O2/мл — порог начала анаэробного гликолиза."
        )
        self.k_lactate_prod = _check_range(
            "k_lactate_prod", k_lactate_prod,
            self._K_LAC_PROD_MIN, self._K_LAC_PROD_MAX,
            "типично 0.05 мг/(мл·с)."
        )
        self.k_lactate_clear = _check_range(
            "k_lactate_clear", k_lactate_clear,
            self._K_LAC_CLEAR_MIN, self._K_LAC_CLEAR_MAX,
            "типично 0.02 1/с."
        )
        self.k_lactate_release = _check_range(
            "k_lactate_release", k_lactate_release,
            self._K_LAC_REL_MIN, self._K_LAC_REL_MAX,
            "типично 0.05 1/с."
        )
        self.C_lactate0 = _check_range(
            "C_lactate0", C_lactate0,
            self._C_LAC0_MIN, self._C_LAC0_MAX,
            "типично 0.10 мг/мл (=1 мМ)."
        )
        self.C_O2_local0 = _check_range(
            "C_O2_local0", C_O2_local0,
            self._C_O2_LOC0_MIN, self._C_O2_LOC0_MAX,
            "типично 0.10 мл O2/мл."
        )

        self._current_outputs = {}

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
    def get_state_size(self) -> int:
        return 3   # [C_O2_local, C_lactate_local, R_eff]

    def get_initial_state(self) -> np.ndarray:
        return np.array([self.C_O2_local0, self.C_lactate0, self.R_base])

    # ------------------------------------------------------------------
    # Ауторегуляция — вынесенные факторы
    # ------------------------------------------------------------------
    def _autoregulation_factor_O2(self, C_O2_local: float) -> float:
        """
        Метаболический фактор (гипоксия → вазодилатация).

        f_O2 = R_min + (1 − R_min) · (C_O2 / O2_norm)^k,  clip [R_min, 1.0]

        Свойства:
            C_O2 = 0        → f_O2 = R_min (максимальная вазодилатация)
            C_O2 = O2_norm  → f_O2 = 1.0   (сосуд в базовом тонусе)
            между ними      → монотонная кривая (линейная при k=1)

        При k > 1 дилатация включается резче при глубокой гипоксии;
        при k < 1 — плавнее.
        """
        ratio = float(np.clip(C_O2_local / max(self.O2_norm, 1e-6), 0.0, 1.0))
        f_O2 = self.R_min_factor + (1.0 - self.R_min_factor) * (ratio ** self.k_O2_autoreg)
        return float(np.clip(f_O2, self.R_min_factor, 1.0))

    def _autoregulation_factor_P(self, P_sa: float) -> float:
        """
        Миогенный фактор (высокое P_sa → вазоконстрикция).

        f_P = 1 + k_P · sign(ΔP) · max(|ΔP| − deadband, 0),  clip [R_min, R_max]

        Внутри deadband (|P_sa − P_norm| ≤ deadband) — f_P = 1.0.
        Используется и в _autoregulation_target, и в диагностических outputs.
        """
        excess = P_sa - self.P_sa_norm
        if abs(excess) <= self.P_myogenic_deadband:
            f_P = 1.0
        else:
            signed = excess - np.sign(excess) * self.P_myogenic_deadband
            f_P = 1.0 + self.k_P_myogenic * signed
        return float(np.clip(f_P, self.R_min_factor, self.R_max_factor))

    def _autoregulation_target(self, P_sa: float, C_O2_local: float) -> float:
        """
        Целевое R_eff — комбинация метаболической и миогенной регуляции.

        R_target = R_base · f_O2(C_O2_local) · f_P(P_sa)
        """
        f_O2 = self._autoregulation_factor_O2(C_O2_local)
        f_P = self._autoregulation_factor_P(P_sa)
        return self.R_base * f_O2 * f_P

    # ------------------------------------------------------------------
    # Производные
    # ------------------------------------------------------------------
    def get_derivatives(self, t, state, inputs):
        C_O2_loc_raw, C_lac_loc, R_eff = state

        # --- Входы ---
        P_sa         = float(inputs.get('P_sa', 85.0))
        P_sv         = float(inputs.get('P_sv', 5.0))
        C_a_O2       = float(inputs.get('C_a_O2', self.C_a_O2_norm))
        C_v_lactate  = float(inputs.get('C_v_lactate', 0.10))
        V_blood      = float(inputs.get('V_blood', 5000.0))
        V_blood      = max(V_blood, 1e-6)

        # --- Мягкие клипы состояния ---
        # Защита от LSODA retries, не от ошибок конфигурации.
        C_O2_loc = float(np.clip(C_O2_loc_raw, 0.0, max(C_a_O2, 0.0)))
        C_lac_loc = max(float(C_lac_loc), 0.0)
        R_eff = max(float(R_eff), 1e-3)

        # --- 1. Ауторегуляция: R_eff релаксирует к целевому ---
        R_target = self._autoregulation_target(P_sa, C_O2_loc)
        dR_eff = (R_target - R_eff) / self.tau_autoreg

        # --- 2. Кровоток через периферию ---
        Q_periph = (P_sa - P_sv) / R_eff
        Q_periph = max(Q_periph, 0.0)   # нет обратного тока

        # --- 3. Баланс O2 в ткани (flow-зависимый VO2) ---
        # Q_factor ∈ [0, 1.5] — нагрузочный множитель.
        # VO2_eff = VO2_base · (basal_frac + (1 − basal_frac)·Q_factor)
        # При Q = 0: VO2 = VO2_base · basal_frac (анаэробный резерв).
        Q_factor = float(np.clip(Q_periph / 20.0, 0.0, 1.5))
        VO2_eff = self.VO2_base * (
            self.VO2_basal_frac + (1.0 - self.VO2_basal_frac) * Q_factor
        )
        O2_delivery_rate = Q_periph * (C_a_O2 - C_O2_loc)
        dC_O2_loc = (O2_delivery_rate - VO2_eff) / self.V_tissue_eff
        # Мягкий пол: при C_O2_loc ≈ 0 не позволяем уходить в минус
        if C_O2_loc <= 1e-6 and dC_O2_loc < 0.0:
            dC_O2_loc = 0.0

        # --- 4. Лактат — базальная продукция + анаэробная надбавка ---
        hypoxia_severity = max(self.C_O2_lactate_threshold - C_O2_loc, 0.0)
        # Базальная продукция подобрана так, чтобы при нормоксии
        # (hypoxia_severity=0) держать dC_lac = 0 при C_lac = C_lactate0.
        lac_prod_base = (
            self.k_lactate_clear * self.C_lactate0
            + self.k_lactate_release * max(self.C_lactate0 - C_v_lactate, 0.0)
        )
        lac_prod_hypoxic = self.k_lactate_prod * hypoxia_severity
        lac_production = lac_prod_base + lac_prod_hypoxic
        lac_clearance  = self.k_lactate_clear * C_lac_loc
        lac_release    = self.k_lactate_release * max(C_lac_loc - C_v_lactate, 0.0)
        dC_lac_loc = lac_production - lac_clearance - lac_release

        # --- 5. Вклады в BloodPool ---
        # Диагностика потребления O2 (в общий баланс НЕ подмешивается —
        # централизовано в whole_body._compute_organ_flows).
        dC_O2_blood = -VO2_eff / V_blood
        # Выделение лактата в кровь
        dC_lactate_blood = (lac_release * self.V_tissue_eff) / V_blood

        # --- 6. Кэш выходов ---
        self._current_outputs = {
            # Гемодинамика
            'Q_peripheral': float(Q_periph),
            'R_eff':        float(R_eff),
            'R_target':     float(R_target),
            'P_tissue':     float(self.P_tissue0),

            # Ауторегуляция — теперь оба фактора согласованы с моделью
            'f_O2_autoreg': float(self._autoregulation_factor_O2(C_O2_loc)),
            'f_P_myogenic': float(self._autoregulation_factor_P(P_sa)),

            # Метаболизм
            'C_O2_local':      float(C_O2_loc),
            'C_lactate_local': float(C_lac_loc),
            'O2_consumption_periph': float(VO2_eff),
            'lactate_production':    float(lac_production),
            'lactate_release_to_blood': float(lac_release),

            # Производные для BloodPool
            '_diagnostic_dC_O2_blood': float(dC_O2_blood),
            'dC_lactate_blood':        float(dC_lactate_blood),
        }

        return np.array([dC_O2_loc, dC_lac_loc, dR_eff])

    def get_outputs(self, state):
        return self._current_outputs.copy()


# =====================================================================
# Быстрый тест (python peripheral_tissues.py)
# =====================================================================
if __name__ == "__main__":
    from scipy.integrate import solve_ivp

    pt = PeripheralTissues()
    print(f"State size: {pt.get_state_size()}")
    print(f"Initial state: {pt.get_initial_state()}")

    INPUTS = {'P_sa': 85.0, 'P_sv': 5.0, 'C_a_O2': 0.20,
              'C_v_lactate': 0.10, 'V_blood': 5000.0}

    def rhs(t, y):
        return pt.get_derivatives(t, y, INPUTS)

    sol = solve_ivp(rhs, (0, 60), pt.get_initial_state(),
                    method='LSODA', rtol=1e-6, atol=1e-8, max_step=0.1)
    y_end = sol.y[:, -1]
    pt.get_derivatives(sol.t[-1], y_end, INPUTS)
    out = pt.get_outputs(y_end)
    print("\nSteady state (здоровый, P_sa=85, C_a_O2=0.20):")
    for k in ('Q_peripheral', 'R_eff', 'C_O2_local', 'C_lactate_local',
              'O2_consumption_periph', 'lactate_production',
              'f_O2_autoreg', 'f_P_myogenic',
              '_diagnostic_dC_O2_blood'):
        print(f"  {k:28s} = {out[k]:+.6g}")

    print("\nГипоксия (P_sa=60, C_a_O2=0.12):")
    pt2 = PeripheralTissues()
    inputs_hyp = dict(INPUTS, P_sa=60.0, C_a_O2=0.12)
    def rhs2(t, y): return pt2.get_derivatives(t, y, inputs_hyp)
    sol2 = solve_ivp(rhs2, (0, 60), pt2.get_initial_state(),
                     method='LSODA', rtol=1e-6, atol=1e-8, max_step=0.1)
    pt2.get_derivatives(sol2.t[-1], sol2.y[:, -1], inputs_hyp)
    out2 = pt2.get_outputs(sol2.y[:, -1])
    for k in ('Q_peripheral', 'R_eff', 'C_O2_local',
              'f_O2_autoreg', 'f_P_myogenic',
              'O2_consumption_periph', 'lactate_production'):
        print(f"  {k:28s} = {out2[k]:+.6g}")