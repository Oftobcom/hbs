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
    Венозный возврат O2 = C_O2_loc (well-mixed tissue approximation).
    """

    # --- Санити-пороги для валидации конфигурации ---
    _R_BASE_MIN, _R_BASE_MAX = 0.1, 50.0
    _C_TISSUE_MIN, _C_TISSUE_MAX = 0.1, 1000.0
    _P_TISSUE0_MIN, _P_TISSUE0_MAX = 0.0, 100.0

    _O2_NORM_MIN, _O2_NORM_MAX = 0.05, 0.25
    _K_O2_MIN, _K_O2_MAX = 0.1, 3.0
    _R_MIN_FACTOR_MIN, _R_MIN_FACTOR_MAX = 0.1, 0.9
    _R_MYOGENIC_MIN_MIN, _R_MYOGENIC_MIN_MAX = 0.1, 0.9
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


    @staticmethod
    def _check_range(name, v, lo, hi, typical=""):
        if v is None:
            raise ValueError(
                f"PeripheralTissues: {name} не задан (None). "
                f"Все параметры обязательны; дефолты удалены. "
                f"Задайте peripheral.{name} в physiology.yaml."
            )
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(
                f"PeripheralTissues: {name}={v!r} должен быть числом, "
                f"получено {type(v).__name__}."
            )
        v = float(v)
        if not np.isfinite(v) or not (lo <= v <= hi):
            raise ValueError(
                f"PeripheralTissues: {name}={v} вне [{lo}, {hi}]. {typical}"
            )
        return v

    # ------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------
    def __init__(self,
        *,
        R_base: float,
        C_tissue: float,
        P_tissue0: float,
        O2_norm: float,
        k_O2_autoreg: float,
        R_min_factor: float,
        R_myogenic_min_factor: float,
        tau_autoreg: float,
        k_P_myogenic: float,
        P_sa_norm: float,
        R_max_factor: float,
        P_myogenic_deadband: float,
        VO2_base: float,
        VO2_basal_frac: float,
        V_tissue_eff: float,
        C_a_O2_norm: float,
        C_O2_lactate_threshold: float,
        k_lactate_prod: float,
        k_lactate_clear: float,
        k_lactate_release: float,
        C_lactate0: float,
        C_O2_local0: float):


        # --- Гемодинамика ---
        self.R_base = self._check_range(
            "R_base", R_base, self._R_BASE_MIN, self._R_BASE_MAX,
            "типично 2–6 мм рт.ст.·с/мл (после калибровки под target_MAP/CO)."
        )
        self.C_tissue = self._check_range(
            "C_tissue", C_tissue, self._C_TISSUE_MIN, self._C_TISSUE_MAX,
            "не используется в текущей модели; оставлено для расширения."
        )
        self.P_tissue0 = self._check_range(
            "P_tissue0", P_tissue0, self._P_TISSUE0_MIN, self._P_TISSUE0_MAX,
            "диагностическое значение, типично 15 мм рт.ст."
        )

        # --- Метаболическая ауторегуляция ---
        self.O2_norm = self._check_range(
            "O2_norm", O2_norm, self._O2_NORM_MIN, self._O2_NORM_MAX,
            "типично 0.15 мл O2/мл."
        )
        self.k_O2_autoreg = self._check_range(
            "k_O2_autoreg", k_O2_autoreg, self._K_O2_MIN, self._K_O2_MAX,
            "типично 1.0 (линейно). k>1 усиливает дилатацию при глубокой гипоксии, "
            "k<1 делает реакцию более вялой."
        )
        self.R_min_factor = self._check_range(
            "R_min_factor", R_min_factor,
            self._R_MIN_FACTOR_MIN, self._R_MIN_FACTOR_MAX,
            "типично 0.4–0.5."
        )
        self.R_myogenic_min_factor = self._check_range(
            "R_myogenic_min_factor", R_myogenic_min_factor,
            self._R_MYOGENIC_MIN_MIN, self._R_MYOGENIC_MIN_MAX,
            "макс. миогенная вазодилатация; типично 0.55–0.7 "
            "(слабее метаболической R_min_factor)."
        )
        self.tau_autoreg = self._check_range(
            "tau_autoreg", tau_autoreg,
            self._TAU_AUTOREG_MIN, self._TAU_AUTOREG_MAX
        )

        # --- Миогенная ауторегуляция ---
        self.k_P_myogenic = self._check_range(
            "k_P_myogenic", k_P_myogenic,
            self._K_P_MYOGENIC_MIN, self._K_P_MYOGENIC_MAX,
            "типично 0.002–0.005."
        )
        self.P_sa_norm = self._check_range(
            "P_sa_norm", P_sa_norm, self._P_SA_NORM_MIN, self._P_SA_NORM_MAX,
            "типично 90 мм рт.ст."
        )
        self.R_max_factor = self._check_range(
            "R_max_factor", R_max_factor,
            self._R_MAX_FACTOR_MIN, self._R_MAX_FACTOR_MAX,
            "типично 2.5 (макс. вазоконстрикция)."
        )
        self.P_myogenic_deadband = self._check_range(
            "P_myogenic_deadband", P_myogenic_deadband,
            self._DEADBAND_MIN, self._DEADBAND_MAX,
            "типично 10 мм рт.ст."
        )

        # --- Потребление O2 ---
        self.VO2_base = self._check_range(
            "VO2_base", VO2_base, self._VO2_BASE_MIN, self._VO2_BASE_MAX,
            "типично 1.5 мл/с (мышцы+кожа, ~90 мл/мин)."
        )
        self.VO2_basal_frac = self._check_range(
            "VO2_basal_frac", VO2_basal_frac,
            self._VO2_BASAL_FRAC_MIN, self._VO2_BASAL_FRAC_MAX,
            "типично 0.05 — анаэробный резерв при Q→0."
        )
        self.V_tissue_eff = self._check_range(
            "V_tissue_eff", V_tissue_eff,
            self._V_TISSUE_MIN, self._V_TISSUE_MAX,
            "типично 400 мл."
        )
        self.C_a_O2_norm = self._check_range(
            "C_a_O2_norm", C_a_O2_norm,
            self._C_A_O2_NORM_MIN, self._C_A_O2_NORM_MAX,
            "типично 0.20 мл O2/мл."
        )

        # --- Лактат ---
        self.C_O2_lactate_threshold = self._check_range(
            "C_O2_lactate_threshold", C_O2_lactate_threshold,
            self._LAC_THRESH_MIN, self._LAC_THRESH_MAX,
            "типично 0.08 мл O2/мл — порог начала анаэробного гликолиза."
        )
        self.k_lactate_prod = self._check_range(
            "k_lactate_prod", k_lactate_prod,
            self._K_LAC_PROD_MIN, self._K_LAC_PROD_MAX,
            "типично 0.05 мг/(мл·с)."
        )
        self.k_lactate_clear = self._check_range(
            "k_lactate_clear", k_lactate_clear,
            self._K_LAC_CLEAR_MIN, self._K_LAC_CLEAR_MAX,
            "типично 0.02 1/с."
        )
        self.k_lactate_release = self._check_range(
            "k_lactate_release", k_lactate_release,
            self._K_LAC_REL_MIN, self._K_LAC_REL_MAX,
            "типично 0.05 1/с."
        )
        self.C_lactate0 = self._check_range(
            "C_lactate0", C_lactate0,
            self._C_LAC0_MIN, self._C_LAC0_MAX,
            "типично 0.10 мг/мл (=1 мМ)."
        )
        self.C_O2_local0 = self._check_range(
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
        return float(np.clip(f_P, self.R_myogenic_min_factor, self.R_max_factor))

    def _autoregulation_target(self, P_sa: float, C_O2_local: float,
                            baro_scale: float = 1.0) -> float:
        """
        R_target = R_base · f_O2 · f_P · baro_scale.

        Все три фактора — независимые мультипликативные модуляции
        радиуса сосуда (Пуазейль: R ∝ 1/r⁴).
        """
        f_O2 = self._autoregulation_factor_O2(C_O2_local)
        f_P = self._autoregulation_factor_P(P_sa)
        R_target = self.R_base * f_O2 * f_P * baro_scale
        return float(np.clip(
            R_target,
            self.R_base * self.R_min_factor,
            self.R_base * self.R_max_factor,
        ))

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
        C_O2_loc = float(np.clip(C_O2_loc_raw, 0.0, max(C_a_O2, 0.0)))
        C_lac_loc = max(float(C_lac_loc), 0.0)
        R_eff = max(float(R_eff), 1e-3)

        # --- 1. Ауторегуляция: R_eff релаксирует к целевому ---
        baro_scale = float(inputs.get('baro_sys_scale', 1.0))
        R_target = self._autoregulation_target(P_sa, C_O2_loc, baro_scale)
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
            'baro_scale_applied': float(baro_scale),
            'R_target_pre_baro':  float(self.R_base
                                        * self._autoregulation_factor_O2(C_O2_loc)
                                        * self._autoregulation_factor_P(P_sa)),

            # Метаболизм
            'C_O2_local':      float(C_O2_loc),
            'C_lactate_local': float(C_lac_loc),
            'O2_consumption_periph': float(VO2_eff),
            'lactate_production':    float(lac_production),
            'lactate_release_to_blood': float(lac_release),

            # Производные для BloodPool
            '_diagnostic_dC_O2_blood': float(dC_O2_blood),
            'dC_lactate_blood':        float(dC_lactate_blood),

            # Венозный возврат из периферии = тканевая концентрация.
            # whole_body использует этот ключ для баланса O2 крови вместо
            # метаболического запроса VO2_eff.
            'C_v_O2_periph': float(C_O2_loc),
        }

        return np.array([dC_O2_loc, dC_lac_loc, dR_eff])

    def get_outputs(self, state):
        return self._current_outputs.copy()

