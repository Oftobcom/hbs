# gitract.py
import numpy as np
from organ_base import OrganModel


class GITract(OrganModel):
    """
    Модель желудочно-кишечного тракта здорового человека.
    Портальная гемодинамика (двухкомпартментный Windkessel) и всасывание.

    Состояние: [P_art, P_cap]
        P_art — давление в артериальном сегменте ЖКТ (мм рт.ст.)
        P_cap — давление в капиллярном сегменте ЖКТ (мм рт.ст.)

    Гемодинамика:
        Q_in  = (P_sa  − P_art)   / R_art
        Q_cap = (P_art − P_cap)   / R_cap
        Q_out = (P_cap − P_portal)/ R_venous
        dP_art = (Q_in  − Q_cap)/C_art
        dP_cap = (Q_cap − Q_out)/C_cap

    Всасывание:
        absorption_X = k_absorption_X · intake_X · f_portal(P_portal)
        f_portal — падение всасывания при портальной гипертензии

    Единицы:
        P        — мм рт.ст.
        Q        — мл/с
        R        — мм рт.ст.·с/мл
        C        — мл/мм рт.ст.
        intake   — мл/с (внешний источник жидкости/нутриентов)
    """

    # --- Санити-пороги для валидации конфигурации ---
    _R_ART_MIN, _R_ART_MAX = 0.1, 20.0
    _R_CAP_MIN, _R_CAP_MAX = 0.1, 20.0
    _R_VENOUS_MIN, _R_VENOUS_MAX = 0.1, 20.0
    _C_ART_MIN, _C_ART_MAX = 0.1, 50.0
    _C_CAP_MIN, _C_CAP_MAX = 0.1, 50.0

    _K_ABS_WATER_MIN, _K_ABS_WATER_MAX = 0.0, 1.0
    _K_ABS_NUTRIENTS_MIN, _K_ABS_NUTRIENTS_MAX = 0.0, 1.0
    _PORTAL_SENS_MIN, _PORTAL_SENS_MAX = 0.0, 1.0

    _P_ART0_MIN, _P_ART0_MAX = 0.0, 200.0
    _P_CAP0_MIN, _P_CAP0_MAX = 0.0, 200.0

    # --- Параметры модели портальной ауторегуляции ---
    _P_PORTAL_NORM = 8.0        # мм рт.ст. — норм. портальное давление
    _ABS_FACTOR_MIN = 0.2       # минимальный фактор всасывания при гипертензии

    def __init__(self,
                 R_art=1.8, R_cap=1.2, R_venous=1.5,
                 C_art=2.0, C_cap=5.0,
                 k_absorption_water=0.1,
                 k_absorption_nutrients=0.05,
                 portal_pressure_sensitivity=0.02,
                 P_art0=60.0, P_cap0=20.0):

        # =================================================================
        # Валидация конфигурации — fail-fast при инициализации.
        # Параметры приходят из YAML и не меняются во время симуляции;
        # ошибки в них должны ловиться один раз, а не в горячем пути RHS.
        # =================================================================
        def _check_range(name, v, lo, hi, typical=""):
            v = float(v)
            if not np.isfinite(v) or not (lo <= v <= hi):
                raise ValueError(
                    f"GITract: {name}={v} вне [{lo}, {hi}]. {typical}"
                )
            return v

        # --- Сопротивления (гемодинамика) ---
        self.R_art = _check_range(
            "R_art", R_art, self._R_ART_MIN, self._R_ART_MAX,
            "мм рт.ст.·с/мл, типично 1.8 (артериальный сегмент ЖКТ)."
        )
        self.R_cap = _check_range(
            "R_cap", R_cap, self._R_CAP_MIN, self._R_CAP_MAX,
            "мм рт.ст.·с/мл, типично 1.2 (капиллярный сегмент)."
        )
        self.R_venous = _check_range(
            "R_venous", R_venous, self._R_VENOUS_MIN, self._R_VENOUS_MAX,
            "мм рт.ст.·с/мл, типично 1.5 (венозный отток → портальная вена)."
        )

        # --- Комплаенсы ---
        self.C_art = _check_range(
            "C_art", C_art, self._C_ART_MIN, self._C_ART_MAX,
            "мл/мм рт.ст., типично 2.0."
        )
        self.C_cap = _check_range(
            "C_cap", C_cap, self._C_CAP_MIN, self._C_CAP_MAX,
            "мл/мм рт.ст., типично 5.0."
        )

        # --- Всасывание ---
        self.k_abs_water = _check_range(
            "k_absorption_water", k_absorption_water,
            self._K_ABS_WATER_MIN, self._K_ABS_WATER_MAX,
            "доля, типично 0.1 — коэффициент всасывания воды."
        )
        self.k_abs_nutrients = _check_range(
            "k_absorption_nutrients", k_absorption_nutrients,
            self._K_ABS_NUTRIENTS_MIN, self._K_ABS_NUTRIENTS_MAX,
            "доля, типично 0.05 — коэффициент всасывания нутриентов."
        )

        # --- Портальная ауторегуляция всасывания ---
        self.portal_pressure_sensitivity = _check_range(
            "portal_pressure_sensitivity", portal_pressure_sensitivity,
            self._PORTAL_SENS_MIN, self._PORTAL_SENS_MAX,
            "1/(мм рт.ст.), типично 0.02 — падение всасывания при гипертензии."
        )

        # --- Начальные давления ---
        self.P_art0 = _check_range(
            "P_art0", P_art0, self._P_ART0_MIN, self._P_ART0_MAX,
            "мм рт.ст., типично 60."
        )
        self.P_cap0 = _check_range(
            "P_cap0", P_cap0, self._P_CAP0_MIN, self._P_CAP0_MAX,
            "мм рт.ст., типично 20."
        )

        self._current_outputs = {}

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
    def get_state_size(self):
        return 2   # [P_art, P_cap]

    def get_initial_state(self):
        return np.array([self.P_art0, self.P_cap0])

    # ------------------------------------------------------------------
    # Портальная ауторегуляция всасывания
    # ------------------------------------------------------------------
    def _absorption_factor(self, P_portal: float) -> float:
        """
        Фактор всасывания в зависимости от портального давления.

        P_portal ≤ P_norm (=8)  →  1.0     (нормальное всасывание)
        P_portal > P_norm       →  1 − k·(P − P_norm), clip ≥ 0.2

        Логика: при портальной гипертензии (>8 мм рт.ст.) отёк стенки
        кишки снижает всасывание, но не до нуля — даже при тяжёлой
        гипертензии сохраняется ~20% базального уровня.
        """
        P = float(P_portal)
        if P <= self._P_PORTAL_NORM:
            return 1.0
        factor = 1.0 - self.portal_pressure_sensitivity * (P - self._P_PORTAL_NORM)
        return float(max(self._ABS_FACTOR_MIN, factor))

    # ------------------------------------------------------------------
    # Основной метод — производные
    # ------------------------------------------------------------------
    def get_derivatives(self, t, state, inputs):
        P_art_raw, P_cap_raw = state

        # --- Мягкие клипы состояния (защита от LSODA retries) ---
        P_art = max(float(P_art_raw), 0.0)
        P_cap = max(float(P_cap_raw), 0.0)

        # --- Входы ---
        P_sa        = float(inputs.get('P_sa', 80.0))
        P_portal    = float(inputs.get('P_portal', inputs.get('P_sv', 5.0)))
        intake_water     = float(inputs.get('intake_water', 0.0))
        intake_nutrients = float(inputs.get('intake_nutrients', 0.0))

        # --- Гемодинамика ---
        Q_in  = (P_sa   - P_art) / self.R_art
        Q_cap = (P_art  - P_cap) / self.R_cap
        Q_out = (P_cap  - P_portal) / self.R_venous

        # --- Всасывание ---
        abs_factor = self._absorption_factor(P_portal)
        absorption_water     = self.k_abs_water     * intake_water     * abs_factor
        absorption_nutrients = self.k_abs_nutrients * intake_nutrients * abs_factor

        # --- Производные давлений ---
        dP_art = (Q_in  - Q_cap) / self.C_art
        dP_cap = (Q_cap - Q_out) / self.C_cap

        # --- Кэш выходов ---
        self._current_outputs = {
            'Q_out': Q_out,
            'absorption_water': absorption_water,
            'absorption_nutrients': absorption_nutrients,
            'portal_pressure_factor': abs_factor,
        }
        return np.array([dP_art, dP_cap])

    def get_outputs(self, state):
        return self._current_outputs.copy()