# baroreflex.py
"""
Барорефлекторная регуляция частоты сердечных сокращений.

Состояние: [HR_current] — текущая ЧСС (уд/мин).

Физиология:
    При ↑P_sa барорецепторы активируются → парасимпатика ↑ → ЧСС ↓
    При ↓P_sa барорецепторы расслабляются → симпатика ↑ → ЧСС ↑, инотропия ↑

Модель:
    HR_target = HR_base · (1 − gain·(P_sa − P_set)), clip [HR_min, HR_max]
    dHR/dt   = (HR_target − HR) / tau

    baro_activation = 1 + k_inotropy·max(1 − P_sa/P_set, 0)
        — множитель E_max для сердца при падении P_sa
        (1.0 при P_sa ≥ P_set, растёт при гипотензии)

Единицы:
    P_set, P_sa   — мм рт.ст.
    HR, HR_base   — уд/мин
    gain          — безразмерный (относительное изменение ЧСС на 1 мм рт.ст.)
    tau           — с
    k_inotropy    — безразмерный
"""

import numpy as np
from organ_base import OrganModel


class Baroreflex(OrganModel):
    """
    Модель барорефлекторной регуляции ЧСС.
    """

    # --- Санити-пороги для валидации конфигурации ---
    _P_SET_MIN, _P_SET_MAX = 40.0, 200.0
    _HR_BASE_MIN, _HR_BASE_MAX = 30.0, 200.0
    _GAIN_MIN, _GAIN_MAX = 0.0, 0.1
    _TAU_MIN, _TAU_MAX = 0.1, 60.0
    _K_INOTROPY_MIN, _K_INOTROPY_MAX = 0.0, 5.0

    # --- Границы целевой ЧСС (клип HR_target) ---
    _HR_TARGET_MIN = 40.0
    _HR_TARGET_MAX = 180.0

    # --- Мягкий клип состояния HR (защита от LSODA retries) ---
    _HR_STATE_MIN = 10.0
    _HR_STATE_MAX = 300.0

    def __init__(self,
                 P_set=80.0,
                 HR_base=70.0,
                 gain=0.002,
                 tau=2.0,
                 k_inotropy=0.5):
        """
        Параметры:
            P_set      – заданное давление (мм рт.ст.), при котором ЧСС = HR_base
            HR_base    – базовая ЧСС (уд/мин)
            gain       – коэффициент усиления (относительное изменение ЧСС
                         на 1 мм рт.ст.)
            tau        – постоянная времени рефлекса (с)
            k_inotropy – коэффициент симпатической инотропии
                         (множитель baro_activation)
        """

        # =================================================================
        # Валидация конфигурации — fail-fast при инициализации.
        # Параметры приходят из YAML и не меняются во время симуляции;
        # ошибки в них должны ловиться один раз, а не в горячем пути RHS.
        # =================================================================
        def _check_range(name, v, lo, hi, typical=""):
            v = float(v)
            if not np.isfinite(v) or not (lo <= v <= hi):
                raise ValueError(
                    f"Baroreflex: {name}={v} вне [{lo}, {hi}]. {typical}"
                )
            return v

        # --- Точка равновесия и базовая ЧСС ---
        self.P_set = _check_range(
            "P_set", P_set, self._P_SET_MIN, self._P_SET_MAX,
            "мм рт.ст., типично 80 — при этом давлении ЧСС = HR_base."
        )
        self.HR_base = _check_range(
            "HR_base", HR_base, self._HR_BASE_MIN, self._HR_BASE_MAX,
            "уд/мин, типично 70."
        )

        # --- Коэффициент усиления ---
        self.gain = _check_range(
            "gain", gain, self._GAIN_MIN, self._GAIN_MAX,
            "безразмерный, типично 0.002–0.015. "
            "0 — нет регуляции; 0.015 → ±100% ЧСС при ΔP≈65 мм рт.ст."
        )

        # --- Постоянная времени ---
        self.tau = _check_range(
            "tau", tau, self._TAU_MIN, self._TAU_MAX,
            "с, типично 2 (быстрый барорефлекс)."
        )

        # --- Инотропный коэффициент ---
        self.k_inotropy = _check_range(
            "k_inotropy", k_inotropy,
            self._K_INOTROPY_MIN, self._K_INOTROPY_MAX,
            "безразмерный, типично 0.5–1.5. "
            "baro_activation = 1 + k·max(1 − P_sa/P_set, 0)."
        )

        self._current_outputs = {}

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
    def get_state_size(self) -> int:
        return 1   # только HR_current

    def get_initial_state(self) -> np.ndarray:
        return np.array([self.HR_base])

    # ------------------------------------------------------------------
    # Основной метод — производные
    # ------------------------------------------------------------------
    def get_derivatives(self, t, state, inputs):
        # --- Разбор состояния с мягким клипом (защита от LSODA retries) ---
        HR_raw = float(state[0])
        HR = float(np.clip(HR_raw, self._HR_STATE_MIN, self._HR_STATE_MAX))

        # --- Вход ---
        P_sa = float(inputs.get('P_sa', 90.0))

        # --- Целевая ЧСС (обратная зависимость от давления) ---
        HR_target = self.HR_base * (1.0 - self.gain * (P_sa - self.P_set))
        HR_target = float(np.clip(
            HR_target, self._HR_TARGET_MIN, self._HR_TARGET_MAX
        ))

        # --- Производная (линейная динамика первого порядка) ---
        dHR = (HR_target - HR) / self.tau

        # --- Диагностика ---
        self._current_outputs = {
            'HR': HR,
            'HR_target': HR_target,
            'hr_factor': HR / self.HR_base,
            'baro_activation': 1.0 + self.k_inotropy * max(
                1.0 - P_sa / self.P_set, 0.0
            ),
        }
        return np.array([dHR])

    def get_outputs(self, state):
        return self._current_outputs.copy()