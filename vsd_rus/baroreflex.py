# baroreflex.py
"""
Барорефлекторная регуляция частоты сердечных сокращений.

Модель (три независимые ветви):
    ΔP = P_sa − P_set                                   [мм рт.ст.]

    dHR/dt      = (HR_target      − HR)      / tau_hr
    dInotr/dt   = (inotr_target   − inotr)   / tau_inotropy
    dVaso/dt    = (vaso_target    − vaso)    / tau_vaso

    HR_target    = HR_base·(1 − k_hr·ΔP),      clip [40, 180]
    inotr_target = 1 − k_inotropy·ΔP,          clip [0.5, 2.0]
    vaso_target  = 1 − k_vasomotor·ΔP,         clip [0.5, 2.0]

Состояние: [HR_current, baro_inotropy, baro_vasomotor]

При R_remodel > 2 симпатическое превышение (target > 1) ослабляется
множителем suppress = 1/(1 + max(R_remodel − 2, 0)); парасимпатическое
снижение HR сохраняется полностью.
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
    _K_INOTROPY_MIN, _K_INOTROPY_MAX = 0.0, 5.0
    _BARO_SYS_MIN, _BARO_SYS_MAX = 0.5, 2.0

    # --- Границы целевой ЧСС (клип HR_target) ---
    _HR_TARGET_MIN = 40.0
    _HR_TARGET_MAX = 180.0

    # --- Мягкий клип состояния HR (защита от LSODA retries) ---
    _HR_STATE_MIN = 10.0
    _HR_STATE_MAX = 300.0

    # --- Хронотропная ветвь ---
    _K_HR_MIN, _K_HR_MAX           = 0.0, 0.05
    _TAU_HR_MIN, _TAU_HR_MAX       = 0.1, 10.0

    # --- Инотропная ветвь ---
    _K_INOTROPY_NEW_MIN, _K_INOTROPY_NEW_MAX = 0.0, 0.02
    _TAU_INOTROPY_MIN, _TAU_INOTROPY_MAX     = 0.1, 15.0

    # --- Вазомоторная ветвь ---
    _K_VASOMOTOR_MIN, _K_VASOMOTOR_MAX = 0.0, 0.05
    _TAU_VASO_MIN, _TAU_VASO_MAX       = 1.0, 60.0    

    def __init__(self,
                P_set=80.0,
                HR_base=70.0,
                # --- хронотропная ветвь (быстрая) ---
                k_hr=0.008,
                tau_hr=2.0,
                # --- инотропная ветвь (средняя) ---
                k_inotropy=0.005,
                tau_inotropy=4.0,
                # --- вазомоторная ветвь (медленная) ---
                k_vasomotor=0.015,
                tau_vaso=10.0,
                # --- пульмональный рефлекс ПЖ (без изменений) ---
                k_inotropy_pulm=0.5):
        """
        Параметры:
            P_set      – заданное давление (мм рт.ст.), при котором ЧСС = HR_base
            HR_base    – базовая ЧСС (уд/мин)
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

        # --- Хронотропная ветвь ---
        self.k_hr = _check_range(
            "k_hr", k_hr, self._K_HR_MIN, self._K_HR_MAX,
            "1/(мм рт.ст.), типично 0.008. "
            "HR_target = HR_base·(1 − k_hr·ΔP), ΔP = P_sa − P_set."
        )
        self.tau_hr = _check_range(
            "tau_hr", tau_hr,
            self._TAU_HR_MIN, self._TAU_HR_MAX,
            "с, типично 2.0 (быстрая хронотропная ветвь)."
        )

        # --- Инотропная ветвь ---
        self.k_inotropy = _check_range(
            "k_inotropy", k_inotropy,
            self._K_INOTROPY_NEW_MIN, self._K_INOTROPY_NEW_MAX,
            "1/(мм рт.ст.), типично 0.005. "
            "baro_activation_target = 1 − k_inotropy·ΔP."
        )
        self.tau_inotropy = _check_range(
            "tau_inotropy", tau_inotropy,
            self._TAU_INOTROPY_MIN, self._TAU_INOTROPY_MAX,
            "с, типично 4.0 (средняя инотропная ветвь)."
        )

        # --- Вазомоторная ветвь ---
        self.k_vasomotor = _check_range(
            "k_vasomotor", k_vasomotor,
            self._K_VASOMOTOR_MIN, self._K_VASOMOTOR_MAX,
            "1/(мм рт.ст.), типично 0.015. "
            "R_sys_scale_target = 1 − k_vasomotor·ΔP."
        )
        self.tau_vaso = _check_range(
            "tau_vaso", tau_vaso,
            self._TAU_VASO_MIN, self._TAU_VASO_MAX,
            "с, типично 10.0 (медленная вазомоторная ветвь)."
        )

        # --- Пульмональный инотропный коэффициент (без изменений) ---
        self.k_inotropy_pulm = _check_range(
            "k_inotropy_pulm", k_inotropy_pulm,
            self._K_INOTROPY_MIN, self._K_INOTROPY_MAX,
            "безразмерный, типично 0.3–1.0 — пульмональный "
            "барорефлекс на ПЖ."
        )

        self._current_outputs = {}

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
    def get_state_size(self) -> int:
        return 3   # [HR_current, baro_inotropy, baro_vasomotor]

    def get_initial_state(self) -> np.ndarray:
        # Инотропная и вазомоторная ветви стартуют «в покое» = 1.0.
        # Это гарантирует, что при t=0 и P_sa = P_set все производные = 0.
        return np.array([self.HR_base, 1.0, 1.0])

    # ------------------------------------------------------------------
    # Основной метод — производные
    # ------------------------------------------------------------------
    def get_derivatives(self, t, state, inputs):
        # --- Разбор состояния с мягкими клипами (защита от LSODA retries) ---
        HR_raw = float(state[0])
        inotr_raw = float(state[1])
        vaso_raw = float(state[2])

        HR   = float(np.clip(HR_raw,   self._HR_STATE_MIN, self._HR_STATE_MAX))
        inotr = float(np.clip(inotr_raw, self._BARO_SYS_MIN, self._BARO_SYS_MAX))
        vaso  = float(np.clip(vaso_raw,  self._BARO_SYS_MIN, self._BARO_SYS_MAX))

        # --- Входы ---
        P_sa = float(inputs.get('P_sa', 90.0))
        P_pa = float(inputs.get('P_pa', 15.0))
        P_pa_set = float(inputs.get('P_pa_set', 15.0))
        R_rem = float(inputs.get('R_remodel', 1.0))

        # --- Абсолютное отклонение давления ---
        dP = P_sa - self.P_set

        # --- Целевые значения ветвей ---
        HR_target = self.HR_base * (1.0 - self.k_hr * dP)
        HR_target = float(np.clip(
            HR_target, self._HR_TARGET_MIN, self._HR_TARGET_MAX
        ))

        inotr_target = 1.0 - self.k_inotropy * dP
        inotr_target = float(np.clip(
            inotr_target, self._BARO_SYS_MIN, self._BARO_SYS_MAX
        ))

        vaso_target = 1.0 - self.k_vasomotor * dP
        vaso_target = float(np.clip(
            vaso_target, self._BARO_SYS_MIN, self._BARO_SYS_MAX
        ))

        # --- Хроническое подавление симпатической ветви ---
        # При R_remodel > 2 (Эйзенменгер) симпатический резерв исчерпан,
        # но парасимпатическое снижение HR сохраняется полностью.
        suppress = 1.0 / (1.0 + max(R_rem - 2.0, 0.0))
        if inotr_target > 1.0:
            inotr_target = 1.0 + (inotr_target - 1.0) * suppress
        if vaso_target > 1.0:
            vaso_target = 1.0 + (vaso_target - 1.0) * suppress

        # --- Производные первого порядка ---
        dHR     = (HR_target    - HR)    / self.tau_hr
        dInotr  = (inotr_target - inotr) / self.tau_inotropy
        dVaso   = (vaso_target  - vaso)  / self.tau_vaso

        # --- Пульмональная бароактивация (ПЖ) — без изменений ---
        pulm_excess = max(P_pa / max(P_pa_set, 1e-6) - 1.0, 0.0)
        baro_pulm = 1.0 + self.k_inotropy_pulm * pulm_excess

        # --- Диагностика ---
        self._current_outputs = {
            'HR': HR,
            'HR_target': HR_target,
            'hr_factor': HR / self.HR_base,

            # Выходы для heart.py (инотропия ЛЖ/ПЖ)
            'baro_activation': inotr,
            'baro_activation_rv': baro_pulm,

            # Выход для peripheral_tissues.py (вазомоторная ветвь)
            'R_sys_scale': vaso,

            # Диагностика для валидации Цели 3
            'baro_inotropy': inotr,
            'baro_vasomotor': vaso,
            'baro_inotropy_target': inotr_target,
            'baro_vasomotor_target': vaso_target,
            'suppress': float(suppress),
        }
        return np.array([dHR, dInotr, dVaso])

    def get_outputs(self, state):
        # Если кэш пуст (первый вызов после __init__) — вычисляем
        # диагностику из state, не трогая self._current_outputs.
        if self._current_outputs:
            return self._current_outputs.copy()

        HR_raw = float(state[0])
        inotr_raw = float(state[1])
        vaso_raw = float(state[2])
        HR = float(np.clip(HR_raw, self._HR_STATE_MIN, self._HR_STATE_MAX))
        inotr = float(np.clip(inotr_raw, self._BARO_SYS_MIN, self._BARO_SYS_MAX))
        vaso = float(np.clip(vaso_raw, self._BARO_SYS_MIN, self._BARO_SYS_MAX))
        return {
            'HR': HR,
            'HR_target': HR,
            'hr_factor': HR / self.HR_base,
            'baro_activation': inotr,
            'baro_activation_rv': 1.0,   # без P_pa недоступно
            'R_sys_scale': vaso,
            'baro_inotropy': inotr,
            'baro_vasomotor': vaso,
            'baro_inotropy_target': inotr,
            'baro_vasomotor_target': vaso,
            'suppress': 1.0,
        }