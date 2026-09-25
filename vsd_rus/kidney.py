# kidney.py
import numpy as np
from organ_base import OrganModel


class KidneyHemodynamic(OrganModel):
    """
    Модель почек здорового человека.
    Авторегуляция СКФ с сигмоидальной зависимостью (более физиологичная).

    Алгебраический орган (0 состояний). Рапортует в whole_body через:
        Q_renal, GFR, urine_output, dV_blood, dC_tox.

    Единицы:
        P          — мм рт.ст.
        Q          — мл/с
        R          — мм рт.ст.·с/мл
        GFR        — мл/с (НЕ мл/мин!)
        dC_tox     — масса/(мл·с)
    """

    # --- Санити-пороги для валидации конфигурации ---
    _GFR_BASE_MIN, _GFR_BASE_MAX = 30.0, 250.0
    _P_AUTOREG_MIN, _P_AUTOREG_MAX = 40.0, 200.0
    _AUTOREG_AMP_MIN, _AUTOREG_AMP_MAX = 0.0, 3.0
    _AUTOREG_SLOPE_MIN, _AUTOREG_SLOPE_MAX = 1e-3, 1.0
    _TOXIN_FRAC_MIN, _TOXIN_FRAC_MAX = 0.0, 1.0
    _REABS_FRAC_MIN, _REABS_FRAC_MAX = 0.5, 0.999

    _R_RENAL_MIN, _R_RENAL_MAX = 0.1, 50.0
    _BASAL_URINE_MIN, _BASAL_URINE_MAX = 0.0, 0.1
    _RBF_TARGET_MIN, _RBF_TARGET_MAX = 5.0, 100.0

    def __init__(self,
                 GFR_base=120.0,
                 P_autoreg=95.0,
                 autoreg_amplitude=0.6,       # амплитуда изменения СКФ (0.4..1.6)
                 autoreg_slope=0.025,         # крутизна сигмоиды
                 toxin_clearance_frac=0.2,
                 volume_reabsorption_frac=0.99,
                 renal_resistance=None,
                 basal_urine_output: float = 0.005,
                 RBF_target: float = 20.0):

        # =================================================================
        # Валидация конфигурации — fail-fast при инициализации.
        # Параметры приходят из YAML и не меняются во время симуляции;
        # ошибки в них должны ловиться один раз, а не в горячем пути RHS.
        # =================================================================
        def _check_range(name, v, lo, hi, typical=""):
            v = float(v)
            if not np.isfinite(v) or not (lo <= v <= hi):
                raise ValueError(
                    f"KidneyHemodynamic: {name}={v} вне [{lo}, {hi}]. {typical}"
                )
            return v

        # --- Базовые показатели ---
        self.GFR_base = _check_range(
            "GFR_base", GFR_base,
            self._GFR_BASE_MIN, self._GFR_BASE_MAX,
            "мл/мин, типично 120."
        )
        self.P_autoreg = _check_range(
            "P_autoreg", P_autoreg,
            self._P_AUTOREG_MIN, self._P_AUTOREG_MAX,
            "мм рт.ст., типично 95 — точка якоря ауторегуляции."
        )

        # --- Ауторегуляция GFR (ветвь прорыва) ---
        self.autoreg_amplitude = _check_range(
            "autoreg_amplitude", autoreg_amplitude,
            self._AUTOREG_AMP_MIN, self._AUTOREG_AMP_MAX,
            "безразмерный, типично 0.6 (превышение над плато при P>>P_high)."
        )
        self.autoreg_slope = _check_range(
            "autoreg_slope", autoreg_slope,
            self._AUTOREG_SLOPE_MIN, self._AUTOREG_SLOPE_MAX,
            "1/(мм рт.ст.), типично 0.025 — крутизна ветви прорыва."
        )

        # --- Клиренс токсина ---
        self.toxin_clearance_frac = _check_range(
            "toxin_clearance_frac", toxin_clearance_frac,
            self._TOXIN_FRAC_MIN, self._TOXIN_FRAC_MAX,
            "доля, типично 0.2 (filtration × secretion)."
        )

        # --- Реабсорбция ---
        self.volume_reabsorption_frac = _check_range(
            "volume_reabsorption_frac", volume_reabsorption_frac,
            self._REABS_FRAC_MIN, self._REABS_FRAC_MAX,
            "доля, типично 0.99 (99% воды возвращается)."
        )

        # --- Целевой RBF (для auto-consistent renal_resistance) ---
        self.RBF_target = _check_range(
            "RBF_target", RBF_target,
            self._RBF_TARGET_MIN, self._RBF_TARGET_MAX,
            "мл/с, типично 20 (1200 мл/мин)."
        )

        # --- renal_resistance: auto-consistent или заданное явно ---
        if renal_resistance is None:
            # Согласованное значение: R_base подбирается под RBF_target
            # так, чтобы RBF(95) = RBF_target.
            # P_sv_assumed = 5 мм рт.ст. (типичное венозное давление почки)
            self.renal_resistance = (self.P_autoreg - 5.0) / self.RBF_target
        else:
            self.renal_resistance = _check_range(
                "renal_resistance", renal_resistance,
                self._R_RENAL_MIN, self._R_RENAL_MAX,
                "мм рт.ст.·с/мл, типично 4.5 = (95−5)/20. "
                "Или передайте None для авто-расчёта под RBF_target."
            )

        # --- Базальный диурез ---
        self.basal_urine_output = _check_range(
            "basal_urine_output", basal_urine_output,
            self._BASAL_URINE_MIN, self._BASAL_URINE_MAX,
            "мл/с, типично 0.005 (=0.3 мл/мин). "
            "Пол диуреза при преренальной азотемии."
        )

        self._current_outputs = {}

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
    def get_state_size(self):
        return 0

    def get_initial_state(self):
        return np.array([])

    def get_derivatives(self, t, state, inputs):
        return np.array([])

    def get_outputs(self, state):
        return self._current_outputs.copy()

    # ------------------------------------------------------------------
    # GFR — три-веточная кривая ауторегуляции
    # ------------------------------------------------------------------
    def _compute_gfr(self, P_art: float) -> float:
        """
        GFR(P_art) — реалистичная ауторегуляция почек.

        Три режима с гладкими переходами:
        • P < P_low   (70 мм рт.ст.) — преренальная ветвь, GFR линейно → 0
        • P_low..P_high (70..180)    — плато ауторегуляции, GFR ≈ GFR_base
        • P > P_high  (180)          — прорыв, GFR растёт с давлением

        Параметры:
        P_autoreg        — точка «якоря» (обычно 95)
        autoreg_amplitude — превышение над плато при P >> P_high
        autoreg_slope     — крутизна ветви прорыва
        """
        P_low  = 70.0
        P_high = 180.0

        # --- Три ветви (значения factor) ---
        f_low = 1.0 / (1.0 + np.exp(-0.15 * (P_art - 55.0)))

        f_plateau = 1.0                                             # плоско
        f_high    = 1.0 + self.autoreg_amplitude * np.tanh(
                        self.autoreg_slope * (P_art - P_high))      # >180

        # --- Гладкие веса (сигмоиды) ---
        k_low  = 6.0     # ширина перехода в районе P_low
        k_high = 10.0    # ширина перехода в районе P_high
        w_low  = 0.5 * (1.0 - np.tanh((P_art - P_low) / k_low))
        w_high = 0.5 * (1.0 + np.tanh((P_art - P_high) / k_high))
        w_pl   = max(1.0 - w_low - w_high, 0.0)

        factor = w_low * f_low + w_pl * f_plateau + w_high * f_high
        return self.GFR_base * float(np.clip(factor, 0.0, 2.2))

    # ------------------------------------------------------------------
    # RBF-стабилизация
    # ------------------------------------------------------------------
    def _renal_resistance_eff(self, P_sa: float) -> float:
        P_low, P_high = 70.0, 180.0
        P_sv_assumed = 5.0
        R_base = self.renal_resistance

        # R_eff если бы RBF был стабилизирован
        RBF_eff = self.RBF_target * (
            1.0 + 0.3 * max(np.tanh((P_sa - P_high) / 10.0), 0.0)
        )
        R_autoreg = (P_sa - P_sv_assumed) / max(RBF_eff, 1e-6)
        # R_eff при максимальной вазодилатации
        R_baseline = R_base

        # Гладкий вес: при P << P_low  → R_baseline;  при P >> P_low → R_autoreg
        w_auto = 0.5 * (1.0 + np.tanh((P_sa - P_low) / 5.0))
        R_eff = (1.0 - w_auto) * R_baseline + w_auto * R_autoreg

        return float(np.clip(R_eff, R_base * 0.3, R_base * 5.0))

    # ------------------------------------------------------------------
    # Давление-натриурез
    # ------------------------------------------------------------------
    def _reabsorption_frac(self, P_sa: float) -> float:
        """
        Давление-натриурез: при ↑P_sa реабсорбция Na/H2O падает.

        Наблюдение: удвоение почечного перфузионного давления
        увеличивает экскрецию Na в 2-3 раза.

        Реализация: линейная модуляция вокруг базовой точки.
        """
        base = self.volume_reabsorption_frac   # 0.99
        sensitivity = 2e-4                     # доля на мм рт.ст.
        frac = base - sensitivity * (P_sa - self.P_autoreg)
        return float(np.clip(frac, 0.90, 0.999))

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------
    def compute_effects(self, P_sa, P_sv, C_tox, V_blood):
        """
        Вычисляет эффекты почек на гемодинамику и клиренс токсина.

        Возвращает:
            Q_renal      — мл/с
            GFR          — мл/с (НЕ мл/мин!)
            urine_output — мл/с
            dV_blood     — мл/с
            dC_tox       — масса/(мл·с)
        """
        R_eff = self._renal_resistance_eff(P_sa)
        Q_renal = max((P_sa - P_sv) / R_eff, 0.0)
        GFR = self._compute_gfr(P_sa) / 60.0   # мл/мин -> мл/с

        toxin_filtered = GFR * C_tox
        toxin_excreted = self.toxin_clearance_frac * toxin_filtered
        dC_tox = -toxin_excreted / max(V_blood, 1e-6)

        reabs = self._reabsorption_frac(P_sa)
        urine_output = max(GFR * (1 - reabs), self.basal_urine_output)
        dV_blood = -urine_output

        self._current_outputs = {
            'Q_renal': Q_renal,
            'GFR': GFR,
            'urine_output': urine_output,
        }
        return {
            'dC_tox': dC_tox,
            'dV_blood': dV_blood,
            'Q_renal': Q_renal,
            'GFR': GFR,
            'urine_output': urine_output,
        }