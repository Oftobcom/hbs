# gas_exchange.py
"""
Альвеолярно-капиллярный газообмен O2/CO2.

Физиологические соглашения:
    - C_O2, C_CO2 измеряются в мл газа / мл крови (совпадает с BloodPool)
    - Нормальные значения:
        C_v_O2  = 0.15   C_a_O2  = 0.20
        C_v_CO2 = 0.52   C_a_CO2 = 0.48
    - Кривая диссоциации O2 — уравнение Хилла
      (P50 = 26.8 мм рт. ст., n = 2.7, формула Северингауза)
    - Кривая диссоциации CO2 — линейная аппроксимация в диапазоне 40–50 мм рт. ст.

Знаки Q_shunt:
    Q_shunt > 0  — лево-правый шунт (L→R), SaO2 не снижается
    Q_shunt < 0  — право-левый шунт (R→L), SaO2 падает
    Q_shunt = 0  — здоровый

Единая формула системного кровотока (из баланса heart.py):
    Q_s = Q_p - Q_shunt
"""

import numpy as np
from organ_base import OrganModel


class GasExchange(OrganModel):
    """
    Модель газообмена O2/CO2 в лёгких.

    Состояния нет — алгебраический орган, вызывается через compute_effects.

    Единицы:
        P         — мм рт.ст.
        C         — мл газа / мл крови
        Q         — мл/с
        Hb        — г/дл
        alpha_O2  — мл O2 / (дл · мм рт.ст.)
    """

    # --- Санити-пороги для валидации конфигурации ---
    _P_ALV_O2_MIN, _P_ALV_O2_MAX = 20.0, 600.0
    _P_ALV_CO2_MIN, _P_ALV_CO2_MAX = 5.0, 100.0
    _HB_MIN, _HB_MAX = 5.0, 25.0
    _P50_MIN, _P50_MAX = 10.0, 60.0
    _N_HILL_MIN, _N_HILL_MAX = 1.0, 5.0
    _ALPHA_O2_MIN, _ALPHA_O2_MAX = 1e-4, 0.02
    _C_CO2_OFFSET_MIN, _C_CO2_OFFSET_MAX = 0.0, 1.0
    _K_CO2_SLOPE_MIN, _K_CO2_SLOPE_MAX = 1e-4, 0.05

    def __init__(self,
                # ---- Альвеолярный газ (фиксирован в MVP) ----
                P_alv_O2=100.0,           # мм рт. ст.
                P_alv_CO2=40.0,           # мм рт. ст.
                # ---- Гемоглобин и кривая Хилла ----
                Hb=15.0,                  # г/дл
                P50=26.8,                 # мм рт. ст.
                n_hill=2.7,
                alpha_O2=0.003,           # мл O2 / (дл · мм рт. ст.) — растворимость
                # ---- CO2 (линейная аппроксимация) ----
                C_CO2_offset=0.22,        # мл/мл при P_CO2 = 0
                k_CO2_slope=0.0065,       # мл/мл на мм рт. ст.
                ):

        # =================================================================
        # Валидация конфигурации — fail-fast при инициализации.
        # Параметры приходят из YAML и не меняются во время симуляции;
        # ошибки в них должны ловиться один раз, а не в горячем пути RHS.
        # =================================================================
        def _check_range(name, v, lo, hi, typical=""):
            v = float(v)
            if not np.isfinite(v) or not (lo <= v <= hi):
                raise ValueError(
                    f"GasExchange: {name}={v} вне [{lo}, {hi}]. {typical}"
                )
            return v

        # --- Альвеолярный газ ---
        self.P_alv_O2 = _check_range(
            "P_alv_O2", P_alv_O2,
            self._P_ALV_O2_MIN, self._P_ALV_O2_MAX,
            "мм рт.ст., типично 100."
        )
        self.P_alv_CO2 = _check_range(
            "P_alv_CO2", P_alv_CO2,
            self._P_ALV_CO2_MIN, self._P_ALV_CO2_MAX,
            "мм рт.ст., типично 40."
        )

        # --- Гемоглобин ---
        self.Hb = _check_range(
            "Hb", Hb, self._HB_MIN, self._HB_MAX,
            "г/дл, типично 15."
        )

        # --- Кривая Хилла ---
        self.P50 = _check_range(
            "P50", P50, self._P50_MIN, self._P50_MAX,
            "мм рт.ст., типично 26.8."
        )
        self.n_hill = _check_range(
            "n_hill", n_hill, self._N_HILL_MIN, self._N_HILL_MAX,
            "безразмерный, типично 2.7 (коэффициент Хилла)."
        )
        self.alpha_O2 = _check_range(
            "alpha_O2", alpha_O2,
            self._ALPHA_O2_MIN, self._ALPHA_O2_MAX,
            "мл O2/(дл·мм рт.ст.), типично 0.003."
        )

        # --- CO2 (линейная аппроксимация) ---
        self.C_CO2_offset = _check_range(
            "C_CO2_offset", C_CO2_offset,
            self._C_CO2_OFFSET_MIN, self._C_CO2_OFFSET_MAX,
            "мл/мл при P_CO2 = 0, типично 0.22."
        )
        self.k_CO2_slope = _check_range(
            "k_CO2_slope", k_CO2_slope,
            self._K_CO2_SLOPE_MIN, self._K_CO2_SLOPE_MAX,
            "мл/мл на мм рт.ст., типично 0.0065."
        )

        # --- Кэш выходов ---
        self._current_outputs = {}

    # =================================================================
    # Обязательный интерфейс OrganModel
    # =================================================================

    def get_state_size(self) -> int:
        return 0

    def get_initial_state(self) -> np.ndarray:
        return np.array([])

    def get_derivatives(self, t, state, inputs) -> np.ndarray:
        return np.array([])

    def get_outputs(self, state) -> dict:
        return self._current_outputs.copy()

    # =================================================================
    # Кривые диссоциации: прямые и обратные
    # =================================================================

    def _SaO2_from_P(self, P_O2: float) -> float:
        """Кривая Хилла: P_O2 -> SaO2 ∈ [0, 1]."""
        P = max(float(P_O2), 0.0)
        Pn = P ** self.n_hill
        P50n = self.P50 ** self.n_hill
        return float(Pn / (P50n + Pn + 1e-12))

    def _C_O2_from_P(self, P_O2: float) -> float:
        """
        Содержание O2 (мл/мл) из P_O2 по формуле Северингауза:
            C = 1.34 · Hb · SaO2 + 0.003 · P_O2
        Результат в мл/мл (исходная формула даёт мл/дл).
        """
        SaO2 = self._SaO2_from_P(P_O2)
        C_per_dl = 1.34 * self.Hb * SaO2 + self.alpha_O2 * float(P_O2)
        return float(C_per_dl / 100.0)

    def _P_O2_from_C(self, C_O2: float) -> float:
        """
        Обратная кривая Хилла: C_O2 (мл/мл) -> P_O2 (мм рт. ст.).

        Учитывает ОБА слагаемых в C = 1.34·Hb·SaO2/100 + α·P/100:
        аналитически не решается, используем Newton.

        Старая версия игнорировала растворённый O2, что давало
        P=100 → C=0.199 → P_back=134 (ошибка 34%).
        """
        C_max = 1.34 * self.Hb / 100.0        # макс. связанный O2, мл/мл
        # Верхняя граница = bound + растворённый O2 (при разумном P_max=500)
        C_max_eff = C_max + self.alpha_O2 * 500.0 / 100.0
        C = float(np.clip(C_O2, 1e-6, 0.999 * C_max_eff))

        P50n = self.P50 ** self.n_hill
        n = self.n_hill
        alpha_per_dl = self.alpha_O2          # мл/(дл·мм рт. ст.)

        # Начальное приближение — из старой формулы (без растворённого O2)
        Sa0 = float(np.clip(C / C_max, 1e-6, 1.0 - 1e-6))
        P = self.P50 * (Sa0 / (1.0 - Sa0)) ** (1.0 / n)
        P = max(P, 1.0)

        for _ in range(20):
            Pn = P ** n
            SaO2 = Pn / (P50n + Pn + 1e-12)
            C_calc = (1.34 * self.Hb * SaO2 + alpha_per_dl * P) / 100.0
            # dC/dP
            dSa_dP = n * P50n * (P ** (n - 1)) / (P50n + Pn) ** 2
            dC_dP = (1.34 * self.Hb * dSa_dP + alpha_per_dl) / 100.0
            dP = (C_calc - C) / max(dC_dP, 1e-12)
            P = P - dP
            if abs(dP) < 1e-6 * max(P, 1.0):
                break
            P = max(P, 1e-6)

        return float(P)

    def _C_CO2_from_P(self, P_CO2: float) -> float:
        """Линейная кривая CO2: C = offset + slope · P."""
        return float(self.C_CO2_offset + self.k_CO2_slope * max(float(P_CO2), 0.0))

    def _P_CO2_from_C(self, C_CO2: float) -> float:
        """Обратная линейная кривая CO2."""
        return float(max((C_CO2 - self.C_CO2_offset) / self.k_CO2_slope, 0.0))

    # =================================================================
    # Основной метод
    # =================================================================

    def compute_effects(self,
                        C_v_O2: float,
                        C_v_CO2: float,
                        Q_p: float,
                        Q_shunt: float) -> dict:
        """
        Параметры
        ---------
        C_v_O2  : смешанная венозная концентрация O2 (мл/мл)
        C_v_CO2 : смешанная венозная концентрация CO2 (мл/мл)
        Q_p     : лёгочный кровоток (мл/с), всегда > 0
        Q_shunt : поток через ДМЖП (мл/с),
                  > 0 — лево-правый (L→R), < 0 — право-левый (R→L)

        Возвращает
        ----------
        Словарь с концентрациями (C_a_O2, C_v_O2, C_pv_O2, C_a_CO2, C_v_CO2, C_pv_CO2),
        парциальными давлениями (P_a_O2, P_v_O2, P_v_CO2) и диагностикой
        (SaO2, shunt_fraction_R2L, Qp_Qs, O2_uptake, CO2_removal, f_bypass).
        """
        # --- Защита от некорректных входов (runtime-клипы, не конфиг) ---
        C_v_O2  = float(np.clip(C_v_O2,  0.001, 0.25))
        C_v_CO2 = float(np.clip(C_v_CO2, 0.05,  1.00))
        Q_p     = max(float(Q_p), 1e-6)

        # === 1. Системный кровоток (единая формула для обоих направлений) ===
        # Из баланса heart.py: Q_aortic = Q_pulmonary - Q_vsd
        Q_s = max(Q_p - Q_shunt, 1e-6)

        # === 2. Насыщение в конце лёгочного капилляра (равновесие с альвеолой) ===
        C_pv_O2  = self._C_O2_from_P(self.P_alv_O2)
        C_pv_CO2 = self._C_CO2_from_P(self.P_alv_CO2)

        # === 3. Смешивание при право-левом шунте ===
        if Q_shunt < 0:
            # Часть венозной крови из ПЖ идёт напрямую в аорту, минуя лёгкие
            Q_bypass = abs(Q_shunt)
            f_bypass = float(np.clip(Q_bypass / Q_s, 0.0, 0.95))
            C_a_O2  = (1.0 - f_bypass) * C_pv_O2  + f_bypass * C_v_O2
            C_a_CO2 = (1.0 - f_bypass) * C_pv_CO2 + f_bypass * C_v_CO2
        else:
            # Лево-правый шунт не снижает сатурацию системной крови
            f_bypass = 0.0
            C_a_O2  = C_pv_O2
            C_a_CO2 = C_pv_CO2

        # === 6. Парциальные давления (для диагностики) ===
        P_v_O2  = self._P_O2_from_C(C_v_O2)
        P_v_CO2 = self._P_CO2_from_C(C_v_CO2)

        # === 7. Диагностические выходы ===
        # При отсутствии R→L шунта C_a_O2 = C_pv_O2 → P_a_O2 = P_alv_O2
        # (равновесие с альвеолой, а не численная инверсия).
        # При наличии R→L — решаем обратную задачу из смешанного C_a_O2.
        if Q_shunt >= 0:
            P_a_O2 = float(self.P_alv_O2)
        else:
            P_a_O2 = self._P_O2_from_C(C_a_O2)
        SaO2 = self._SaO2_from_P(P_a_O2)

        # Доля право-левого шунта в системном выбросе
        shunt_fraction_R2L = max(-Q_shunt, 0.0) / Q_s
        # Отношение лёгочного кровотока к системному
        Qp_Qs = Q_p / Q_s

        # Интегральные показатели газообмена (мл газа / с)
        O2_uptake   = Q_p * max(C_pv_O2  - C_v_O2,  0.0)
        CO2_removal = Q_p * max(C_v_CO2 - C_pv_CO2, 0.0)

        self._current_outputs = {
            # --- Концентрации (мл/мл) ---
            'C_a_O2':   float(C_a_O2),
            'C_v_O2':   float(C_v_O2),
            'C_pv_O2':  float(C_pv_O2),
            'C_a_CO2':  float(C_a_CO2),
            'C_v_CO2':  float(C_v_CO2),
            'C_pv_CO2': float(C_pv_CO2),
            # --- Парциальные давления (мм рт. ст.) ---
            'P_v_O2':   float(P_v_O2),
            'P_v_CO2':  float(P_v_CO2),
            'P_a_O2':   float(P_a_O2),
            'P_alv_O2': float(self.P_alv_O2),
            'P_alv_CO2': float(self.P_alv_CO2),
            # --- Диагностика ---
            'SaO2':               float(SaO2),           # сатурация артериальной крови
            'oxygenation_index':  float(SaO2),           # алиас для обратной совместимости
            'shunt_fraction_R2L': float(shunt_fraction_R2L),
            'Qp_Qs':              float(Qp_Qs),
            'O2_uptake':          float(O2_uptake),
            'CO2_removal':        float(CO2_removal),
            'f_bypass':           float(f_bypass),
        }
        return self._current_outputs


# =====================================================================
# Быстрый тест (python gas_exchange.py)
# =====================================================================
if __name__ == "__main__":
    gas = GasExchange()

    print("=" * 70)
    print("Тест 1: Здоровый (Q_shunt = 0)")
    print("=" * 70)
    out = gas.compute_effects(C_v_O2=0.15, C_v_CO2=0.52,
                              Q_p=83.0, Q_shunt=0.0)
    for k in ('C_a_O2', 'C_v_O2', 'SaO2', 'Qp_Qs', 'shunt_fraction_R2L'):
        print(f"  {k:22s} = {out[k]:+.6g}")

    print("\n" + "=" * 70)
    print("Тест 2: Тяжёлый Эйзенменгер (Q_shunt = −60)")
    print("=" * 70)
    out = gas.compute_effects(C_v_O2=0.15, C_v_CO2=0.52,
                              Q_p=80.0, Q_shunt=-60.0)
    for k in ('C_a_O2', 'SaO2', 'P_a_O2', 'shunt_fraction_R2L'):
        print(f"  {k:22s} = {out[k]:+.6g}")