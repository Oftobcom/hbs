# gas_exchange.py
"""
Альвеолярно-капиллярный газообмен O2/CO2.

Алгебраический орган без состояния (по аналогии с KidneyHemodynamic).
Возвращает производные смешанных венозных концентраций O2 и CO2,
которые передаются в BloodPool через dC_blood_arr в whole_body.py.

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
    Эффекты на BloodPool: dC_O2, dC_CO2 — производные смешанной венозной
    концентрации (мл газа / мл крови / с).
    """

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
                 # ---- Метаболизм (весь организм) ----
                 VO2_base=4.2,             # мл O2 / с   (≈ 250 мл/мин)
                 VCO2_base=3.3,            # мл CO2 / с  (≈ 200 мл/мин)
                 Q_norm=83.0):             # мл/с — нормальный системный кровоток
        # Альвеолярный газ
        self.P_alv_O2 = float(P_alv_O2)
        self.P_alv_CO2 = float(P_alv_CO2)
        # Гемоглобин
        self.Hb = float(Hb)
        self.P50 = float(P50)
        self.n_hill = float(n_hill)
        self.alpha_O2 = float(alpha_O2)
        # CO2
        self.C_CO2_offset = float(C_CO2_offset)
        self.k_CO2_slope = float(k_CO2_slope)
        # Метаболизм
        self.VO2_base = float(VO2_base)
        self.VCO2_base = float(VCO2_base)
        self.Q_norm = float(Q_norm)
        # Кэш выходов
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
        """Обратная кривая Хилла: C_O2 (мл/мл) -> P_O2 (мм рт. ст.)."""
        C_max = 1.34 * self.Hb / 100.0        # макс. связанный O2, мл/мл
        C = float(np.clip(C_O2, 1e-4, 0.999 * C_max))
        SaO2 = C / C_max
        SaO2 = float(np.clip(SaO2, 1e-6, 1.0 - 1e-6))
        return float(self.P50 * (SaO2 / (1.0 - SaO2)) ** (1.0 / self.n_hill))

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
                        Q_shunt: float,
                        V_blood: float) -> dict:
        """
        Параметры
        ---------
        C_v_O2  : смешанная венозная концентрация O2 (мл/мл)
        C_v_CO2 : смешанная венозная концентрация CO2 (мл/мл)
        Q_p     : лёгочный кровоток (мл/с), всегда > 0
        Q_shunt : поток через ДМЖП (мл/с),
                  > 0 — лево-правый (L→R), < 0 — право-левый (R→L)
        V_blood : объём крови (мл)

        Возвращает
        ----------
        Словарь с dC_O2, dC_CO2 и диагностическими полями.
        """
        # --- Защита от некорректных входов ---
        C_v_O2  = float(np.clip(C_v_O2,  0.001, 0.25))
        C_v_CO2 = float(np.clip(C_v_CO2, 0.05,  1.00))
        Q_p     = max(float(Q_p), 1e-6)
        V_safe  = max(float(V_blood), 1e-6)

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

        # === 4. Потребление O2 и продукция CO2, масштабированные по Q_s ===
        Q_factor = float(np.clip(Q_s / self.Q_norm, 0.0, 1.0))
        VO2_eff  = self.VO2_base  * Q_factor
        VCO2_eff = self.VCO2_base * Q_factor

        # === 5. Баланс смешанного венозного резервуара ===
        # dC_v/dt = [Q_s · (C_a − C_v) ∓ метаболизм] / V_blood
        dC_O2  = (Q_s * (C_a_O2  - C_v_O2)  - VO2_eff)  / V_safe
        dC_CO2 = (Q_s * (C_a_CO2 - C_v_CO2) + VCO2_eff) / V_safe

        # === 6. Парциальные давления (для диагностики) ===
        P_v_O2  = self._P_O2_from_C(C_v_O2)
        P_v_CO2 = self._P_CO2_from_C(C_v_CO2)

        # === 7. Диагностические выходы ===
        # Артериальная сатурация — из C_a_O2 через обратную кривую Хилла
        P_a_O2 = self._P_O2_from_C(C_a_O2)
        SaO2   = self._SaO2_from_P(P_a_O2)

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
            # --- Производные для BloodPool (мл/мл/с) ---
            'dC_O2':    float(dC_O2),
            'dC_CO2':   float(dC_CO2),
            # --- Диагностика ---
            'SaO2':               float(SaO2),           # сатурация артериальной крови
            'oxygenation_index':  float(SaO2),           # алиас для обратной совместимости
            'shunt_fraction_R2L': float(shunt_fraction_R2L),
            'Qp_Qs':              float(Qp_Qs),
            'O2_uptake':          float(O2_uptake),
            'CO2_removal':        float(CO2_removal),
            'VO2_eff':            float(VO2_eff),
            'VCO2_eff':           float(VCO2_eff),
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
                              Q_p=83.0, Q_shunt=0.0, V_blood=5000.0)
    for k in ('C_a_O2', 'C_v_O2', 'SaO2', 'dC_O2', 'Qp_Qs', 'shunt_fraction_R2L'):
        print(f"  {k:22s} = {out[k]:+.6g}")

    print("\n" + "=" * 70)
    print("Тест 2: Большой L→R шунт (Q_shunt = +50)")
    print("=" * 70)
    out = gas.compute_effects(C_v_O2=0.15, C_v_CO2=0.52,
                              Q_p=133.0, Q_shunt=+50.0, V_blood=5000.0)
    for k in ('C_a_O2', 'SaO2', 'Qp_Qs', 'shunt_fraction_R2L'):
        print(f"  {k:22s} = {out[k]:+.6g}")

    print("\n" + "=" * 70)
    print("Тест 3: Право-левый шунт (Q_shunt = −30)")
    print("=" * 70)
    out = gas.compute_effects(C_v_O2=0.15, C_v_CO2=0.52,
                              Q_p=80.0, Q_shunt=-30.0, V_blood=5000.0)
    for k in ('C_a_O2', 'SaO2', 'P_a_O2', 'Qp_Qs', 'shunt_fraction_R2L', 'f_bypass'):
        print(f"  {k:22s} = {out[k]:+.6g}")

    print("\n" + "=" * 70)
    print("Тест 4: Тяжёлый Эйзенменгер (Q_shunt = −60)")
    print("=" * 70)
    out = gas.compute_effects(C_v_O2=0.15, C_v_CO2=0.52,
                              Q_p=80.0, Q_shunt=-60.0, V_blood=5000.0)
    for k in ('C_a_O2', 'SaO2', 'P_a_O2', 'shunt_fraction_R2L'):
        print(f"  {k:22s} = {out[k]:+.6g}")