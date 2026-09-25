# jugular_vein.py — отдельный компартмент яремной вены
"""
Яремная вена как буфер между мозгом и системными венами.

Состояние: [V_jv, C_jv_O2, C_jv_CO2]
  V_jv     — объем яремной вены (мл)
  C_jv_O2  — O2 концентрация в яремной вене (мл/мл), SjvO2 ~ 60-75%
  C_jv_CO2 — CO2 концентрация (мл/мл)

Гемодинамика:
  Q_in  = Q_brain (из мозга)
  Q_out = (P_jv - P_sv)/R_out   → в системные вены / ПП
  dV/dt = Q_in - Q_out + (V_target - V)/tau

Газовый баланс (полное перемешивание):
  d(C_jv_O2)/dt  = Q_in · (C_v_brain_O2  - C_jv_O2 ) / V_jv
  d(C_jv_CO2)/dt = Q_in · (C_v_brain_CO2 - C_jv_CO2) / V_jv

Диагностика: SjvO2 = C_jv_O2 / C_max_O2, P_jv, Q_jv_out, P_jv_O2

Единицы:
  P        — мм рт.ст.
  Q        — мл/с
  V        — мл
  R        — мм рт.ст.·с/мл
  C, C_max — мл O2 / мл крови
  Hb       — г/дл
"""

import numpy as np
from organ_base import OrganModel


class JugularVein(OrganModel):
    """
    Модель яремной вены как буфера между мозгом и системными венами.
    """

    # --- Санити-пороги для валидации конфигурации ---
    _C_MIN, _C_MAX = 1.0, 200.0
    _P0_MIN, _P0_MAX = 0.0, 30.0
    _V0_MIN, _V0_MAX = 10.0, 2000.0
    _R_OUT_MIN, _R_OUT_MAX = 1e-3, 20.0
    _TAU_TARGET_MIN, _TAU_TARGET_MAX = 1.0, 1e5

    _C_O2_INIT_MIN, _C_O2_INIT_MAX = 0.0, 0.25
    _C_CO2_INIT_MIN, _C_CO2_INIT_MAX = 0.0, 1.0
    _HB_MIN, _HB_MAX = 5.0, 25.0

    def __init__(self,
                 C=20.0,                  # мл/мм рт.ст., комплаенс яремной вены
                 P0=6.0,                  # мм рт.ст., базовое давление
                 V0=150.0,                # мл, объем при P0
                 R_out=0.5,               # мм рт.ст.·с/мл, сопротивление оттока
                 target_fraction=0.05,    # доля V_blood, которую стремится занять V_jv
                 tau_target=200.0,        # с, время релаксации к V_target
                 C_O2_init=0.12,          # мл/мл, начальная O2 (венозная мозга)
                 C_CO2_init=0.56,         # мл/мл, начальная CO2
                 Hb=15.0):                # г/дл, для расчета сатурации

        # =================================================================
        # Валидация конфигурации — fail-fast при инициализации.
        # Параметры приходят из YAML и не меняются во время симуляции;
        # ошибки в них должны ловиться один раз, а не в горячем пути RHS.
        # =================================================================
        def _check_range(name, v, lo, hi, typical=""):
            v = float(v)
            if not np.isfinite(v) or not (lo <= v <= hi):
                raise ValueError(
                    f"JugularVein: {name}={v} вне [{lo}, {hi}]. {typical}"
                )
            return v

        # --- Гемодинамика ---
        self.C = _check_range(
            "C", C, self._C_MIN, self._C_MAX,
            "мл/мм рт.ст., типично 20 (высокий комплаенс вены)."
        )
        self.P0 = _check_range(
            "P0", P0, self._P0_MIN, self._P0_MAX,
            "мм рт.ст., типично 6."
        )
        self.V0 = _check_range(
            "V0", V0, self._V0_MIN, self._V0_MAX,
            "мл, типично 150–290 (≈5% от V_blood)."
        )
        self.R_out = _check_range(
            "R_out", R_out, self._R_OUT_MIN, self._R_OUT_MAX,
            "мм рт.ст.·с/мл, типично 0.5."
        )

        # --- Масс-баланс к целевой доле V_blood ---
        tf = float(target_fraction)
        if not np.isfinite(tf) or not (0.0 < tf < 1.0):
            raise ValueError(
                f"JugularVein: target_fraction={target_fraction} вне (0, 1). "
                f"Типично 0.05 (5% V_blood)."
            )
        self.target_fraction = tf

        self.tau_target = _check_range(
            "tau_target", tau_target,
            self._TAU_TARGET_MIN, self._TAU_TARGET_MAX,
            "с, типично 200–300."
        )

        # --- Начальные концентрации газов ---
        self.C_O2_init = _check_range(
            "C_O2_init", C_O2_init,
            self._C_O2_INIT_MIN, self._C_O2_INIT_MAX,
            "мл O2/мл, типично 0.12 (SjvO2 ≈ 60%)."
        )
        self.C_CO2_init = _check_range(
            "C_CO2_init", C_CO2_init,
            self._C_CO2_INIT_MIN, self._C_CO2_INIT_MAX,
            "мл CO2/мл, типично 0.56."
        )

        # --- Гемоглобин ---
        self.Hb = _check_range(
            "Hb", Hb, self._HB_MIN, self._HB_MAX,
            "г/дл, типично 15."
        )

        # Максимальная связанная ёмкость O2 (формула Северinгауза)
        self.C_max_O2 = 1.34 * self.Hb / 100.0   # мл/мл, ~0.201 при Hb=15

        self._current_outputs = {}

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
    def get_state_size(self):
        return 3  # [V, C_O2, C_CO2]

    def get_initial_state(self):
        return np.array([self.V0, self.C_O2_init, self.C_CO2_init])

    # ------------------------------------------------------------------
    # Основной метод — производные
    # ------------------------------------------------------------------
    def get_derivatives(self, t, state, inputs):
        V_jv_raw, C_jv_O2_raw, C_jv_CO2_raw = state

        # --- Мягкие клипы состояния (защита от LSODA retries) ---
        V_jv      = max(float(V_jv_raw), 0.0)
        C_jv_O2   = max(float(C_jv_O2_raw), 0.0)
        C_jv_CO2  = max(float(C_jv_CO2_raw), 0.0)

        # --- Входы ---
        Q_in      = float(inputs.get('Q_in', 0.0))
        C_in_O2   = float(inputs.get('C_in_O2', self.C_O2_init))
        C_in_CO2  = float(inputs.get('C_in_CO2', self.C_CO2_init))
        P_sv      = float(inputs.get('P_sv', 5.0))
        V_blood   = inputs.get('V_blood', None)

        # --- Давление из объёма ---
        P_jv = self.P0 + (V_jv - self.V0) / self.C
        P_jv = max(P_jv, 0.0)

        # --- Отток в системные вены ---
        Q_out = (P_jv - P_sv) / self.R_out
        Q_out = max(Q_out, 0.0)

        # --- Масс-баланс: медленная релаксация к целевой доле V_blood ---
        dV = Q_in - Q_out
        if self.target_fraction is not None and V_blood is not None:
            V_target = self.target_fraction * float(V_blood)
            dV += (V_target - V_jv) / self.tau_target

        # --- Мягкий пол: ниже 50% V0 гасим отток ---
        if V_jv < 0.5 * self.V0 and dV < 0:
            softness = (V_jv - 0.5 * self.V0) / (0.5 * self.V0)
            softness = float(np.clip(softness, 0.0, 1.0))
            dV *= softness

        # --- Газовый баланс — полное перемешивание ---
        V_safe = max(V_jv, 1.0)   # защита от деления на 0
        dC_O2  = Q_in * (C_in_O2  - C_jv_O2)  / V_safe
        dC_CO2 = Q_in * (C_in_CO2 - C_jv_CO2) / V_safe

        # --- Диагностика ---
        SjvO2 = float(np.clip(C_jv_O2 / max(self.C_max_O2, 1e-6), 0.0, 1.0))

        # Парциальное давление O2 из C_jv (упрощённая обратная Хилла)
        P50 = 26.8
        n = 2.7
        Sa = float(np.clip(C_jv_O2 / max(self.C_max_O2, 1e-6), 1e-6, 0.999))
        if Sa < 0.999:
            P_jv_O2 = P50 * (Sa / (1.0 - Sa)) ** (1.0 / n)
        else:
            P_jv_O2 = 100.0

        self._current_outputs = {
            'P_jv':       float(P_jv),
            'V_jv':       float(V_jv),
            'V':          float(V_jv),   # алиас для Windkessel-совместимости
            'C_jv_O2':    float(C_jv_O2),
            'C_jv_CO2':   float(C_jv_CO2),
            'SjvO2':      float(SjvO2),
            'P_jv_O2':    float(P_jv_O2),
            'Q_jv_out':   float(Q_out),
            'Q_in':       float(Q_in),
            'C_in_O2':    float(C_in_O2),
        }
        return np.array([dV, dC_O2, dC_CO2])

    def get_outputs(self, state):
        return self._current_outputs.copy()


# =====================================================================
# Быстрый тест (python jugular_vein.py)
# =====================================================================
if __name__ == "__main__":
    jv = JugularVein()
    print(f"State size {jv.get_state_size()}, init {jv.get_initial_state()}")
    # тест: мозг дает Q=5, C_v=0.10
    y = jv.get_initial_state()
    for i in range(10):
        dy = jv.get_derivatives(0, y, {
            'Q_in': 5.0, 'C_in_O2': 0.10, 'C_in_CO2': 0.56,
            'P_sv': 5.0, 'V_blood': 5800,
        })
        y = y + dy * 0.1
    print("After 1s", jv.get_outputs(y))