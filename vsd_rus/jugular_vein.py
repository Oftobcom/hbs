# jugular_vein.py — отдельный компартмент яремной вены
"""
Яремная вена как буфер между мозгом и системными венами.

Состояние: [V_jv, C_jv_O2, C_jv_CO2]
  V_jv — объем яремной вены (мл)
  C_jv_O2 — O2 концентрация в яремной вене (мл/мл), SjvO2 ~ 60-75%
  C_jv_CO2 — CO2 концентрация (мл/мл)

Гемодинамика:
  Q_in = Q_brain (из мозга)
  Q_out = (P_jv - P_sv)/R_out -> в системные вены / ПП
  dV/dt = Q_in - Q_out + (V_target - V)/tau

Газовый баланс (полное перемешивание):
  d(C_jv_O2)/dt = Q_in*(C_v_brain_O2 - C_jv_O2)/V_jv
  d(C_jv_CO2)/dt = Q_in*(C_v_brain_CO2 - C_jv_CO2)/V_jv

Диагностика: SjvO2 = C_jv_O2 / C_a_O2_max, P_jv, Q_jv_out
"""

import numpy as np
from organ_base import OrganModel

class JugularVein(OrganModel):
    def __init__(self,
                 C=20.0,           # мл/мм рт.ст., комплаенс яремной вены (высокий)
                 P0=6.0,           # мм рт.ст., базовое давление
                 V0=150.0,         # мл, объем при P0
                 R_out=0.5,        # мм рт.ст.*с/мл, сопротивление оттока в системные вены
                 target_fraction=0.05, # доля V_blood, которую стремится занять яремная вена
                 tau_target=200.0, # с, как у sys_ven
                 C_O2_init=0.12,   # мл/мл, начальная O2 (венозная мозга)
                 C_CO2_init=0.56,  # мл/мл, начальная CO2
                 Hb=15.0):         # для расчета сатурации
        self.C = float(C)
        self.P0 = float(P0)
        self.V0 = float(V0)
        self.R_out = float(R_out)
        self.target_fraction = float(target_fraction)
        self.tau_target = float(tau_target)
        self.C_O2_init = float(C_O2_init)
        self.C_CO2_init = float(C_CO2_init)
        self.Hb = float(Hb)
        self.C_max_O2 = 1.34 * self.Hb / 100.0  # макс связанный O2, мл/мл ~0.201
        self._current_outputs = {}

    def get_state_size(self):
        return 3  # V, C_O2, C_CO2

    def get_initial_state(self):
        return np.array([self.V0, self.C_O2_init, self.C_CO2_init])

    def get_derivatives(self, t, state, inputs):
        V_jv, C_jv_O2, C_jv_CO2 = state
        Q_in = float(inputs.get('Q_in', 0.0))  # Q_brain
        C_in_O2 = float(inputs.get('C_in_O2', self.C_O2_init))  # C_v_brain
        C_in_CO2 = float(inputs.get('C_in_CO2', self.C_CO2_init))
        P_sv = float(inputs.get('P_sv', 5.0))
        V_blood = inputs.get('V_blood', None)

        # Давление из объема
        P_jv = self.P0 + (V_jv - self.V0) / self.C
        P_jv = max(P_jv, 0.0)

        # Отток в системные вены
        Q_out = (P_jv - P_sv) / self.R_out
        Q_out = max(Q_out, 0.0)

        # Объем с масс-балансом к целевой доле V_blood
        dV = Q_in - Q_out
        if self.target_fraction is not None and V_blood is not None:
            V_target = self.target_fraction * float(V_blood)
            dV += (V_target - V_jv) / self.tau_target

        # Мягкий пол как у sys_ven
        if V_jv < 0.5 * self.V0 and dV < 0:
            softness = (V_jv - 0.5 * self.V0) / (0.5 * self.V0)
            softness = float(np.clip(softness, 0.0, 1.0))
            dV *= softness

        # Газовый баланс — полное перемешивание
        V_safe = max(V_jv, 1.0)  # защита от деления на 0
        dC_O2 = Q_in * (C_in_O2 - C_jv_O2) / V_safe
        dC_CO2 = Q_in * (C_in_CO2 - C_jv_CO2) / V_safe

        # Диагностика
        SjvO2 = float(np.clip(C_jv_O2 / max(self.C_max_O2, 1e-6), 0.0, 1.0))
        # Парциальное давление O2 из C_jv (обратная Хилла упрощенная)
        # Используем ту же формулу что в gas_exchange для диагностики
        P50 = 26.8
        n = 2.7
        Sa = np.clip(C_jv_O2 / max(self.C_max_O2, 1e-6), 1e-6, 0.999)
        P_jv_O2 = P50 * (Sa / (1.0 - Sa)) ** (1.0 / n) if Sa < 0.999 else 100.0

        self._current_outputs = {
            'P_jv': float(P_jv),
            'V_jv': float(V_jv),
            'V': float(V_jv),  # алиас для совместимости с Windkessel
            'C_jv_O2': float(C_jv_O2),
            'C_jv_CO2': float(C_jv_CO2),
            'SjvO2': float(SjvO2),
            'P_jv_O2': float(P_jv_O2),
            'Q_jv_out': float(Q_out),
            'Q_in': float(Q_in),
            'C_in_O2': float(C_in_O2),
        }
        return np.array([dV, dC_O2, dC_CO2])

    def get_outputs(self, state):
        return self._current_outputs.copy()


if __name__ == "__main__":
    jv = JugularVein()
    print(f"State size {jv.get_state_size()}, init {jv.get_initial_state()}")
    # тест: мозг дает Q=5, C_v=0.10
    y = jv.get_initial_state()
    for i in range(10):
        dy = jv.get_derivatives(0, y, {'Q_in':5.0, 'C_in_O2':0.10, 'C_in_CO2':0.56, 'P_sv':5.0, 'V_blood':5800})
        y = y + dy*0.1
    print("After 1s", jv.get_outputs(y))
