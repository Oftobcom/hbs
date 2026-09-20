# brain.py
import numpy as np
from organ_base import OrganModel

class Brain(OrganModel):
    """
    Модель головного мозга здорового человека.
    Гемодинамика и метаболизм, без патологического ингибирования.
    """
    def __init__(self,
                R_base=7, C=4.0, autoreg_gain=0.3, P_autoreg=80.0,
                O2_demand_target = 0.4,
                C_v_min=0.09, max_extraction=0.12,
                glucose_extraction=0.1, P0=47.5,
                C_a_O2_norm=0.20,        # нормальная артериальная O2
                k_hypoxic_dilation=0.25): # чувствительность R_eff к гипоксии
        self.R_base = R_base
        self.C = C
        self.autoreg_gain = autoreg_gain
        self.P_autoreg = P_autoreg
        self.O2_demand_target = O2_demand_target # целевая потребность
        self.C_v_min = C_v_min
        self.max_extraction = max_extraction # мозг может повысить экстракцию до 60% при гипоксии
        self.glucose_extraction = glucose_extraction
        self.P0 = P0
        self.C_a_O2_norm = float(C_a_O2_norm)
        self.k_hypoxic_dilation = float(k_hypoxic_dilation)
        self._current_outputs = {}

    def get_state_size(self):
        return 1

    def get_initial_state(self):
        return np.array([self.P0])

    def _autoregulation_resistance(self, P_sa, C_a_O2):
        x = (P_sa - self.P_autoreg) / self.P_autoreg
        reg_P = 1.0 + self.autoreg_gain * np.tanh(x)
        hypoxia = max(self.C_a_O2_norm - C_a_O2, 0.0) / self.C_a_O2_norm
        reg_O2 = 1.0 - self.k_hypoxic_dilation * hypoxia
        reg = np.clip(reg_P * reg_O2, 0.5, 2.0)
        return self.R_base * reg

    def get_derivatives(self, t, state, inputs):
        P_br = state[0]
        P_sa = inputs.get('P_sa', 80.0)
        P_sv = inputs.get('P_sv', 5.0)
        C_a_O2 = float(inputs.get('C_a_O2', self.C_a_O2_norm))

        # --- Гемодинамика: баро + гипоксическая вазодилатация ---
        R_eff = self._autoregulation_resistance(P_sa, C_a_O2)
        Q_br = max((P_sa - P_br) / R_eff, 0.0)
        Q_out = (P_br - P_sv) / R_eff
        dP_br = (Q_br - Q_out) / self.C

        # Метаболическая потребность (фиксированная, в модельном масштабе)
        # Сколько экстракции нужно, чтобы покрыть потребность при текущем Q_br
        if Q_br > 1e-6:
            extraction_needed = self.O2_demand_target / Q_br
        else:
            extraction_needed = self.max_extraction
        # Физический предел
        extraction_avail = max(C_a_O2 - self.C_v_min, 0.0)
        extraction_used  = min(extraction_needed, extraction_avail, self.max_extraction)
        O2_consumption = Q_br * extraction_used
        C_v_O2 = C_a_O2 - extraction_used # для диагностики, всегда >= C_v_min

        glucose_consumption = Q_br * self.glucose_extraction

        self._current_outputs = {
            'Q_br': Q_br,
            'O2_consumption': O2_consumption,
            'C_v_O2': C_v_O2,
            'R_eff': R_eff,
            'extraction_used': extraction_used,
            'extraction_avail': extraction_avail,
            'glucose_consumption': glucose_consumption,
            'metabolic_inhibition': 1.0,
        }
        return np.array([dP_br])

    def get_outputs(self, state):
        return self._current_outputs.copy()