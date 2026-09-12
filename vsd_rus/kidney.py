# kidney.py
import numpy as np
from organ_base import OrganModel

class KidneyHemodynamic(OrganModel):
    """
    Модель почек здорового человека.
    Авторегуляция СКФ с сигмоидальной зависимостью (более физиологичная).
    """
    def __init__(self,
                 GFR_base=120.0,
                 P_autoreg=95.0,
                 autoreg_amplitude=0.6,       # амплитуда изменения СКФ (0.4..1.6)
                 autoreg_slope=0.025,         # крутизна сигмоиды
                 toxin_clearance_frac=0.2,
                 volume_reabsorption_frac=0.99,
                 renal_resistance=3.75):
        self.GFR_base = GFR_base
        self.P_autoreg = P_autoreg
        self.autoreg_amplitude = autoreg_amplitude
        self.autoreg_slope = autoreg_slope
        self.toxin_clearance_frac = toxin_clearance_frac
        self.volume_reabsorption_frac = volume_reabsorption_frac
        self.renal_resistance = renal_resistance
        self._current_outputs = {}

    def get_state_size(self):
        return 0

    def get_initial_state(self):
        return np.array([])

    def get_derivatives(self, t, state, inputs):
        return np.array([])

    def get_outputs(self, state):
        return self._current_outputs.copy()

    def _compute_gfr(self, P_art):
        # Низкое давление - линейный рост с плавным включением
        low = P_art / 80.0  # 0 -> 0, 80 -> 1
        # Высокое давление - сигмоида как в варианте 1
        delta = P_art - self.P_autoreg
        high = 1.0 + self.autoreg_amplitude * np.tanh(self.autoreg_slope * delta)
        # Склейка через логистику в районе 80
        w = 0.5 * (1 + np.tanh((P_art - 80)/5))  # 0 при P<<80, 1 при P>>80
        factor = (1-w)*low + w*high
        factor = np.clip(factor, 0.0, 2.2)  # разрешаем 0 - анурию
        return self.GFR_base * factor

    def compute_effects(self, P_sa, P_sv, C_tox, V_blood):
        """
        Вычисляет эффекты почек на гемодинамику и клиренс токсина.
        """
        R_eff = self.renal_resistance
        Q_renal = (P_sa - P_sv) / R_eff
        GFR = self._compute_gfr(P_sa) / 60.0   # мл/мин -> мл/с

        toxin_filtered = GFR * C_tox
        toxin_excreted = self.toxin_clearance_frac * toxin_filtered
        dC_tox = -toxin_excreted / max(V_blood, 1e-6)

        urine_output = GFR * (1 - self.volume_reabsorption_frac)
        dV_blood = -urine_output

        self._current_outputs = {'Q_renal': Q_renal, 'GFR': GFR, 'urine_output': urine_output}
        return {
            'dC_tox': dC_tox,
            'dV_blood': dV_blood,
            'Q_renal': Q_renal,
            'GFR': GFR,
            'urine_output': urine_output
        }