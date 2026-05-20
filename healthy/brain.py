# brain.py
import numpy as np
from organ_base import OrganModel

class Brain(OrganModel):
    """
    Healthy brain model – cerebral hemodynamics and metabolism.
    No pathological inhibition.
    """
    def __init__(self,
                 R_base=0.8,
                 C=2.0,
                 autoreg_gain=0.02,
                 P_autoreg=80.0,
                 O2_extraction=0.4,
                 glucose_extraction=0.1,
                 P0=60.0):
        self.R_base = R_base
        self.C = C
        self.autoreg_gain = autoreg_gain
        self.P_autoreg = P_autoreg
        self.O2_extraction = O2_extraction
        self.glucose_extraction = glucose_extraction
        self.P0 = P0
        self._current_outputs = {}

    def get_state_size(self):
        return 1

    def get_initial_state(self):
        return np.array([self.P0])

    def _autoregulation_resistance(self, P_sa):
        delta = P_sa - self.P_autoreg
        reg = 1.0 - self.autoreg_gain * delta
        reg = max(0.5, min(reg, 2.0))
        return self.R_base * reg

    def get_derivatives(self, t, state, inputs):
        P_br = state[0]
        P_sa = inputs.get('P_sa', 80.0)
        P_sv = inputs.get('P_sv', 5.0)

        R_eff = self._autoregulation_resistance(P_sa)
        Q_br = (P_sa - P_br) / R_eff
        Q_out = (P_br - P_sv) / R_eff
        dP_br = (Q_br - Q_out) / self.C

        # Healthy metabolism – no inhibition
        O2_consumption = Q_br * self.O2_extraction
        glucose_consumption = Q_br * self.glucose_extraction

        self._current_outputs = {
            'Q_br': Q_br,
            'O2_consumption': O2_consumption,
            'glucose_consumption': glucose_consumption,
            'metabolic_inhibition': 1.0
        }
        return np.array([dP_br])

    def get_outputs(self, state):
        return self._current_outputs.copy()