# lungs.py
import numpy as np
from organ_base import OrganModel

class Lungs2Chamber(OrganModel):
    """
    Модель лёгких здорового человека.
    Фиксированные сопротивления, минимальный шунт.
    """
    def __init__(self,
                 R1=0.5, R2=0.5,
                 C1=2.0, C2=3.0,
                 shunt_fraction=0.02):
        self.R1 = R1
        self.R2 = R2
        self.C1 = C1
        self.C2 = C2
        self.shunt_fraction = shunt_fraction
        self._current_outputs = {}

    def get_state_size(self):
        return 2

    def get_initial_state(self):
        return np.array([15.0, 10.0])

    def get_derivatives(self, t, state, inputs):
        P_prox, P_dist = state
        Q_pulm = inputs.get('Q_pulmonary', 0.0)
        P_pv = inputs.get('P_pv', 5.0)

        dP_prox = (Q_pulm - (P_prox - P_dist) / self.R1) / self.C1
        dP_dist = ((P_prox - P_dist) / self.R1 - (P_dist - P_pv) / self.R2) / self.C2

        oxygenation_index = 1.0 - self.shunt_fraction

        self._current_outputs = {
            'P_pa': P_prox,
            'oxygenation_index': oxygenation_index,
            'shunt_fraction': self.shunt_fraction
        }
        return np.array([dP_prox, dP_dist])

    def get_outputs(self, state):
        return self._current_outputs.copy()