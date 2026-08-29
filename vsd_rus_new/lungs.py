# lungs.py
import numpy as np
from organ_base import OrganModel

class Lungs2Chamber(OrganModel):
    """
    Модель лёгких.
    Для ДМЖП можно увеличивать сопротивления при длительном повышении кровотока.
    """
    def __init__(self,
                 R1=0.5, R2=0.5,
                 C1=2.0, C2=3.0,
                 shunt_fraction=0.02,
                 flow_dependent_resistance=False,
                 flow_sensitivity=0.05):
        self.R1_base = R1
        self.R2_base = R2
        self.C1 = C1
        self.C2 = C2
        self.shunt_fraction = shunt_fraction
        self.flow_dependent_resistance = flow_dependent_resistance
        self.flow_sensitivity = flow_sensitivity   # увеличение сопротивления на 1% на каждые 10% превышения нормы
        self._current_outputs = {}

    def get_state_size(self):
        return 2

    def get_initial_state(self):
        return np.array([15.0, 10.0])

    def _effective_resistances(self, Q_pulm):
        if not self.flow_dependent_resistance:
            return self.R1_base, self.R2_base
        # Нормальный лёгочный кровоток ~ 80 мл/с (около 5 л/мин)
        Q_norm = 80.0
        if Q_pulm <= Q_norm:
            factor = 1.0
        else:
            excess = (Q_pulm - Q_norm) / Q_norm
            factor = 1.0 + self.flow_sensitivity * excess
            factor = min(factor, 3.0)   # не более чем в 3 раза выше
        return self.R1_base * factor, self.R2_base * factor

    def get_derivatives(self, t, state, inputs):
        P_prox, P_dist = state
        Q_pulm = inputs.get('Q_pulmonary', 0.0)
        P_pv = inputs.get('P_pv', 5.0)

        R1_eff, R2_eff = self._effective_resistances(Q_pulm)

        dP_prox = (Q_pulm - (P_prox - P_dist) / R1_eff) / self.C1
        dP_dist = ((P_prox - P_dist) / R1_eff - (P_dist - P_pv) / R2_eff) / self.C2

        oxygenation_index = 1.0 - self.shunt_fraction

        self._current_outputs = {
            'P_pa': P_prox,
            'oxygenation_index': oxygenation_index,
            'shunt_fraction': self.shunt_fraction,
            'R1_eff': R1_eff,
            'R2_eff': R2_eff
        }
        return np.array([dP_prox, dP_dist])

    def get_outputs(self, state):
        return self._current_outputs.copy()