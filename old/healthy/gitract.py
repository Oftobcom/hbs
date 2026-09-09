# gitract.py
import numpy as np
from organ_base import OrganModel

class GITract(OrganModel):
    """
    GI tract – portal hemodynamics and absorption.
    Healthy version: portal pressure sensitivity is mild.
    """
    def __init__(self,
                 R_art=0.3, R_cap=0.2, R_venous=0.2,
                 C_art=2.0, C_cap=5.0,
                 k_absorption_water=0.1,
                 k_absorption_nutrients=0.05,
                 portal_pressure_sensitivity=0.02,
                 P_art0=75.0, P_cap0=10.0):
        self.R_art = R_art
        self.R_cap = R_cap
        self.R_venous = R_venous
        self.C_art = C_art
        self.C_cap = C_cap
        self.k_abs_water = k_absorption_water
        self.k_abs_nutrients = k_absorption_nutrients
        self.portal_pressure_sensitivity = portal_pressure_sensitivity
        self.P_art0 = P_art0
        self.P_cap0 = P_cap0
        self._current_outputs = {}

    def get_state_size(self):
        return 2

    def get_initial_state(self):
        return np.array([self.P_art0, self.P_cap0])

    def _absorption_factor(self, P_portal):
        norm_p = 8.0
        if P_portal <= norm_p:
            return 1.0
        else:
            factor = 1.0 - self.portal_pressure_sensitivity * (P_portal - norm_p)
            return max(0.2, factor)

    def get_derivatives(self, t, state, inputs):
        P_art, P_cap = state
        P_sa = inputs.get('P_sa', 80.0)
        P_portal = inputs.get('P_portal', inputs.get('P_sv', 5.0))
        intake_water = inputs.get('intake_water', 0.0)
        intake_nutrients = inputs.get('intake_nutrients', 0.0)

        Q_in = (P_sa - P_art) / self.R_art
        Q_cap = (P_art - P_cap) / self.R_cap
        Q_out = (P_cap - P_portal) / self.R_venous

        abs_factor = self._absorption_factor(P_portal)
        absorption_water = self.k_abs_water * intake_water * abs_factor
        absorption_nutrients = self.k_abs_nutrients * intake_nutrients * abs_factor

        dP_art = (Q_in - Q_cap) / self.C_art
        dP_cap = (Q_cap + absorption_water - Q_out) / self.C_cap

        self._current_outputs = {
            'Q_out': Q_out,
            'absorption_water': absorption_water,
            'absorption_nutrients': absorption_nutrients,
            'portal_pressure_factor': abs_factor
        }
        return np.array([dP_art, dP_cap])

    def get_outputs(self, state):
        return self._current_outputs.copy()