# kidney.py
import numpy as np
from organ_base import OrganModel

class KidneyHemodynamic(OrganModel):
    """
    Healthy kidneys – autoregulation of GFR, no toxin/disease effects.
    """
    def __init__(self,
                 GFR_base=120.0,
                 P_autoreg=95.0,
                 autoreg_gain=0.02,
                 toxin_clearance_frac=0.2,
                 volume_reabsorption_frac=0.99,
                 renal_resistance=0.02):
        self.GFR_base = GFR_base / 2.0
        self.P_autoreg = P_autoreg
        self.autoreg_gain = autoreg_gain
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
        delta = P_art - self.P_autoreg
        reg = 1.0 + self.autoreg_gain * delta
        reg = max(0.2, min(reg, 1.8))
        return 2 * self.GFR_base * reg

    def compute_effects(self, P_sa, P_sv, C_tox, V_blood):
        R_eff = self.renal_resistance
        Q_renal = (P_sa - P_sv) / R_eff
        GFR = self._compute_gfr(P_sa)

        toxin_filtered = GFR * C_tox
        toxin_excreted = self.toxin_clearance_frac * toxin_filtered
        dC_tox = -toxin_excreted / max(V_blood, 1e-6)

        urine_output = GFR * (1 - self.volume_reabsorption_frac)
        dV_blood = -urine_output

        self._current_outputs = {'Q_renal': Q_renal, 'GFR': GFR}
        return {
            'dC_tox': dC_tox,
            'dV_blood': dV_blood,
            'Q_renal': Q_renal,
            'GFR': GFR
        }