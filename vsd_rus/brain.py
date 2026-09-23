# brain.py
"""
brain.py

Рапортует O₂/CO₂; в кровь не пишет; принимает occlusion_factor.
Состояния: [P_br, C_O2_tis, C_CO2_tis, C_lac_tis, C_amm_tis] — 5 состояний.
"""

import numpy as np
from organ_base import OrganModel

class Brain(OrganModel):
    def __init__(self,
                R_base=7.0, C=4.0, P_autoreg=80.0,
                CMRO2_target=0.55,
                C_v_min=0.06, max_extraction=0.60,
                glucose_extraction=0.1,
                P0=47.5,
                C_a_O2_norm=0.20, C_a_CO2_norm=0.50,
                k_hypoxic_dilation=0.6, k_hypercapnic_dilation=1.2, k_myo=0.30,
                V_tissue=150.0, RQ=0.85,
                C_O2_critical=0.08,
                # Лактат
                C_lac_norm=0.8, k_lac_prod=0.08, k_lac_clear=0.02, k_lac_release=0.03,
                # Аммиак BBB
                C_amm_norm=0.3, k_amm_bbb_in=0.02, k_amm_bbb_out=0.01, k_amm_detox=0.01,
                k_amm_inhibition=0.15):
        self.R_base = float(R_base)
        self.C = float(C)
        self.P_autoreg = float(P_autoreg)
        self.CMRO2_target = float(CMRO2_target)
        self.C_v_min = float(C_v_min)
        self.max_extraction = float(max_extraction)
        self.glucose_extraction = float(glucose_extraction)
        self.P0 = float(P0)
        self.C_a_O2_norm = float(C_a_O2_norm)
        self.C_a_CO2_norm = float(C_a_CO2_norm)
        self.k_hypoxic_dilation = float(k_hypoxic_dilation)
        self.k_hypercapnic_dilation = float(k_hypercapnic_dilation)
        self.k_myo = float(k_myo)
        self.V_tissue = float(V_tissue)
        self.RQ = float(RQ)
        self.C_O2_critical = float(C_O2_critical)
        # лактат
        self.C_lac_norm = float(C_lac_norm)
        self.k_lac_prod = float(k_lac_prod)
        self.k_lac_clear = float(k_lac_clear)
        self.k_lac_release = float(k_lac_release)
        # аммиак
        self.C_amm_norm = float(C_amm_norm)
        self.k_amm_bbb_in = float(k_amm_bbb_in)
        self.k_amm_bbb_out = float(k_amm_bbb_out)
        self.k_amm_detox = float(k_amm_detox)
        self.k_amm_inhibition = float(k_amm_inhibition)
        self._current_outputs = {}

    def get_state_size(self):
        return 5

    def get_initial_state(self) -> np.ndarray:
        return np.array([
            self.P0,               # P_br
            self.C_a_O2_norm,      # C_O2_tis
            self.C_a_CO2_norm,     # C_CO2_tis
            self.C_lac_norm,       # C_lac_tis
            self.C_amm_norm,       # C_amm_tis
        ])

    def _autoregulation_resistance(self, P_sa, C_a_O2, C_a_CO2=None, C_tissue_CO2=None):
        x = (P_sa - self.P_autoreg) / self.P_autoreg
        f_P = 1.0 + self.k_myo * np.tanh(x)
        hypoxia = max(self.C_a_O2_norm - C_a_O2, 0.0) / self.C_a_O2_norm
        f_O2 = 1.0 - self.k_hypoxic_dilation * hypoxia
        if C_a_CO2 is None:
            C_a_CO2 = self.C_a_CO2_norm
        hypercapnia = (C_a_CO2 - self.C_a_CO2_norm) / self.C_a_CO2_norm
        if C_tissue_CO2 is not None:
            hypercapnia = max(hypercapnia, (C_tissue_CO2 - self.C_a_CO2_norm)/self.C_a_CO2_norm)
        f_CO2 = 1.0 - self.k_hypercapnic_dilation * hypercapnia
        reg = f_P * f_O2 * f_CO2
        reg = np.clip(reg, 0.35, 2.5)
        return self.R_base * reg, {'f_P': f_P, 'f_O2': f_O2, 'f_CO2': f_CO2}

    def get_derivatives(self, t, state, inputs):
        # разбор состояния
        P_br = state[0]
        C_O2_tis = state[1]
        C_CO2_tis = state[2]
        C_lac_tis = state[3]
        C_amm_tis = state[4]

        P_sa = float(inputs.get('P_sa', 80.0))
        P_sv = float(inputs.get('P_sv', 5.0))
        C_a_O2 = float(inputs.get('C_a_O2', self.C_a_O2_norm))
        C_a_CO2 = float(inputs.get('C_a_CO2', self.C_a_CO2_norm))
        C_a_lac = float(inputs.get('C_lactate_blood', inputs.get('C_a_lactate', 0.8)))
        C_a_amm = float(inputs.get('C_ammonia', inputs.get('C_a_ammonia', self.C_amm_norm)))
        V_blood = float(inputs.get('V_blood', 5800.0))
        V_blood = max(V_blood, 1e-6)

        occlusion = float(inputs.get('occlusion_factor', 1.0))  # 1.0 = норма, 0.0 = полная окклюзия

        # гемодинамика
        R_eff, f_autoreg = self._autoregulation_resistance(P_sa, C_a_O2, C_a_CO2, C_CO2_tis)
        Q_br_healthy = max((P_sa - P_br) / R_eff, 0.0)
        Q_br = Q_br_healthy * np.clip(occlusion, 0.0, 1.0)
        Q_out = max((P_br - P_sv)/R_eff, 0.0) * np.clip(occlusion, 0.0, 1.0)
        dP_br = (Q_br - Q_out) / self.C

        # O2 метаболизм
        if Q_br > 1e-6:
            extraction_needed = self.CMRO2_target / Q_br
        else:
            extraction_needed = self.max_extraction
        extraction_avail = max(C_a_O2 - self.C_v_min, 0.0)
        extraction_used = min(extraction_needed, extraction_avail, self.max_extraction)
        O2_cons = Q_br * extraction_used

        # ишемия + гипераммониемия -> ингибирование
        if C_O2_tis < self.C_O2_critical:
            inhib_O2 = np.clip(C_O2_tis / self.C_O2_critical, 0.2, 1.0)
        else:
            inhib_O2 = 1.0
        inhib_amm = 1.0 / (1.0 + self.k_amm_inhibition * max(C_amm_tis - self.C_amm_norm, 0.0))
        inhibition = inhib_O2 * inhib_amm

        O2_cons_eff = O2_cons * inhibition
        C_v_O2_brain = max(C_a_O2 - extraction_used, self.C_v_min)

        # CO2
        CO2_prod = O2_cons_eff * self.RQ
        C_v_CO2_brain = C_a_CO2 + (CO2_prod / Q_br if Q_br>1e-6 else 0.04)

        # тканевые O2/CO2
        dC_O2 = (Q_br * (C_a_O2 - C_O2_tis) - O2_cons_eff) / self.V_tissue
        dC_CO2 = (Q_br * (C_a_CO2 - C_CO2_tis) + CO2_prod) / self.V_tissue

        # лактат — анаэробный при гипоксии
        hypoxia_sev = max(self.C_O2_critical - C_O2_tis, 0.0) / self.C_O2_critical
        lac_prod = self.k_lac_prod * hypoxia_sev * (1.0 + 0.5*max(C_amm_tis - self.C_amm_norm,0.0))
        lac_clear = self.k_lac_clear * C_lac_tis
        lac_release = self.k_lac_release * max(C_lac_tis - C_a_lac, 0.0)
        dC_lac = lac_prod - lac_clear - lac_release
        C_v_lac_brain = C_lac_tis  # венозный лактат ≈ тканевой

        # аммиак BBB
        amm_in = self.k_amm_bbb_in * max(C_a_amm - C_amm_tis, 0.0)
        amm_out = self.k_amm_bbb_out * max(C_amm_tis - C_a_amm, 0.0)
        amm_detox = self.k_amm_detox * C_amm_tis
        dC_amm = amm_in - amm_out - amm_detox
        C_v_amm_brain = C_amm_tis

        # вклады в кровь
        dC_lac_blood = (lac_release * self.V_tissue) / V_blood
        dC_amm_blood = -dC_amm * self.V_tissue / V_blood

        self._current_outputs = {
            'Q_br': float(Q_br), 'Q_brain': float(Q_br),
            'VO2_brain': float(O2_cons_eff),
            'CMRO2_target': float(self.CMRO2_target),
            'C_v_O2': float(C_v_O2_brain),
            'C_v_O2_brain': float(C_v_O2_brain),
            'C_v_CO2_brain': float(C_v_CO2_brain),
            'C_v_lactate_brain': float(C_v_lac_brain),
            'C_v_ammonia_brain': float(C_v_amm_brain),
            'C_O2_tissue': float(C_O2_tis),
            'C_CO2_tissue': float(C_CO2_tis),
            'C_lactate_tissue': float(C_lac_tis),
            'C_ammonia_tissue': float(C_amm_tis),
            'R_eff': float(R_eff),
            'f_P_myogenic': float(f_autoreg['f_P']),
            'f_O2_autoreg': float(f_autoreg['f_O2']),
            'f_CO2_autoreg': float(f_autoreg['f_CO2']),
            'extraction_used': float(extraction_used),
            'metabolic_inhibition': float(inhibition),
            'inhib_O2': float(inhib_O2),
            'inhib_amm': float(inhib_amm),
            'dC_lactate_blood': float(dC_lac_blood),
            'dC_ammonia_blood': float(dC_amm_blood),
            'CO2_production': float(CO2_prod),
            'lactate_production': float(lac_prod),
            'Q_out': float(Q_out),
            'occlusion_factor': float(occlusion),
        }
        return np.array([dP_br, dC_O2, dC_CO2, dC_lac, dC_amm])

    def get_outputs(self, state):
        return self._current_outputs.copy()
