# liver.py
import numpy as np
from organ_base import OrganModel

class Liver(OrganModel):
    """
    Модель печени здорового человека.
    Нормальная гемодинамика, клиренс билирубина и аммиака, синтез альбумина.
    Состояние: [P_hv, C_bilirubin, C_ammonia, C_albumin, резерв]
    """
    def __init__(self,
                 R_ha=17.0, R_pv_base=0.25, R_hv_base=0.12, C=5.0, 
                 C_portal=1.5, P_hv0=8.0, P_portal0=8.0,
                 albumin_prod_base=0.1,
                 bilirubin_clearance_base=0.2,
                 ammonia_clearance_base=0.15,
                 C_bilirubin0=0.0, C_ammonia0=0.0, C_albumin0=1.0):
        self.R_ha = R_ha
        self.R_pv_base = R_pv_base
        self.R_hv_base = R_hv_base
        self.C = C
        self.C_portal = C_portal
        self.P_hv0 = P_hv0
        self.P_portal0 = P_portal0
        self.albumin_prod_base = albumin_prod_base
        self.bilirubin_clearance_base = bilirubin_clearance_base
        self.ammonia_clearance_base = ammonia_clearance_base

        self.C_bilirubin0 = C_bilirubin0
        self.C_ammonia0 = C_ammonia0
        self.C_albumin0 = C_albumin0

        self._current_outputs = {}

    def get_state_size(self):
        return 6   # [P_hv, C_bil, C_amm, C_alb, reserve, P_portal]

    def get_initial_state(self):
        return np.array([self.P_hv0, 
                         self.C_bilirubin0, 
                         self.C_ammonia0, 
                         self.C_albumin0, 
                         0.0, 
                         self.P_portal0])

    def get_derivatives(self, t, state, inputs):
        P_hv = state[0]
        C_bil = state[1]
        C_amm = state[2]
        C_alb = state[3]
        # state[4] - резерв
        P_portal = state[5] # состояние из Windkessel

        P_sa = inputs['P_sa']
        P_sv = inputs['P_sv']
        C_bil_blood = inputs.get('C_bilirubin_blood', 0.0)
        C_amm_blood = inputs.get('C_ammonia_blood', 0.0)
        C_alb_blood = inputs.get('C_albumin_blood', 1.0)
        V_blood = inputs.get('V_blood', 5000.0)

        # явная связь с ЖКТ - приходит из whole_body.py
        Q_gut_out = inputs.get('Q_gut_out', inputs.get('Q_portal', 8.0))

        # Гемодинамика
        Q_pv = (P_portal - P_hv) / self.R_pv_base # Q_pv_into_liver
        Q_ha = (P_sa - P_hv) / self.R_ha # (90-8)/17=4.8
        Q_out = (P_hv - P_sv) / self.R_hv_base # в системные вены

        dP_hv = (Q_ha + Q_pv - Q_out) / self.C
        dP_portal = (Q_gut_out - Q_pv) / self.C_portal # <-- Windkessel портальной вены

        # Метаболизм
        uptake_bil = 0.1 * (C_bil_blood - C_bil)
        clearance_bil = self.bilirubin_clearance_base * C_bil
        dC_bil = uptake_bil - clearance_bil

        uptake_amm = 0.1 * (C_amm_blood - C_amm)
        clearance_amm = self.ammonia_clearance_base * C_amm
        dC_amm = uptake_amm - clearance_amm

        synthesis_alb = self.albumin_prod_base
        degradation_alb = 0.01 * C_alb
        release_alb = 0.05 * (C_alb - C_alb_blood)
        dC_alb = synthesis_alb - degradation_alb - release_alb

        dC_bil_blood = -clearance_bil / V_blood
        dC_amm_blood = -clearance_amm / V_blood
        dC_alb_blood = +release_alb / V_blood

        self._current_outputs = {
            'Q_liver_out': Q_out,
            'P_portal': P_portal, # теперь из состояния, а не P_mes
            'dC_bilirubin': dC_bil_blood,
            'dC_ammonia': dC_amm_blood,
            'dC_albumin': dC_alb_blood,
            'Q_ha': Q_ha,
            'Q_pv': Q_pv,
            'Q_gut_out': Q_gut_out,
            'functional': 1.0
        }

        return np.array([dP_hv, dC_bil, dC_amm, dC_alb, 0.0, dP_portal])

    def get_outputs(self, state):
        return self._current_outputs.copy()