# heart.py
import numpy as np
from organ_base import OrganModel

class Heart4Chambers(OrganModel):
    """
    Четырёхкамерная модель сердца с клапанами.
    Поддерживает дефект межжелудочковой перегородки (VSD) через параметр R_vsd.
    """
    def __init__(self,
                 hr=70,
                 E_max_la=0.25, E_min_la=0.05,
                 E_max_lv=2.5,  E_min_lv=0.06,
                 E_max_ra=0.20, E_min_ra=0.04,
                 E_max_rv=0.8,  E_min_rv=0.03,
                 V0_la=5, V0_lv=10, V0_ra=5, V0_rv=15,
                 R_mitral=0.05, R_aortic=0.03,
                 R_tricuspid=0.05, R_pulmonary=0.03,
                 R_venous=0.15,
                 R_vsd=np.inf,          # сопротивление дефекта (бесконечность = нет шунта)
                 hr_min=20, hr_max=200):
        self.hr_base = hr
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.T_base = 60 / hr

        self.E_max_base = {'LA': E_max_la, 'LV': E_max_lv,
                           'RA': E_max_ra, 'RV': E_max_rv}
        self.E_min = {'LA': E_min_la, 'LV': E_min_lv,
                      'RA': E_min_ra, 'RV': E_min_rv}
        self.V0 = {'LA': V0_la, 'LV': V0_lv, 'RA': V0_ra, 'RV': V0_rv}
        self.R_valve = {
            'mitral': R_mitral,
            'aortic': R_aortic,
            'tricuspid': R_tricuspid,
            'pulmonary': R_pulmonary
        }
        self.R_venous = R_venous
        self.R_vsd = R_vsd
        self._current_hr = self.hr_base
        self._current_T = self.T_base
        self._current_E_max = self.E_max_base.copy()
        self._current_flows = {}

    def get_state_size(self):
        return 4

    # def get_initial_state(self, P_la=1.5, P_lv=7.5, P_ra=1.2, P_rv=3.15):
    def get_initial_state(self, P_la=1.5, P_ra=1.5, P_lv=7.0, P_rv=3.0):
        # стационар диастолы
        V_la = self.V0['LA'] + P_la / self.E_min['LA']
        V_lv = self.V0['LV'] + P_lv / self.E_min['LV']
        V_ra = self.V0['RA'] + P_ra / self.E_min['RA']
        V_rv = self.V0['RV'] + P_rv / self.E_min['RV']
        return np.array([V_la, V_lv, V_ra, V_rv])
    # дефолт даст [35, 143, 85, 115] -> клипни к [35,135,35,115] для численной стабильности

    def _update_parameters(self, inputs):
        hr_factor = inputs.get('hr_factor', 1.0)
        new_hr = self.hr_base * hr_factor
        new_hr = np.clip(new_hr, self.hr_min, self.hr_max)
        self._current_hr = new_hr
        self._current_T = 60 / new_hr

        inotropy_factor = inputs.get('inotropy_factor', 1.0)
        for chamber in self.E_max_base:
            self._current_E_max[chamber] = self.E_max_base[chamber] * inotropy_factor

    def _elastance(self, t, chamber):
        tau = (t % self._current_T) / self._current_T
        Emax = self._current_E_max[chamber] # с учетом inotropy_factor
        Emin = self.E_min[chamber]

        if chamber in ('LA', 'RA'):
            # Предсердие: 0.8-1.0 цикла, полный косинусный купол 0->1->0
            if 0.8 <= tau <= 1.0:
                ph = (tau - 0.8) / 0.2
                e = 0.5 * (1 - np.cos(2 * np.pi * ph)) # C1=0 на краях
                return Emin + (Emax - Emin) * e
            else:
                return Emin
        else:
            # Желудочек: Stergiopulos - подъем 0-0.3, спад 0.3-0.45
            if tau <= 0.3:
                # изоволюмическое сокращение, медленный старт
                ph = tau / 0.3
                e = 0.5 * (1 - np.cos(np.pi * ph)) # 0->1
                return Emin + (Emax - Emin) * e
            elif tau <= 0.45:
                # изоволюмическое расслабление 0.15 цикла ~ 130мс при HR=70
                ph = (tau - 0.3) / 0.15
                e = 0.5 * (1 + np.cos(np.pi * ph)) # 1->0, было 1->0.5 с багом
                return Emin + (Emax - Emin) * e
            else:
                # диастола
                return Emin

    def _pressure(self, chamber, V, t):
        # Не допускаем объём меньше 50% от мёртвого объёма V0
        V_clamped = max(V, 0.5 * self.V0[chamber])
        effective_volume = V_clamped - self.V0[chamber]
        return self._elastance(t, chamber) * effective_volume

    def get_derivatives(self, t, state, inputs):
        V_la, V_lv, V_ra, V_rv = state
        self._update_parameters(inputs)

        P_sa = inputs['P_sa']
        P_sv = inputs['P_sv']
        P_pa = inputs['P_pa']
        P_pv = inputs['P_pv']

        P_la = self._pressure('LA', V_la, t)
        P_lv = self._pressure('LV', V_lv, t)
        P_ra = self._pressure('RA', V_ra, t)
        P_rv = self._pressure('RV', V_rv, t)

        mitral_open = P_la > P_lv
        aortic_open = P_lv > P_sa
        tricuspid_open = P_ra > P_rv
        pulmonary_open = P_rv > P_pa

        Q_mitral = (P_la - P_lv) / self.R_valve['mitral'] if mitral_open else 0.0
        Q_aortic = (P_lv - P_sa) / self.R_valve['aortic'] if aortic_open else 0.0
        Q_tricuspid = (P_ra - P_rv) / self.R_valve['tricuspid'] if tricuspid_open else 0.0
        Q_pulmonary = (P_rv - P_pa) / self.R_valve['pulmonary'] if pulmonary_open else 0.0

        # Шунт через ДМЖП (слева направо, если P_lv > P_rv)
        # if np.isfinite(self.R_vsd) and self.R_vsd > 0:
        #     Q_vsd = max(0.0, (P_lv - P_rv) / self.R_vsd)  # только слева направо
        # else:
        #     Q_vsd = 0.0

        # Шунт через ДМЖП - ДВУНАПРАВЛЕННЫЙ, для Эйзенменгера
        if np.isfinite(self.R_vsd) and self.R_vsd > 0:
            Q_vsd = (P_lv - P_rv) / self.R_vsd 
        else:
            Q_vsd = 0.0

        R_venous = self.R_venous
        Q_sv_to_ra = (P_sv - P_ra) / R_venous
        Q_pv_to_la = (P_pv - P_la) / R_venous

        self._current_flows = {
            'Q_aortic': Q_aortic,
            'Q_pulmonary': Q_pulmonary,
            'Q_sv_to_ra': Q_sv_to_ra,
            'Q_pv_to_la': Q_pv_to_la,
            'Q_vsd': Q_vsd,
            'Q_mitral': Q_mitral,
            'Q_tricuspid': Q_tricuspid
        }

        dV_la = Q_pv_to_la - Q_mitral
        dV_lv = Q_mitral - Q_aortic - Q_vsd
        dV_ra = Q_sv_to_ra - Q_tricuspid
        dV_rv = Q_tricuspid - Q_pulmonary + Q_vsd

        # Предотвращение отрицательных объёмов (volume clamp)
        min_frac = 0.5
        if V_la < min_frac * self.V0['LA'] and dV_la < 0:
            dV_la = 0.0
        if V_lv < min_frac * self.V0['LV'] and dV_lv < 0:
            dV_lv = 0.0
        if V_ra < min_frac * self.V0['RA'] and dV_ra < 0:
            dV_ra = 0.0
        if V_rv < min_frac * self.V0['RV'] and dV_rv < 0:
            dV_rv = 0.0

        return np.array([dV_la, dV_lv, dV_ra, dV_rv])

    def get_outputs(self, state):
        return self._current_flows.copy()