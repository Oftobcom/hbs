# heart.py
import numpy as np
from organ_base import OrganModel


class Heart4Chambers(OrganModel):
    """
    Четырёхкамерная модель сердца с клапанами и дефектом межжелудочковой перегородки (VSD).
    Модель допускает модуляцию частоты сердечных сокращений и сократимости
    под влиянием внешних факторов (например, воспаления при гепатите B).

    Состояние: [V_LA, V_LV, V_RA, V_RV] (мл)

    Входы (inputs):
        P_sa, P_sv, P_pa, P_pv (float) – давления в сосудах (мм рт. ст.)
        hr_factor (float, optional) – множитель для ЧСС (по умолчанию 1.0)
        inotropy_factor (float, optional) – множитель для сократимости (по умолчанию 1.0)

    Выходы:
        Q_aortic, Q_pulmonary, Q_sv_to_ra, Q_pv_to_la, Q_vsd (мл/с)
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
                 R_vsd=np.inf,
                 hr_min=20, hr_max=200):        # ограничения для ЧСС
        self.hr_base = hr
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.T_base = 60 / hr                     # базовый период

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
        self.R_vsd = R_vsd

        # Текущие значения (будут обновляться в каждом шаге)
        self._current_hr = self.hr_base
        self._current_T = self.T_base
        self._current_E_max = self.E_max_base.copy()
        self._current_flows = {}

    def get_state_size(self):
        return 4

    def get_initial_state(self):
        return np.array([30.0, 120.0, 30.0, 110.0])

    def _update_parameters(self, inputs):
        """Обновляет ЧСС и сократимость на основе входных факторов."""
        # Частота
        hr_factor = inputs.get('hr_factor', 1.0)
        new_hr = self.hr_base * hr_factor
        new_hr = np.clip(new_hr, self.hr_min, self.hr_max)
        self._current_hr = new_hr
        self._current_T = 60 / new_hr

        # Сократимость (inotropy)
        inotropy_factor = inputs.get('inotropy_factor', 1.0)
        for chamber in self.E_max_base:
            self._current_E_max[chamber] = self.E_max_base[chamber] * inotropy_factor

    def _elastance(self, t, chamber):
        """
        Вычисляет эластинс камеры в момент времени t с учётом
        текущей ЧСС и текущей максимальной эластинсы.
        """
        tau = (t % self._current_T) / self._current_T
        Emax = self._current_E_max[chamber]
        Emin = self.E_min[chamber]

        if chamber in ('LA', 'RA'):          # предсердия
            if 0.8 <= tau <= 1.0:
                phase = (tau - 0.8) / 0.2
                return Emin + (Emax - Emin) * np.sin(np.pi * phase)
            else:
                return Emin
        else:                                 # желудочки
            if tau <= 0.35:
                phase = tau / 0.35
                return Emin + (Emax - Emin) * np.sin(np.pi * phase)
            else:
                return Emin

    def _pressure(self, chamber, V, t):
        """Давление в камере: P = E(t)*(V - V0)."""
        return self._elastance(t, chamber) * (V - self.V0[chamber])

    def get_derivatives(self, t, state, inputs):
        V_la, V_lv, V_ra, V_rv = state

        # Обновляем параметры под влиянием внешних факторов
        self._update_parameters(inputs)

        # Давления в сосудах (обязательные входы)
        P_sa = inputs['P_sa']
        P_sv = inputs['P_sv']
        P_pa = inputs['P_pa']
        P_pv = inputs['P_pv']

        # Давления в камерах
        P_la = self._pressure('LA', V_la, t)
        P_lv = self._pressure('LV', V_lv, t)
        P_ra = self._pressure('RA', V_ra, t)
        P_rv = self._pressure('RV', V_rv, t)

        # Состояния клапанов и потоки
        mitral_open = P_la > P_lv
        aortic_open = P_lv > P_sa
        tricuspid_open = P_ra > P_rv
        pulmonary_open = P_rv > P_pa

        Q_mitral = (P_la - P_lv) / self.R_valve['mitral'] if mitral_open else 0.0
        Q_aortic = (P_lv - P_sa) / self.R_valve['aortic'] if aortic_open else 0.0
        Q_tricuspid = (P_ra - P_rv) / self.R_valve['tricuspid'] if tricuspid_open else 0.0
        Q_pulmonary = (P_rv - P_pa) / self.R_valve['pulmonary'] if pulmonary_open else 0.0

        # VSD (положительное — слева направо)
        if np.isfinite(self.R_vsd):
            Q_vsd = (P_lv - P_rv) / self.R_vsd
        else:
            Q_vsd = 0.0

        # Венозный возврат (сопротивление 0.1 выбрано эмпирически)
        R_venous = 0.1
        Q_sv_to_ra = (P_sv - P_ra) / R_venous
        Q_pv_to_la = (P_pv - P_la) / R_venous

        # Сохраняем для get_outputs
        self._current_flows = {
            'Q_aortic': Q_aortic,
            'Q_pulmonary': Q_pulmonary,
            'Q_sv_to_ra': Q_sv_to_ra,
            'Q_pv_to_la': Q_pv_to_la,
            'Q_vsd': Q_vsd,
            'Q_mitral': Q_mitral,
            'Q_tricuspid': Q_tricuspid
        }

        # Производные объёмов
        dV_la = Q_pv_to_la - Q_mitral
        dV_lv = Q_mitral - Q_aortic - Q_vsd
        dV_ra = Q_sv_to_ra - Q_tricuspid
        dV_rv = Q_tricuspid - Q_pulmonary + Q_vsd

        return np.array([dV_la, dV_lv, dV_ra, dV_rv])

    def get_outputs(self, state):
        return self._current_flows.copy()