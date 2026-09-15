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
                    #  E_max_la=0.25, E_min_la=0.20,
                    E_max_la=0.25, E_min_la=0.08,  # комплаенс ЛП 5→12.5 мл/мм
                    E_max_lv=3.5,  E_min_lv=0.06,
                    #  E_max_ra=0.20, E_min_ra=0.08,
                    E_max_ra=0.20, E_min_ra=0.04,
                    E_max_rv=0.8,  E_min_rv=0.03,
                    #  V0_la=15, V0_lv=10, V0_ra=8, V0_rv=15,
                    V0_la=10, V0_lv=10, V0_ra=5, V0_rv=10,
                    #  R_mitral=0.02, R_aortic=0.15,
                    R_mitral=0.03, R_aortic=0.10,   # митральный в 1.6× меньше
                    #  R_tricuspid=0.05, R_pulmonary=0.06,
                    R_tricuspid=0.03, R_pulmonary=0.05,
                    #  R_venous=0.10, 
                    R_venous=0.05, # для тестов с низким венозным сопротивлением
                    R_vsd=np.inf,          # сопротивление дефекта (бесконечность = нет шунта)
                    hr_min=30, hr_max=130,
                    k_valve=20.0):
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
        self.k_valve = float(k_valve)
        self._current_hr = self.hr_base
        self._current_T = self.T_base
        self._current_E_max = self.E_max_base.copy()
        self._current_flows = {}

    def get_state_size(self):
        return 4

    # def get_initial_state(self, P_la=1.5, P_lv=7.5, P_ra=1.2, P_rv=3.15):
    # def get_initial_state(self, P_la=1.5, P_ra=1.5, P_lv=7.0, P_rv=3.0):
    def get_initial_state(self, P_la=8.0, P_ra=5.0, P_lv=5.0, P_rv=2.5):
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
        # Симпатическая активация: при падении P_sa растёт E_max
        # через барорефлексный сигнал
        baro_activation = inputs.get('baro_activation', 1.0)
        for chamber in self.E_max_base:
            factor = 1.0
            if chamber in ('LV','RV'):
                factor = baro_activation
            elif chamber in ('LA','RA'):
                factor = 1.0 + 0.2*(baro_activation-1.0) # предсердия слабее
            self._current_E_max[chamber] = self.E_max_base[chamber] * inotropy_factor * factor

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
        effective_volume = max(V - self.V0[chamber], 0.0)
        # Не допускаем объём меньше 50% от мёртвого объёма V0
        # V_clamped = max(V, 0.5 * self.V0[chamber])
        # effective_volume = V_clamped - self.V0[chamber]
        return self._elastance(t, chamber) * effective_volume
    
    def _valve_flow(self, dP, R):
        """
        Гладкий поток через клапан (замена ступеньки P_la > P_lv).

        При dP >> 0  → dP/R (клапан полностью открыт)
        При dP << 0  → ~0   (клапан закрыт, обратного тока нет)
        Переход шириной ~ 5/k_valve мм рт. ст.

        Численно устойчивая реализация: аргумент экспоненты клиппится
        в [-60, 60], чтобы избежать overflow при |dP| > ~1000 мм рт. ст.
        При z = -60: exp(60) ≈ 1e26, 1 + exp(60) ≈ 1e26, sigmoid ≈ 0.
        При z = +60: exp(-60) ≈ 1e-26, sigmoid ≈ 1.
        """
        z = -self.k_valve * dP
        z = np.clip(z, -60.0, 60.0)
        return dP / R / (1.0 + np.exp(z))

    def _soft_clamp(self, V, V_min, dV):
        """
        Плавно обнуляет dV у нижней границы V_min.

        При V > 1.5·V_min        — возвращает dV без изменений.
        При V_min < V ≤ 1.5·V_min — плавное затухание: softness ∈ (0, 1).
        При V ≤ V_min             — softness = 0, dV вниз обнуляется.

        Численно устойчивая реализация: аргумент экспоненты клиппится
        в [-60, 60], а softness — в [0, 1]. Это защищает от overflow,
        когда V уходит глубоко в минус из-за численных сбоев.
        """
        # Нормальный режим: далеко от границы — ничего не трогаем
        if V > 1.5 * V_min:
            return dV

        # Только для dV < 0 имеет смысл давить (dV > 0 — приток, не мешаем)
        if dV >= 0.0:
            return dV

        # Безопасное вычисление softness
        z = -10.0 * (V - V_min) / V_min
        z = float(np.clip(z, -60.0, 60.0))
        softness = 1.0 - float(np.exp(z))
        softness = float(np.clip(softness, 0.0, 1.0))

        return dV * softness
    
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

        # Q_mitral = (P_la - P_lv) / self.R_valve['mitral'] if mitral_open else 0.0
        # Q_aortic = (P_lv - P_sa) / self.R_valve['aortic'] if aortic_open else 0.0
        # Q_tricuspid = (P_ra - P_rv) / self.R_valve['tricuspid'] if tricuspid_open else 0.0
        # Q_pulmonary = (P_rv - P_pa) / self.R_valve['pulmonary'] if pulmonary_open else 0.0

        # Гладкие потоки через клапаны (C^∞, без разрывов)
        Q_mitral    = self._valve_flow(P_la - P_lv, self.R_valve['mitral'])
        Q_aortic    = self._valve_flow(P_lv - P_sa, self.R_valve['aortic'])
        Q_tricuspid = self._valve_flow(P_ra - P_rv, self.R_valve['tricuspid'])
        Q_pulmonary = self._valve_flow(P_rv - P_pa, self.R_valve['pulmonary'])

        # Шунт через ДМЖП (слева направо, если P_lv > P_rv)
        # if np.isfinite(self.R_vsd) and self.R_vsd > 0:
        #     Q_vsd = max(0.0, (P_lv - P_rv) / self.R_vsd)  # только слева направо
        # else:
        #     Q_vsd = 0.0

        # Шунт через ДМЖП - ДВУНАПРАВЛЕННЫЙ, для Эйзенменгера
        if np.isfinite(self.R_vsd) and self.R_vsd > 0:
            # Q_vsd = (P_lv - P_rv) / self.R_vsd 
            Q_vsd = self._valve_flow(P_lv - P_rv, self.R_vsd)
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
            'Q_tricuspid': Q_tricuspid,
            # --- давления в камерах (для PV-петель и диагностики) ---
            'P_la': P_la,
            'P_lv': P_lv,
            'P_ra': P_ra,
            'P_rv': P_rv,
        }

        dV_la = Q_pv_to_la - Q_mitral
        dV_lv = Q_mitral - Q_aortic - Q_vsd
        dV_ra = Q_sv_to_ra - Q_tricuspid
        dV_rv = Q_tricuspid - Q_pulmonary + Q_vsd

        dV_la = self._soft_clamp(V_la, self.V0['LA'], dV_la)
        dV_lv = self._soft_clamp(V_lv, self.V0['LV'], dV_lv)
        dV_ra = self._soft_clamp(V_ra, self.V0['RA'], dV_ra)
        dV_rv = self._soft_clamp(V_rv, self.V0['RV'], dV_rv)

        return np.array([dV_la, dV_lv, dV_ra, dV_rv])

    def get_outputs(self, state):
        return self._current_flows.copy()