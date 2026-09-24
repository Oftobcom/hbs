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
                    E_max_la=0.25, E_min_la=0.09,   # было 0.08
                    E_max_ra=0.20, E_min_ra=0.05,   # было 0.03
                    E_max_lv=3.5,  E_min_lv=0.03,
                    E_max_rv=0.8,  E_min_rv=0.02,
                    V0_la=10, V0_lv=10, V0_ra=5, V0_rv=10,
                    EDV_la=80.0, EDV_lv=120.0, EDV_ra=40.0, EDV_rv=120.0,
                    R_mitral=0.02, R_aortic=0.10,   # митральный в 1.6× меньше
                    R_tricuspid=0.02, R_pulmonary=0.05,
                    R_venous_sys=0.08,   # системные вены (valves, гравитация)
                    R_venous_pulm=0.03,  # лёгочные вены — низкорезистивные
                    R_vsd=np.inf,  # сопротивление дефекта (бесконечность = нет шунта)
                    hr_min=30, hr_max=130,
                    k_valve=9.0):
        self.hr_base = hr
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.T_base = 60 / hr

        self.E_max_base = {'LA': E_max_la, 'LV': E_max_lv,
                           'RA': E_max_ra, 'RV': E_max_rv}
        self.E_min = {'LA': E_min_la, 'LV': E_min_lv,
                      'RA': E_min_ra, 'RV': E_min_rv}
        self.V0 = {'LA': V0_la, 'LV': V0_lv, 'RA': V0_ra, 'RV': V0_rv}
        self.EDV = {'LA': EDV_la, 'LV': EDV_lv, 'RA': EDV_ra, 'RV': EDV_rv}
        self.R_valve = {
            'mitral': R_mitral,
            'aortic': R_aortic,
            'tricuspid': R_tricuspid,
            'pulmonary': R_pulmonary
        }
        self.R_venous_sys = R_venous_sys
        self.R_venous_pulm = R_venous_pulm
        self.R_vsd = R_vsd
        self.k_valve = float(k_valve)
        self._current_hr = self.hr_base
        self._current_T = self.T_base
        self._current_E_max = self.E_max_base.copy()
        self._current_flows = {}

    def get_state_size(self):
        return 4

    def get_initial_state(self) -> np.ndarray:
        """
        Начальное состояние — физиологические конечно-диастолические
        объёмы (мл). Соответствующие давления в диастолу:
            P_la ≈ 2.4, P_lv ≈ 6.6, P_ra ≈ 1.4, P_rv ≈ 3.3 мм рт. ст.
        """
        return np.array([
            self.EDV['LA'], self.EDV['LV'], self.EDV['RA'], self.EDV['RV'],
        ])

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
        Emax = self._current_E_max[chamber]
        Emin = self.E_min[chamber]

        if chamber in ('LA', 'RA'):
            # Предсердия — без изменений
            if 0.8 <= tau <= 1.0:
                ph = (tau - 0.8) / 0.2
                e = 0.5 * (1 - np.cos(2 * np.pi * ph))
                return Emin + (Emax - Emin) * e
            return Emin

        # --- Желудочки: асимметричный колокол ---
        # t_peak = 0.35 (стандарт физиологии)
        # t_end  = 0.50 (систола 50%, компромисс)
        # beta   = 1.0 (Shim, стандарт)

        T_PEAK = 0.33
        T_END = 0.45
        BETA = 1.1 # медленный старт релаксации, быстрый конец

        if tau <= T_PEAK:
            ph = tau / T_PEAK
            e = 0.5 * (1 - np.cos(np.pi * ph))
        elif tau <= T_END:
            ph = (tau - T_PEAK) / (T_END - T_PEAK)
            e = 0.5 * (1 + np.cos(np.pi * (ph ** BETA)))
        else:
            e = 0.0

        return Emin + (Emax - Emin) * e

    def _pressure(self, chamber, V, t):
        dV = max(V - self.V0[chamber], 0.0)
        E = self._elastance(t, chamber)

        if chamber in ('LA', 'RA'):
            A = 0.4 if chamber=='LA' else 0.3
            k = 0.025 if chamber=='LA' else 0.02
            exp_arg = float(np.clip(k * dV, 0.0, 8.0))
            P_passive = A * (np.exp(exp_arg) - 1.0)
            return E * dV + P_passive

        # Желудочки — линейная + пассивная экспонента
        if chamber == 'LV':
            A_v, k_v = 0.03, 0.02
        else:
            A_v, k_v = 0.02, 0.015
        exp_arg = float(np.clip(k_v * dV, 0.0, 8.0))
        P_passive = A_v * (np.exp(exp_arg) - 1.0)
        return E * dV + P_passive
    
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

        # Гладкие потоки через клапаны (C^∞, без разрывов)
        Q_mitral    = self._valve_flow(P_la - P_lv, self.R_valve['mitral'])
        Q_aortic    = self._valve_flow(P_lv - P_sa, self.R_valve['aortic'])
        Q_tricuspid = self._valve_flow(P_ra - P_rv, self.R_valve['tricuspid'])
        Q_pulmonary = self._valve_flow(P_rv - P_pa, self.R_valve['pulmonary'])

        # Шунт через ДМЖП — двунаправленный линейный резистор (отверстие
        # в перегородке, а не клапан). НЕ использовать _valve_flow — он
        # односторонний и блокирует R→L при Эйзенменгере.
        # Q_vsd > 0 — L→R, Q_vsd < 0 — R→L.
        if np.isfinite(self.R_vsd) and self.R_vsd > 1e-9:
            Q_vsd = (P_lv - P_rv) / self.R_vsd
        else:
            Q_vsd = 0.0

        Q_sv_to_ra = (P_sv - P_ra) / self.R_venous_sys
        Q_pv_to_la = (P_pv - P_la) / self.R_venous_pulm

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