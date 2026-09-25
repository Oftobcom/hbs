# heart.py
import numpy as np
from organ_base import OrganModel


class Heart4Chambers(OrganModel):
    """
    Четырёхкамерная модель сердца с клапанами.
    Поддерживает дефект межжелудочковой перегородки (VSD) через параметр R_vsd.

    Состояние: [V_la, V_lv, V_ra, V_rv] — объёмы камер, мл.

    4 клапана односторонние (smooth ReLU, v(0)=0):
        mitral, aortic, tricuspid, pulmonary.
    VSD — двунаправленный линейный резистор (отверстие, не клапан):
        Q_vsd > 0 — L→R, Q_vsd < 0 — R→L.
    """

    # --- Санити-пороги для валидации конфигурации ---
    _K_VALVE_MIN = 1.0
    _K_VALVE_MAX = 100.0
    _R_VALVE_MIN = 1e-3
    _R_VALVE_MAX = 10.0
    _R_VENOUS_MIN = 1e-3
    _R_VENOUS_MAX = 10.0
    _HR_MIN = 20.0
    _HR_MAX = 250.0
    _E_MIN_LO = 0.0
    _E_MIN_HI = 1.0
    _E_MAX_LO = 0.01
    _E_MAX_HI = 20.0
    _V0_MIN = 0.0
    _V0_MAX = 100.0
    _EDV_MIN = 10.0
    _EDV_MAX = 500.0

    def __init__(self,
                 hr=70,
                 E_max_la=0.25, E_min_la=0.09,
                 E_max_ra=0.20, E_min_ra=0.05,
                 E_max_lv=3.5,  E_min_lv=0.03,
                 E_max_rv=0.8,  E_min_rv=0.02,
                 V0_la=10, V0_lv=10, V0_ra=5, V0_rv=10,
                 EDV_la=80.0, EDV_lv=120.0, EDV_ra=40.0, EDV_rv=120.0,
                 R_mitral=0.02, R_aortic=0.10,
                 R_tricuspid=0.02, R_pulmonary=0.05,
                 R_venous_sys=0.08,
                 R_venous_pulm=0.03,
                 R_vsd=np.inf,
                 hr_min=30, hr_max=130,
                 k_valve=9.0):

        # =================================================================
        # Валидация конфигурации — fail-fast при инициализации.
        # Параметры приходят из YAML и не меняются во время симуляции;
        # ошибки ловятся один раз, а не в горячем пути RHS.
        # =================================================================

        def _check_range(name, v, lo, hi, typical=""):
            v = float(v)
            if not np.isfinite(v) or not (lo <= v <= hi):
                raise ValueError(
                    f"Heart4Chambers: {name}={v} вне [{lo}, {hi}]. {typical}"
                )
            return v

        # --- k_valve (крутизна виртуального клапана) ---
        # Тот же диапазон, что у k_flow в lungs.py, для согласованности.
        self.k_valve = _check_range(
            "k_valve", k_valve, self._K_VALVE_MIN, self._K_VALVE_MAX,
            "типично 9 (ширина виртуального клапана ~1/k_valve мм рт.ст.). "
            "k_valve → 0 даёт нефизичную утечку v(0)=δ/(2R); "
            "k_valve > 100 делает клапан численно жёстким."
        )

        # --- HR и его границы ---
        hr_f = _check_range(
            "hr", hr, self._HR_MIN, self._HR_MAX,
            "типично 60–90 уд/мин."
        )
        hr_min_f = _check_range(
            "hr_min", hr_min, self._HR_MIN, self._HR_MAX,
            "типично 30 уд/мин."
        )
        hr_max_f = _check_range(
            "hr_max", hr_max, self._HR_MIN, self._HR_MAX,
            "типично 130 уд/мин."
        )
        if hr_min_f >= hr_max_f:
            raise ValueError(
                f"Heart4Chambers: hr_min={hr_min_f} должно быть < hr_max={hr_max_f}."
            )

        self.hr_base = hr_f
        self.hr_min = hr_min_f
        self.hr_max = hr_max_f
        self.T_base = 60.0 / hr_f

        # --- E_min / E_max по камерам ---
        # E_min — пассивная эластанс в диастолу (обычно < 0.1)
        # E_max — активная эластанс в систолу (LV обычно 3.5, RV 0.8)
        E_min = {'LA': E_min_la, 'LV': E_min_lv, 'RA': E_min_ra, 'RV': E_min_rv}
        E_max = {'LA': E_max_la, 'LV': E_max_lv, 'RA': E_max_ra, 'RV': E_max_rv}
        for chamber, val in E_min.items():
            self_check = _check_range(
                f"E_min_{chamber.lower()}", val, self._E_MIN_LO, self._E_MIN_HI,
                "типично 0.02–0.10 мм рт.ст./мл."
            )
            E_min[chamber] = self_check
        for chamber, val in E_max.items():
            checked = _check_range(
                f"E_max_{chamber.lower()}", val, self._E_MAX_LO, self._E_MAX_HI,
                "LV ~3.5, RV ~0.8, LA ~0.25, RA ~0.20 мм рт.ст./мл."
            )
            E_max[chamber] = checked
            if E_max[chamber] <= E_min[chamber]:
                raise ValueError(
                    f"Heart4Chambers: E_max_{chamber.lower()}={E_max[chamber]} "
                    f"должно быть > E_min_{chamber.lower()}={E_min[chamber]}."
                )
        self.E_min = E_min
        self.E_max_base = E_max

        # --- V0 (unstressed volume) ---
        V0_in = {'LA': V0_la, 'LV': V0_lv, 'RA': V0_ra, 'RV': V0_rv}
        for chamber, val in V0_in.items():
            V0_in[chamber] = _check_range(
                f"V0_{chamber.lower()}", val, self._V0_MIN, self._V0_MAX,
                "типично 5–15 мл."
            )
        self.V0 = V0_in

        # --- EDV (начальное состояние) ---
        EDV_in = {'LA': EDV_la, 'LV': EDV_lv, 'RA': EDV_ra, 'RV': EDV_rv}
        for chamber, val in EDV_in.items():
            EDV_in[chamber] = _check_range(
                f"EDV_{chamber.lower()}", val, self._EDV_MIN, self._EDV_MAX,
                "типично 80–150 мл."
            )
            if EDV_in[chamber] <= V0_in[chamber]:
                raise ValueError(
                    f"Heart4Chambers: EDV_{chamber.lower()}={EDV_in[chamber]} "
                    f"должно быть > V0_{chamber.lower()}={V0_in[chamber]} "
                    f"(иначе P(EDV)=0 и камера не наполнена)."
                )
        self.EDV = EDV_in

        # --- Клапанные сопротивления ---
        R_valve_in = {
            'mitral': R_mitral,
            'aortic': R_aortic,
            'tricuspid': R_tricuspid,
            'pulmonary': R_pulmonary,
        }
        for name, val in R_valve_in.items():
            R_valve_in[name] = _check_range(
                f"R_{name}", val, self._R_VALVE_MIN, self._R_VALVE_MAX,
                "типично 0.02–0.10 мм рт.ст.·с/мл."
            )
        self.R_valve = R_valve_in

        # --- Венозные сопротивления ---
        self.R_venous_sys = _check_range(
            "R_venous_sys", R_venous_sys, self._R_VENOUS_MIN, self._R_VENOUS_MAX,
            "типично 0.05–0.10 мм рт.ст.·с/мл (SVC/IVC → RA)."
        )
        self.R_venous_pulm = _check_range(
            "R_venous_pulm", R_venous_pulm, self._R_VENOUS_MIN, self._R_VENOUS_MAX,
            "типично 0.02–0.05 мм рт.ст.·с/мл (PV → LA)."
        )

        # --- VSD (двунаправленный резистор) ---
        # Может быть np.inf (нет шунта), положительным числом, или None.
        # Негативные/нулевые значения физически невозможны.
        if R_vsd is None or (isinstance(R_vsd, float) and np.isinf(R_vsd)):
            self.R_vsd = np.inf
        else:
            r_vsd = float(R_vsd)
            if not np.isfinite(r_vsd) or r_vsd <= 0.0:
                raise ValueError(
                    f"Heart4Chambers: R_vsd={R_vsd} должно быть > 0 "
                    f"или np.inf (нет шунта)."
                )
            self.R_vsd = r_vsd

        # --- Служебные поля ---
        self._current_hr = self.hr_base
        self._current_T = self.T_base
        self._current_E_max = self.E_max_base.copy()
        self._current_flows = {}

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
    def get_state_size(self):
        return 4   # [V_la, V_lv, V_ra, V_rv]

    def get_initial_state(self) -> np.ndarray:
        """
        Начальное состояние — физиологические конечно-диастолические
        объёмы (мл). Соответствующие давления в диастолу:
            P_la ≈ 2.4, P_lv ≈ 6.6, P_ra ≈ 1.4, P_rv ≈ 3.3 мм рт. ст.
        """
        return np.array([
            self.EDV['LA'], self.EDV['LV'], self.EDV['RA'], self.EDV['RV'],
        ])

    # ------------------------------------------------------------------
    # Обновление параметров (HR, инотропия, бароактивация)
    # ------------------------------------------------------------------
    def _update_parameters(self, inputs):
        hr_factor = inputs.get('hr_factor', 1.0)
        new_hr = self.hr_base * hr_factor
        new_hr = np.clip(new_hr, self.hr_min, self.hr_max)
        self._current_hr = new_hr
        self._current_T = 60.0 / new_hr

        inotropy_factor = inputs.get('inotropy_factor', 1.0)
        # Симпатическая активация: при падении P_sa растёт E_max
        # через барорефлексный сигнал
        baro_activation = inputs.get('baro_activation', 1.0)
        for chamber in self.E_max_base:
            factor = 1.0
            if chamber in ('LV', 'RV'):
                factor = baro_activation
            elif chamber in ('LA', 'RA'):
                factor = 1.0 + 0.2 * (baro_activation - 1.0)  # предсердия слабее
            self._current_E_max[chamber] = (
                self.E_max_base[chamber] * inotropy_factor * factor
            )

    # ------------------------------------------------------------------
    # Эластанс-профиль цикла
    # ------------------------------------------------------------------
    def _elastance(self, t, chamber):
        tau = (t % self._current_T) / self._current_T
        Emax = self._current_E_max[chamber]
        Emin = self.E_min[chamber]

        if chamber in ('LA', 'RA'):
            # Предсердия — систола в конце цикла
            if 0.8 <= tau <= 1.0:
                ph = (tau - 0.8) / 0.2
                e = 0.5 * (1 - np.cos(2 * np.pi * ph))
                return Emin + (Emax - Emin) * e
            return Emin

        # --- Желудочки: асимметричный колокол ---
        T_PEAK = 0.33
        T_END = 0.45
        BETA = 1.1  # медленный старт релаксации, быстрый конец

        if tau <= T_PEAK:
            ph = tau / T_PEAK
            e = 0.5 * (1 - np.cos(np.pi * ph))
        elif tau <= T_END:
            ph = (tau - T_PEAK) / (T_END - T_PEAK)
            e = 0.5 * (1 + np.cos(np.pi * (ph ** BETA)))
        else:
            e = 0.0

        return Emin + (Emax - Emin) * e

    # ------------------------------------------------------------------
    # Давление в камере (линейная + пассивная экспонента)
    # ------------------------------------------------------------------
    def _pressure(self, chamber, V, t):
        dV = max(V - self.V0[chamber], 0.0)
        E = self._elastance(t, chamber)

        if chamber in ('LA', 'RA'):
            A = 0.4 if chamber == 'LA' else 0.3
            k = 0.025 if chamber == 'LA' else 0.02
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

    # ------------------------------------------------------------------
    # Гладкий односторонний клапан (smooth ReLU с δ-вычитанием)
    # ------------------------------------------------------------------
    def _valve_flow(self, dP: float, R: float) -> float:
        """
        Гладкий односторонний клапан.

            v(dP) = max( 0, (dP + sqrt(dP² + δ²) − δ) / (2R) )
            где δ = 1/k_valve — ширина сглаживания.

        Свойства:
            dP = 0     →  v = 0              (клапан полностью закрыт, утечки нет)
            dP >> δ    →  v ≈ dP/R − δ/(2R)  (ламинарный поток)
            dP << −δ   →  v = 0              (обратного тока нет)

        Гарантированно ≥ 0 при любом dP. Гладкая C¹ (излом производной
        только в dP = 0). Монотонно не убывает.

        δ вычитается, чтобы v(0) = 0. Без него v(0) = δ/(2R) > 0 —
        постоянная «утечка», которая в переходных процессах даёт
        нефизичный ретроградный поток через закрытый клапан.

        Применяется к 4 клапанам (mitral, aortic, tricuspid, pulmonary).
        НЕ применяется к VSD — это отверстие в перегородке с двунаправленным
        потоком (Q_vsd = (P_lv − P_rv) / R_vsd).

        Численная защита:
          • R клипуется к 1e-6 (от solver retries, не от ошибок конфига).
          • R=NaN/Inf отлавливается через np.isfinite.
          • np.hypot(dP, δ) вместо sqrt(dP²+δ²) — не переполняется
            при больших |dP|.
        """
        if not np.isfinite(R):
            R_safe = 1e-6
        else:
            R_safe = max(float(R), 1e-6)

        delta = 1.0 / self.k_valve          # k_valve ≥ 1 (валидировано в __init__)
        q_raw = (float(dP) + np.hypot(float(dP), delta) - delta) / (2.0 * R_safe)
        return float(q_raw) if q_raw > 0.0 else 0.0

    # ------------------------------------------------------------------
    # Мягкий клип объёма у нижней границы
    # ------------------------------------------------------------------
    def _soft_clamp(self, V, V_min, dV):
        """
        Плавно обнуляет dV у нижней границы V_min.

        При V > 1.5·V_min        — возвращает dV без изменений.
        При V_min < V ≤ 1.5·V_min — плавное затухание: softness ∈ (0, 1).
        При V ≤ V_min             — softness = 0, dV вниз обнуляется.

        Численно устойчивая реализация: аргумент экспоненты клиппится
        в [-60, 60], а softness — в [0, 1].
        """
        if V > 1.5 * V_min:
            return dV

        if dV >= 0.0:
            return dV

        z = -10.0 * (V - V_min) / V_min
        z = float(np.clip(z, -60.0, 60.0))
        softness = 1.0 - float(np.exp(z))
        softness = float(np.clip(softness, 0.0, 1.0))

        return dV * softness

    # ------------------------------------------------------------------
    # Основной метод — производные
    # ------------------------------------------------------------------
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

        # Гладкие односторонние потоки через 4 клапана (C¹)
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