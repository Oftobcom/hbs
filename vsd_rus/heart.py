# heart.py
import numpy as np
from organ_base import OrganModel


class Heart4Chambers(OrganModel):
    """
    Четырёхкамерная модель сердца с клапанами.
    Поддерживает дефект межжелудочковой перегородки (VSD) через параметр R_vsd.

    Состояние: [V_la, V_lv, V_ra, V_rv, phi]

    4 клапана односторонние (smooth ReLU, v(0)=0):
        mitral, aortic, tricuspid, pulmonary.
    VSD — двунаправленный линейный резистор (отверстие, не клапан):
        Q_vsd > 0 — L→R, Q_vsd < 0 — R→L.

    Единицы:
        P    — мм рт.ст.
        V    — мл
        Q    — мл/с
        R    — мм рт.ст.·с/мл
        E    — мм рт.ст./мл
        HR   — уд/мин
    """

    # --- Санити-пороги параметров конфигурации ---
    _K_VALVE_MIN, _K_VALVE_MAX = 1.0, 100.0
    _R_VALVE_MIN, _R_VALVE_MAX = 1e-3, 10.0
    _R_VENOUS_MIN, _R_VENOUS_MAX = 1e-3, 10.0
    _HR_MIN, _HR_MAX = 20.0, 250.0
    _E_MIN_LO, _E_MIN_HI = 0.0, 1.0
    _E_MAX_LO, _E_MAX_HI = 0.01, 20.0
    _RV_CAP_BONUS = 20.0
    _V0_MIN, _V0_MAX = 0.0, 100.0
    _EDV_MIN, _EDV_MAX = 10.0, 500.0

    # --- Санити-границы runtime-факторов (защита от барорефлексных выбросов) ---
    _HR_FACTOR_MIN, _HR_FACTOR_MAX = 0.2, 3.0
    _INOTROPY_MIN,  _INOTROPY_MAX  = 0.5, 2.0
    _BARO_MIN,      _BARO_MAX      = 0.5, 2.0

    def __init__(self,
                 hr=70,
                 E_max_la=0.25, E_min_la=0.09,
                 E_max_ra=0.20, E_min_ra=0.05,
                 E_max_lv=3.5,  E_min_lv=0.03,
                 E_max_rv=0.8,  E_min_rv=0.02,
                 V0_la=10, V0_lv=10, V0_ra=5, V0_rv=10,
                 EDV_la=80.0, EDV_lv=120.0, EDV_ra=40.0, EDV_rv=120.0,
                 R_mitral=0.02, R_aortic=0.10,
                 R_tricuspid=0.01, R_pulmonary=0.05,
                 R_venous_sys=0.04,
                 R_venous_pulm=0.03,
                 R_vsd=np.inf,
                 hr_min=30, hr_max=130,
                 k_valve=9.0,
                 rv_hypertrophy_sensitivity: float = 1.5,
                 # --- Асимметрия симпатика/парасимпатика для E_max ---
                 k_lv_sympathetic: float = 1.0,
                 k_lv_parasympathetic: float = 0.3,
                 k_rv_sympathetic: float = 1.0,
                 k_rv_parasympathetic: float = 0.2,
                 k_atria_inotropy: float = 0.2):

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

        self.rv_hypertrophy_sensitivity = _check_range(
            "rv_hypertrophy_sensitivity", rv_hypertrophy_sensitivity,
            0.0, 5.0,
            "безразмерный, типично 1.0–2.0 — прирост E_max_rv "
            "при полном ремоделировании лёгких."
        )
        self._rv_afterload = 0.0
        # --- Асимметрия симпатика/парасимпатика ---
        # Симпатический отклик ЛЖ/ПЖ сильнее парасимпатического в 3–5 раз.
        # k_* ∈ [0, 5]: физиологически 0.2–1.5, запас на калибровку.
        self.k_lv_sympathetic = _check_range(
            "k_lv_sympathetic", k_lv_sympathetic, 0.0, 5.0,
            "безразмерный, типично 1.0 — усиление E_max_lv при гипотензии."
        )
        self.k_lv_parasympathetic = _check_range(
            "k_lv_parasympathetic", k_lv_parasympathetic, 0.0, 5.0,
            "безразмерный, типично 0.3 — ослабление E_max_lv при гипертензии."
        )
        self.k_rv_sympathetic = _check_range(
            "k_rv_sympathetic", k_rv_sympathetic, 0.0, 5.0,
            "безразмерный, типично 1.0 — усиление E_max_rv при гипотензии."
        )
        self.k_rv_parasympathetic = _check_range(
            "k_rv_parasympathetic", k_rv_parasympathetic, 0.0, 5.0,
            "безразмерный, типично 0.2 — ослабление E_max_rv при гипертензии."
        )
        self.k_atria_inotropy = _check_range(
            "k_atria_inotropy", k_atria_inotropy, 0.0, 5.0,
            "безразмерный, типично 0.2 — слабый отклик предсердий."
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
            E_min[chamber] = _check_range(
                f"E_min_{chamber.lower()}", val, self._E_MIN_LO, self._E_MIN_HI,
                "типично 0.02–0.10 мм рт.ст./мл."
            )
        for chamber, val in E_max.items():
            E_max[chamber] = _check_range(
                f"E_max_{chamber.lower()}", val, self._E_MAX_LO, self._E_MAX_HI,
                "LV ~3.5, RV ~0.8, LA ~0.25, RA ~0.20 мм рт.ст./мл."
            )
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

        # Начальная фаза кардиоцикла ∈ [0, 1). phi0 = 0 — начало систолы
        # желудочков: E(0) = E_min, объёмы камер = EDV, что согласовано
        # с get_initial_state().
        self.phi0 = 0.0

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
    def get_state_size(self):
        return 5   # [V_la, V_lv, V_ra, V_rv, phi]

    def get_initial_state(self) -> np.ndarray:
        """
        Начальное состояние — физиологические конечно-диастолические
        объёмы (мл) + начальная фаза кардиоцикла.
        """
        return np.array([
            self.EDV['LA'], self.EDV['LV'], self.EDV['RA'], self.EDV['RV'],
            self.phi0,
        ])


    # ------------------------------------------------------------------
    # Асимметрия симпатика/парасимпатика
    # ------------------------------------------------------------------
    def _asymmetric_inotropy(self, baro: float,
                             k_sym: float, k_para: float) -> float:
        """
        Асимметричный отклик E_max на бароактивацию.

        При baro > 1.0 (симпатика) множитель растёт с коэффициентом k_sym,
        при baro < 1.0 (парасимпатика) — падает с k_para.

        Физиология: парасимпатическая иннервация желудочков скудная,
        поэтому k_para << k_sym. При baro = 1.0 (норма) множитель = 1.0.
        """
        if baro >= 1.0:
            return 1.0 + k_sym * (baro - 1.0)
        return 1.0 + k_para * (baro - 1.0)


    # ------------------------------------------------------------------
    # Обновление параметров (HR, инотропия, бароактивация)
    # ------------------------------------------------------------------
    def _update_parameters(self, inputs):
        """
        Обновляет HR и E_max по runtime-факторам.

        Runtime-факторы приходят из барорефлекса и могут теоретически
        выйти за физиологический диапазон при численных сбоях.
        """
        hr_factor = float(inputs.get('hr_factor', 1.0))
        hr_factor = float(np.clip(hr_factor, self._HR_FACTOR_MIN, self._HR_FACTOR_MAX))
        new_hr = self.hr_base * hr_factor
        new_hr = np.clip(new_hr, self.hr_min, self.hr_max)
        self._current_hr = new_hr
        self._current_T = 60.0 / new_hr

        inotropy_factor = float(inputs.get('inotropy_factor', 1.0))
        inotropy_factor = float(np.clip(inotropy_factor,
                                        self._INOTROPY_MIN, self._INOTROPY_MAX))

        # Симпатическая активация: при падении P_sa растёт E_max
        # через барорефлексный сигнал
        baro_activation = float(inputs.get('baro_activation', 1.0))
        baro_activation = float(np.clip(baro_activation,
                                        self._BARO_MIN, self._BARO_MAX))

        # Пульмональный барорефлекс — обособленный сигнал для RV.
        # Fallback на системный, если baroreflex.py ещё не обновлён.
        baro_activation_rv = float(inputs.get(
            'baro_activation_rv', baro_activation
        ))
        baro_activation_rv = float(np.clip(
            baro_activation_rv, self._BARO_MIN, self._BARO_MAX
        ))

        # Постнагрузка ПЖ (0 = норма, 1 = полное ремоделирование лёгких).
        # Приходит из whole_body.py, где вычисляется из R_remodel лёгких.
        rv_afterload = float(inputs.get('rv_afterload', 0.0))
        rv_afterload = float(np.clip(rv_afterload, 0.0, 5.0))
        self._rv_afterload = rv_afterload

        # Гипертрофия ПЖ от хронической лёгочной гипертензии.
        rv_hypertrophy = 1.0 + self.rv_hypertrophy_sensitivity * rv_afterload

        for chamber in self.E_max_base:
            factor = 1.0
            cap = self._E_MAX_HI            # дефолт на случай новой камеры

            if chamber == 'LV':
                factor = self._asymmetric_inotropy(
                    baro_activation,
                    self.k_lv_sympathetic,
                    self.k_lv_parasympathetic,
                )
            elif chamber == 'RV':
                factor = self._asymmetric_inotropy(
                    baro_activation_rv,
                    self.k_rv_sympathetic,
                    self.k_rv_parasympathetic,
                ) * rv_hypertrophy
                cap = self._E_MAX_HI + self._RV_CAP_BONUS * rv_afterload
            elif chamber in ('LA', 'RA'):
                factor = 1.0 + self.k_atria_inotropy * (baro_activation - 1.0)

            e_new = self.E_max_base[chamber] * inotropy_factor * factor
            self._current_E_max[chamber] = float(
                np.clip(e_new, self._E_MAX_LO, cap)
            )

    # ------------------------------------------------------------------
    # Эластанс-профиль цикла
    # ------------------------------------------------------------------
    def _elastance(self, phi, chamber):
        # phi — накопленное число циклов (state[4]), может быть > 1.
        # Свёртка в [0, 1) периодическая; профиль e(τ) в точках τ=0 и
        # τ=1 стыкуется с e=0, de/dτ=0 (C¹ на стыке циклов).
        tau = float(phi) % 1.0
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
        if chamber == 'RV':
            # Дилатация/гипертрофия ПЖ сдвигает пик позже и удлиняет изгнание:
            # возникает окно, где ПЖ ещё сокращён, а ЛЖ уже расслаблен.
            rv_al = min(float(getattr(self, '_rv_afterload', 0.0)), 2.0)
            T_PEAK = 0.33 + 0.03 * rv_al    # 0.33 → 0.39
            T_END  = 0.45 + 0.05 * rv_al    # 0.45 → 0.55
        else:
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
    def _pressure(self, chamber, V, phi):
        dV = max(V - self.V0[chamber], 0.0)
        E = self._elastance(phi, chamber)

        if chamber in ('LA', 'RA'):
            A = 0.4 if chamber == 'LA' else 0.3
            k = 0.025 if chamber == 'LA' else 0.02
            exp_arg = float(np.clip(k * dV, 0.0, 8.0))
            P_passive = A * (np.exp(exp_arg) - 1.0)
            return E * dV + P_passive

        # Желудочки — линейная + пассивная экспонента
        if chamber == 'LV':
            A_v, k_v = 0.03, 0.02
        else:  # RV
            # При дилатации/гипертрофии стенка ПЖ становится жёстче.
            rv_al = min(float(getattr(self, '_rv_afterload', 0.0)), 2.0)
            stiffness = 1.0 + 0.4 * rv_al
            A_v = 0.02 * stiffness
            k_v = 0.015 * stiffness
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
    def _floor_factor(self, V, V_min):
        z = -10.0 * (V - V_min) / V_min
        z = float(np.clip(z, -60.0, 60.0))
        return float(np.clip(1.0 - np.exp(z), 0.0, 1.0))

    def _ceil_factor(self, V, V_max):
        z = -10.0 * (V_max - V) / V_max
        z = float(np.clip(z, -60.0, 60.0))
        return float(np.clip(1.0 - np.exp(z), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Основной метод — производные
    # ------------------------------------------------------------------
    def get_derivatives(self, t, state, inputs):
        V_la, V_lv, V_ra, V_rv, phi = state
        self._update_parameters(inputs)

        dphi = 1.0 / self._current_T

        P_sa = inputs['P_sa']
        P_sv = inputs['P_sv']
        P_pa = inputs['P_pa']
        P_pv = inputs['P_pv']

        P_la = self._pressure('LA', V_la, phi)
        P_lv = self._pressure('LV', V_lv, phi)
        P_ra = self._pressure('RA', V_ra, phi)
        P_rv = self._pressure('RV', V_rv, phi)

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

        Q_sv_to_ra = self._valve_flow(P_sv - P_ra, self.R_venous_sys)
        V_sv_in  = inputs.get('V_sv', None)
        V0_sv_in = inputs.get('V0_sv', None)
        if V_sv_in is not None and V0_sv_in is not None and V0_sv_in > 0.0:
            V_sv_in = max(float(V_sv_in), 0.0)
            if V_sv_in < 0.5 * V0_sv_in:
                factor = float(np.clip(V_sv_in / (0.5 * V0_sv_in), 0.0, 1.0))
                Q_sv_to_ra *= factor
        Q_pv_to_la = self._valve_flow(P_pv - P_la, self.R_venous_pulm)

        rv_al = min(float(getattr(self, '_rv_afterload', 0.0)), 2.0)
        V_min_rv_eff = self.V0['RV'] * max(0.3, 1.0 - 0.3 * rv_al)
        V_max_rv_eff = 250.0 * (1.0 + 0.5 * rv_al)

        s_la = self._floor_factor(V_la, self.V0['LA'])
        s_lv = self._floor_factor(V_lv, self.V0['LV'])
        s_ra = self._floor_factor(V_ra, self.V0['RA'])
        s_rv = self._floor_factor(V_rv, V_min_rv_eff)
        c_rv = self._ceil_factor(V_rv, V_max_rv_eff)

        Q_mitral    *= s_la
        Q_aortic    *= s_lv
        Q_tricuspid *= s_ra * c_rv
        Q_pulmonary *= s_rv

        if Q_vsd > 0:
            Q_vsd *= s_lv * c_rv
        else:
            Q_vsd *= s_rv

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
            # --- E_max по камерам (для валидации Цели 3) ---
            'E_max_lv': float(self._current_E_max['LV']),
            'E_max_rv': float(self._current_E_max['RV']),
            'E_max_la': float(self._current_E_max['LA']),
            'E_max_ra': float(self._current_E_max['RA']),
            # --- HR и период текущего шага ---
            'HR_current': float(self._current_hr),
            'T_current':  float(self._current_T),
            # --- Фаза кардиоцикла ---
            # накопленное число циклов, монотонно
            'phi_raw': float(phi),
            # текущая фаза в [0, 1)
            'phi':     float(phi % 1.0),
            # --- rv_afterload (пробрасывается из whole_body) ---
            'rv_afterload': float(self._rv_afterload),
        }

        dV_la = Q_pv_to_la - Q_mitral
        dV_lv = Q_mitral - Q_aortic - Q_vsd
        dV_ra = Q_sv_to_ra - Q_tricuspid
        dV_rv = Q_tricuspid - Q_pulmonary + Q_vsd

        return np.array([dV_la, dV_lv, dV_ra, dV_rv, dphi])

    def get_outputs(self, state):
        return self._current_flows.copy()