# brain.py
"""
brain.py

Модель мозга: гемодинамика, ауторегуляция, метаболизм O2/CO2, лактат, аммиак.

Рапортует O₂/CO₂ в whole_body; в кровь напрямую не пишет (кроме
dC_lactate_blood и dC_ammonia_blood, которые учитывает whole_body).

Состояния: [P_br, C_O2_tis, C_CO2_tis, C_lac_tis, C_amm_tis] — 5 состояний.

Единицы:
    P            — мм рт.ст.
    Q            — мл/с
    C_*          — мл газа / мл крови (или мг/мл для лактата/аммиака)
    R            — мм рт.ст.·с/мл
    V_tissue     — мл
    CMRO2_target — мл O2/с
"""

import numpy as np
from organ_base import OrganModel


class Brain(OrganModel):
    """
    Модель мозга здорового человека.

    Гемодинамика:
        R_eff = R_base · f_P(myogenic) · f_O2(hypoxic) · f_CO2(hypercapnic)
        Q_br  = max((P_sa − P_br)/R_eff, 0) · occlusion_factor
        Q_out = max((P_br − P_sv)/R_eff, 0) · occlusion_factor
        dP_br = (Q_br − Q_out)/C

    Метаболизм:
        O2_cons = Q_br · extraction_used · inhibition
        CO2_prod = O2_cons_eff · RQ
        dC_O2 = (Q_br·(C_a_O2 − C_O2_tis) − O2_cons_eff) / V_tissue
        dC_CO2 = (Q_br·(C_a_CO2 − C_CO2_tis) + CO2_prod) / V_tissue

    Лактат и аммиак — по формулам кинетики первого порядка
    с базальной продукцией, компенсированной при нормоксии/нормоаммониемии.
    """

    # --- Санити-пороги для валидации конфигурации ---
    _R_BASE_MIN, _R_BASE_MAX = 0.5, 50.0
    _C_MIN, _C_MAX = 0.1, 50.0
    _P_AUTOREG_MIN, _P_AUTOREG_MAX = 30.0, 200.0
    _CMRO2_MIN, _CMRO2_MAX = 0.1, 5.0

    _C_V_MIN_MIN, _C_V_MIN_MAX = 0.01, 0.15
    _MAX_EXTRACTION_MIN, _MAX_EXTRACTION_MAX = 0.1, 1.0
    _GLUCOSE_EXTR_MIN, _GLUCOSE_EXTR_MAX = 0.0, 1.0
    _P0_MIN, _P0_MAX = 10.0, 100.0

    _C_A_O2_NORM_MIN, _C_A_O2_NORM_MAX = 0.05, 0.25
    _C_A_CO2_NORM_MIN, _C_A_CO2_NORM_MAX = 0.20, 1.00

    _K_HYPOXIC_MIN, _K_HYPOXIC_MAX = 0.0, 3.0
    _K_HYPERCAPNIC_MIN, _K_HYPERCAPNIC_MAX = 0.0, 5.0
    _K_MYO_MIN, _K_MYO_MAX = 0.0, 2.0

    _V_TISSUE_MIN, _V_TISSUE_MAX = 10.0, 5000.0
    _RQ_MIN, _RQ_MAX = 0.5, 1.5
    _C_O2_CRITICAL_MIN, _C_O2_CRITICAL_MAX = 0.01, 0.15

    _C_LAC_NORM_MIN, _C_LAC_NORM_MAX = 0.0, 1.0
    _K_LAC_PROD_MIN, _K_LAC_PROD_MAX = 0.0, 1.0
    _K_LAC_CLEAR_MIN, _K_LAC_CLEAR_MAX = 0.0, 1.0
    _K_LAC_RELEASE_MIN, _K_LAC_RELEASE_MAX = 0.0, 1.0

    _C_AMM_NORM_MIN, _C_AMM_NORM_MAX = 0.0, 2.0
    _K_AMM_BBB_IN_MIN, _K_AMM_BBB_IN_MAX = 0.0, 1.0
    _K_AMM_BBB_OUT_MIN, _K_AMM_BBB_OUT_MAX = 0.0, 1.0
    _K_AMM_DETOX_MIN, _K_AMM_DETOX_MAX = 0.0, 1.0
    _K_AMM_INHIB_MIN, _K_AMM_INHIB_MAX = 0.0, 2.0

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
                 C_lac_norm=0.15, k_lac_prod=0.08, k_lac_clear=0.02, k_lac_release=0.03,
                 # Аммиак BBB
                 C_amm_norm=0.3, k_amm_bbb_in=0.02, k_amm_bbb_out=0.01,
                 k_amm_detox=0.01, k_amm_inhibition=0.15):

        # =================================================================
        # Валидация конфигурации — fail-fast при инициализации.
        # Параметры приходят из YAML и не меняются во время симуляции;
        # ошибки в них должны ловиться один раз, а не в горячем пути RHS.
        # =================================================================
        def _check_range(name, v, lo, hi, typical=""):
            v = float(v)
            if not np.isfinite(v) or not (lo <= v <= hi):
                raise ValueError(
                    f"Brain: {name}={v} вне [{lo}, {hi}]. {typical}"
                )
            return v

        # --- Гемодинамика ---
        self.R_base = _check_range(
            "R_base", R_base, self._R_BASE_MIN, self._R_BASE_MAX,
            "мм рт.ст.·с/мл, типично 7."
        )
        self.C = _check_range(
            "C", C, self._C_MIN, self._C_MAX,
            "мл/мм рт.ст., типично 4."
        )
        self.P_autoreg = _check_range(
            "P_autoreg", P_autoreg, self._P_AUTOREG_MIN, self._P_AUTOREG_MAX,
            "мм рт.ст., типично 80 — точка якоря ауторегуляции."
        )

        # --- Метаболизм O2 ---
        self.CMRO2_target = _check_range(
            "CMRO2_target", CMRO2_target,
            self._CMRO2_MIN, self._CMRO2_MAX,
            "мл O2/с, типично 0.55 (≈50 мл/мин, ~20% всего VO2)."
        )
        self.C_v_min = _check_range(
            "C_v_min", C_v_min,
            self._C_V_MIN_MIN, self._C_V_MIN_MAX,
            "мл O2/мл, типично 0.06 — минимальная венозная O2 мозга."
        )
        self.max_extraction = _check_range(
            "max_extraction", max_extraction,
            self._MAX_EXTRACTION_MIN, self._MAX_EXTRACTION_MAX,
            "доля, типично 0.60 — максимальная экстракция O2 из крови."
        )
        self.glucose_extraction = _check_range(
            "glucose_extraction", glucose_extraction,
            self._GLUCOSE_EXTR_MIN, self._GLUCOSE_EXTR_MAX,
            "доля, типично 0.1."
        )

        # --- Начальное давление и норм. концентрации ---
        self.P0 = _check_range(
            "P0", P0, self._P0_MIN, self._P0_MAX,
            "мм рт.ст., типично 47.5 — ICP + венозное."
        )
        self.C_a_O2_norm = _check_range(
            "C_a_O2_norm", C_a_O2_norm,
            self._C_A_O2_NORM_MIN, self._C_A_O2_NORM_MAX,
            "мл O2/мл, типично 0.20 — норм. артериальная O2."
        )
        self.C_a_CO2_norm = _check_range(
            "C_a_CO2_norm", C_a_CO2_norm,
            self._C_A_CO2_NORM_MIN, self._C_A_CO2_NORM_MAX,
            "мл CO2/мл, типично 0.50 — норм. артериальная CO2."
        )

        # --- Коэффициенты ауторегуляции ---
        self.k_hypoxic_dilation = _check_range(
            "k_hypoxic_dilation", k_hypoxic_dilation,
            self._K_HYPOXIC_MIN, self._K_HYPOXIC_MAX,
            "безразмерный, типично 0.6 — вазодилатация при гипоксии."
        )
        self.k_hypercapnic_dilation = _check_range(
            "k_hypercapnic_dilation", k_hypercapnic_dilation,
            self._K_HYPERCAPNIC_MIN, self._K_HYPERCAPNIC_MAX,
            "безразмерный, типично 1.2 — вазодилатация при гиперкапнии."
        )
        self.k_myo = _check_range(
            "k_myo", k_myo, self._K_MYO_MIN, self._K_MYO_MAX,
            "безразмерный, типично 0.30 — миогенная ауторегуляция."
        )

        # --- Тканевые параметры ---
        self.V_tissue = _check_range(
            "V_tissue", V_tissue,
            self._V_TISSUE_MIN, self._V_TISSUE_MAX,
            "мл, типично 150 — эффективный объём мозга."
        )
        self.RQ = _check_range(
            "RQ", RQ, self._RQ_MIN, self._RQ_MAX,
            "безразмерный, типично 0.85 — дыхательный коэффициент мозга."
        )
        self.C_O2_critical = _check_range(
            "C_O2_critical", C_O2_critical,
            self._C_O2_CRITICAL_MIN, self._C_O2_CRITICAL_MAX,
            "мл O2/мл, типично 0.08 — порог ишемии."
        )

        # --- Лактат ---
        self.C_lac_norm = _check_range(
            "C_lac_norm", C_lac_norm,
            self._C_LAC_NORM_MIN, self._C_LAC_NORM_MAX,
            "мг/мл, типично 0.15."
        )
        self.k_lac_prod = _check_range(
            "k_lac_prod", k_lac_prod,
            self._K_LAC_PROD_MIN, self._K_LAC_PROD_MAX,
            "1/с, типично 0.08."
        )
        self.k_lac_clear = _check_range(
            "k_lac_clear", k_lac_clear,
            self._K_LAC_CLEAR_MIN, self._K_LAC_CLEAR_MAX,
            "1/с, типично 0.02."
        )
        self.k_lac_release = _check_range(
            "k_lac_release", k_lac_release,
            self._K_LAC_RELEASE_MIN, self._K_LAC_RELEASE_MAX,
            "1/с, типично 0.03."
        )

        # --- Аммиак (BBB) ---
        self.C_amm_norm = _check_range(
            "C_amm_norm", C_amm_norm,
            self._C_AMM_NORM_MIN, self._C_AMM_NORM_MAX,
            "мг/мл, типично 0.3."
        )
        self.k_amm_bbb_in = _check_range(
            "k_amm_bbb_in", k_amm_bbb_in,
            self._K_AMM_BBB_IN_MIN, self._K_AMM_BBB_IN_MAX,
            "1/с, типично 0.02 — приток через BBB."
        )
        self.k_amm_bbb_out = _check_range(
            "k_amm_bbb_out", k_amm_bbb_out,
            self._K_AMM_BBB_OUT_MIN, self._K_AMM_BBB_OUT_MAX,
            "1/с, типично 0.01 — отток через BBB."
        )
        self.k_amm_detox = _check_range(
            "k_amm_detox", k_amm_detox,
            self._K_AMM_DETOX_MIN, self._K_AMM_DETOX_MAX,
            "1/с, типично 0.01 — детоксикация в мозге."
        )
        self.k_amm_inhibition = _check_range(
            "k_amm_inhibition", k_amm_inhibition,
            self._K_AMM_INHIB_MIN, self._K_AMM_INHIB_MAX,
            "безразмерный, типично 0.15 — ингибирование метаболизма."
        )

        self._current_outputs = {}

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Ауторегуляция: R_eff и диагностические факторы
    # ------------------------------------------------------------------
    def _autoregulation_resistance(self, P_sa, C_a_O2,
                                   C_a_CO2=None, C_tissue_CO2=None):
        x = (P_sa - self.P_autoreg) / self.P_autoreg
        f_P = 1.0 + self.k_myo * np.tanh(x)

        hypoxia = max(self.C_a_O2_norm - C_a_O2, 0.0) / self.C_a_O2_norm
        f_O2 = 1.0 - self.k_hypoxic_dilation * hypoxia

        if C_a_CO2 is None:
            C_a_CO2 = self.C_a_CO2_norm
        hypercapnia = (C_a_CO2 - self.C_a_CO2_norm) / self.C_a_CO2_norm
        if C_tissue_CO2 is not None:
            hypercapnia = max(
                hypercapnia,
                (C_tissue_CO2 - self.C_a_CO2_norm) / self.C_a_CO2_norm,
            )
        f_CO2 = 1.0 - self.k_hypercapnic_dilation * hypercapnia

        reg = f_P * f_O2 * f_CO2
        reg = np.clip(reg, 0.35, 2.5)
        return self.R_base * reg, {'f_P': f_P, 'f_O2': f_O2, 'f_CO2': f_CO2}

    # ------------------------------------------------------------------
    # Основной метод — производные
    # ------------------------------------------------------------------
    def get_derivatives(self, t, state, inputs):
        # --- Разбор состояния с мягкими клипами (защита от LSODA retries) ---
        P_br        = float(state[0])
        C_O2_tis    = max(float(state[1]), 0.0)
        C_CO2_tis   = max(float(state[2]), 0.0)
        C_lac_tis   = max(float(state[3]), 0.0)
        C_amm_tis   = max(float(state[4]), 0.0)

        # --- Входы ---
        P_sa      = float(inputs.get('P_sa', 80.0))
        P_sv      = float(inputs.get('P_sv', 5.0))
        C_a_O2    = float(inputs.get('C_a_O2', self.C_a_O2_norm))
        C_a_CO2   = float(inputs.get('C_a_CO2', self.C_a_CO2_norm))
        C_a_lac   = float(inputs.get('C_lactate_blood',
                                     inputs.get('C_a_lactate', 0.1)))
        C_a_amm   = float(inputs.get('C_ammonia',
                                     inputs.get('C_a_ammonia', self.C_amm_norm)))
        V_blood   = float(inputs.get('V_blood', 5800.0))
        V_blood   = max(V_blood, 1e-6)

        occlusion = float(inputs.get('occlusion_factor', 1.0))
        occlusion = float(np.clip(occlusion, 0.0, 1.0))

        # --- Гемодинамика ---
        R_eff, f_autoreg = self._autoregulation_resistance(
            P_sa, C_a_O2, C_a_CO2, C_CO2_tis
        )
        Q_br_healthy = max((P_sa - P_br) / R_eff, 0.0)
        Q_br = Q_br_healthy * occlusion
        Q_out = max((P_br - P_sv) / R_eff, 0.0) * occlusion
        dP_br = (Q_br - Q_out) / self.C

        # --- O2 метаболизм ---
        if Q_br > 1e-6:
            extraction_needed = self.CMRO2_target / Q_br
        else:
            extraction_needed = self.max_extraction
        extraction_avail = max(C_a_O2 - self.C_v_min, 0.0)
        extraction_used = min(extraction_needed, extraction_avail,
                              self.max_extraction)
        O2_cons = Q_br * extraction_used

        # Ишемия + гипераммониемия → ингибирование метаболизма
        if C_O2_tis < self.C_O2_critical:
            inhib_O2 = np.clip(C_O2_tis / self.C_O2_critical, 0.2, 1.0)
        else:
            inhib_O2 = 1.0
        inhib_amm = 1.0 / (
            1.0 + self.k_amm_inhibition * max(C_amm_tis - self.C_amm_norm, 0.0)
        )
        inhibition = inhib_O2 * inhib_amm

        O2_cons_eff = O2_cons * inhibition
        C_v_O2_brain = max(C_a_O2 - extraction_used, self.C_v_min)

        # --- CO2 ---
        CO2_prod = O2_cons_eff * self.RQ
        C_v_CO2_brain = C_a_CO2 + (CO2_prod / Q_br if Q_br > 1e-6 else 0.04)

        # --- Тканевые O2/CO2 ---
        dC_O2  = (Q_br * (C_a_O2  - C_O2_tis)  - O2_cons_eff) / self.V_tissue
        dC_CO2 = (Q_br * (C_a_CO2 - C_CO2_tis) + CO2_prod)    / self.V_tissue

        # --- Лактат ---
        # Базальная продукция подобрана так, чтобы при нормоксии
        # (hypoxia_sev=0) держать dC_lac = 0 при C_lac = C_lac_norm.
        # Компенсирует И клиренс, И релиз в кровь.
        hypoxia_sev = max(self.C_O2_critical - C_O2_tis, 0.0) / self.C_O2_critical
        amm_excess = max(C_amm_tis - self.C_amm_norm, 0.0)

        lac_prod_base = (
            self.k_lac_clear * self.C_lac_norm
            + self.k_lac_release * max(self.C_lac_norm - C_a_lac, 0.0)
        )
        lac_prod_hypoxic = self.k_lac_prod * hypoxia_sev * (1.0 + 0.5 * amm_excess)
        lac_prod = lac_prod_base + lac_prod_hypoxic
        lac_clear = self.k_lac_clear * C_lac_tis
        lac_release = self.k_lac_release * max(C_lac_tis - C_a_lac, 0.0)
        dC_lac = lac_prod - lac_clear - lac_release
        C_v_lac_brain = C_lac_tis   # венозный лактат ≈ тканевой

        # --- Аммиак BBB ---
        amm_in = self.k_amm_bbb_in * max(C_a_amm - C_amm_tis, 0.0)
        amm_out = self.k_amm_bbb_out * max(C_amm_tis - C_a_amm, 0.0)
        amm_detox = self.k_amm_detox * C_amm_tis
        dC_amm = amm_in - amm_out - amm_detox
        C_v_amm_brain = C_amm_tis

        # --- Вклады в кровь ---
        dC_lac_blood = (lac_release * self.V_tissue) / V_blood
        dC_amm_blood = -dC_amm * self.V_tissue / V_blood

        # --- Кэш выходов ---
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