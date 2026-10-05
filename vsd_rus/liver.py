# liver.py
import numpy as np
from organ_base import OrganModel


class Liver(OrganModel):
    """
    Модель печени здорового человека.

    Гемодинамика: двухкомпартментный Windkessel
        P_hv     — давление в печёночной вене
        P_portal — давление в портальной вене

    Метаболизм:
        C_bilirubin — концентрация билирубина в печени
        C_ammonia   — концентрация аммиака в печени
        C_albumin   — концентрация альбумина в печени
        reserve     — резерв (задел, всегда 0)
        V_liver     — эффективный объём распределения метаболитов (мл),
                используется для перевода тканевой скорости
                [мг/(мл·с)] в вклад в кровь [мг/(мл·с)].

    Состояние: [P_hv, C_bilirubin, C_ammonia, C_albumin, reserve, P_portal]

    Единицы:
        P          — мм рт.ст.
        Q          — мл/с
        R          — мм рт.ст.·с/мл
        C          — мл/мм рт.ст.
        Концентрации — мг/мл
    """

    # --- Санити-пороги для валидации конфигурации ---
    _R_HA_MIN, _R_HA_MAX = 1.0, 100.0
    _R_PV_MIN, _R_PV_MAX = 0.01, 10.0
    _R_HV_MIN, _R_HV_MAX = 0.01, 10.0
    _C_MIN, _C_MAX = 0.1, 200.0
    _C_PORTAL_MIN, _C_PORTAL_MAX = 0.1, 100.0

    _P_HV0_MIN, _P_HV0_MAX = 0.0, 50.0
    _P_PORTAL0_MIN, _P_PORTAL0_MAX = 0.0, 50.0

    _ALB_PROD_MIN, _ALB_PROD_MAX = 0.0, 5.0
    _BIL_CLEAR_MIN, _BIL_CLEAR_MAX = 0.0, 10.0
    _AMM_CLEAR_MIN, _AMM_CLEAR_MAX = 0.0, 10.0
    _LAC_CLEAR_MIN, _LAC_CLEAR_MAX = 0.01, 1.0

    _C_BIL0_MIN, _C_BIL0_MAX = 0.0, 20.0
    _C_AMM0_MIN, _C_AMM0_MAX = 0.0, 20.0
    _C_ALB0_MIN, _C_ALB0_MAX = 0.0, 20.0

    _K_UPTAKE_BIL_MIN, _K_UPTAKE_BIL_MAX = 1e-4, 10.0
    _K_UPTAKE_AMM_MIN, _K_UPTAKE_AMM_MAX = 1e-4, 10.0
    _K_DEG_ALB_MIN, _K_DEG_ALB_MAX = 1e-6, 5.0
    _K_RELEASE_ALB_MIN, _K_RELEASE_ALB_MAX = 1e-6, 5.0
    _K_LAC_CLEAR_MIN, _K_LAC_CLEAR_MAX = 0.01, 100.0
    _V_LIVER_MIN, _V_LIVER_MAX = 100.0, 3000.0

    def __init__(self, *,
                R_ha: float,
                R_pv_base: float,
                R_hv_base: float,
                C: float,
                C_portal: float,
                P_hv0: float,
                P_portal0: float,
                V_liver: float,
                albumin_prod_base: float,
                bilirubin_clearance_base: float,
                ammonia_clearance_base: float,
                lactate_clearance_base: float,
                C_bilirubin0: float,
                C_ammonia0: float,
                C_albumin0: float,
                k_uptake_bil: float,
                k_uptake_amm: float,
                k_deg_alb: float,
                k_release_alb: float,
                k_lac_clear: float
                ):

        # =================================================================
        # Валидация конфигурации — fail-fast при инициализации.
        # Параметры приходят из YAML и не меняются во время симуляции;
        # ошибки в них должны ловиться один раз, а не в горячем пути RHS.
        # =================================================================
        def _check_range(name, v, lo, hi, typical=""):
            if v is None:
                raise ValueError(
                    f"Liver: {name} не задан (None). "
                    f"Все параметры обязательны; дефолты удалены. "
                    f"Задайте liver.{name} в physiology.yaml."
                )
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(
                    f"Liver: {name}={v!r} должен быть числом, "
                    f"получено {type(v).__name__}."
                )
            v = float(v)
            if not np.isfinite(v) or not (lo <= v <= hi):
                raise ValueError(
                    f"Liver: {name}={v} вне [{lo}, {hi}]. {typical}"
                )
            return v

        # --- Гемодинамические сопротивления ---
        self.R_ha = _check_range(
            "R_ha", R_ha, self._R_HA_MIN, self._R_HA_MAX,
            "мм рт.ст.·с/мл, типично 17 (hepatic artery)."
        )
        self.R_pv_base = _check_range(
            "R_pv_base", R_pv_base, self._R_PV_MIN, self._R_PV_MAX,
            "мм рт.ст.·с/мл, типично 0.25 (portal vein)."
        )
        self.R_hv_base = _check_range(
            "R_hv_base", R_hv_base, self._R_HV_MIN, self._R_HV_MAX,
            "мм рт.ст.·с/мл, типично 0.12 (hepatic vein)."
        )

        # --- Комплаенсы ---
        self.C = _check_range(
            "C", C, self._C_MIN, self._C_MAX,
            "мл/мм рт.ст., типично 5 (hepatic vein)."
        )
        self.C_portal = _check_range(
            "C_portal", C_portal, self._C_PORTAL_MIN, self._C_PORTAL_MAX,
            "мл/мм рт.ст., типично 1.5 (portal vein Windkessel)."
        )

        # --- Начальные давления ---
        self.P_hv0 = _check_range(
            "P_hv0", P_hv0, self._P_HV0_MIN, self._P_HV0_MAX,
            "мм рт.ст., типично 8."
        )
        self.P_portal0 = _check_range(
            "P_portal0", P_portal0, self._P_PORTAL0_MIN, self._P_PORTAL0_MAX,
            "мм рт.ст., типично 8."
        )
        self.V_liver = _check_range(
            "V_liver", V_liver, self._V_LIVER_MIN, self._V_LIVER_MAX,
            "мл, типично 1500 — эффективный объём распределения метаболитов."
        )

        # --- Скорости метаболизма ---
        self.albumin_prod_base = _check_range(
            "albumin_prod_base", albumin_prod_base,
            self._ALB_PROD_MIN, self._ALB_PROD_MAX,
            "мг/(мл·с), типично 0.1."
        )
        self.bilirubin_clearance_base = _check_range(
            "bilirubin_clearance_base", bilirubin_clearance_base,
            self._BIL_CLEAR_MIN, self._BIL_CLEAR_MAX,
            "1/с, типично 0.2."
        )
        self.ammonia_clearance_base = _check_range(
            "ammonia_clearance_base", ammonia_clearance_base,
            self._AMM_CLEAR_MIN, self._AMM_CLEAR_MAX,
            "1/с, типично 0.15."
        )
        self.lactate_clearance_base = _check_range(
            "lactate_clearance_base", lactate_clearance_base,
            self._LAC_CLEAR_MIN, self._LAC_CLEAR_MAX,
            "безразмерный, типично 0.05 — доля экстракции лактата за проход."
        )

        # --- Начальные концентрации в печени ---
        self.C_bilirubin0 = _check_range(
            "C_bilirubin0", C_bilirubin0,
            self._C_BIL0_MIN, self._C_BIL0_MAX,
            "мг/мл, типично 0.0."
        )
        self.C_ammonia0 = _check_range(
            "C_ammonia0", C_ammonia0,
            self._C_AMM0_MIN, self._C_AMM0_MAX,
            "мг/мл, типично 0.0."
        )
        self.C_albumin0 = _check_range(
            "C_albumin0", C_albumin0,
            self._C_ALB0_MIN, self._C_ALB0_MAX,
            "мг/мл, типично 1.0."
        )

        # --- Коэффициенты кинетики (не были валидированы ранее) ---
        self.k_uptake_bil = _check_range(
            "k_uptake_bil", k_uptake_bil,
            self._K_UPTAKE_BIL_MIN, self._K_UPTAKE_BIL_MAX,
            "1/с, типично 0.1."
        )
        self.k_uptake_amm = _check_range(
            "k_uptake_amm", k_uptake_amm,
            self._K_UPTAKE_AMM_MIN, self._K_UPTAKE_AMM_MAX,
            "1/с, типично 0.1."
        )
        self.k_deg_alb = _check_range(
            "k_deg_alb", k_deg_alb,
            self._K_DEG_ALB_MIN, self._K_DEG_ALB_MAX,
            "1/с, типично 0.01."
        )
        self.k_release_alb = _check_range(
            "k_release_alb", k_release_alb,
            self._K_RELEASE_ALB_MIN, self._K_RELEASE_ALB_MAX,
            "1/с, типично 0.05."
        )
        self.k_lac_clear = _check_range(
            "k_lac_clear", k_lac_clear,
            self._K_LAC_CLEAR_MIN, self._K_LAC_CLEAR_MAX,
            "безразмерный, типично 2.0."
        )

        self._current_outputs = {}

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
    def get_state_size(self):
        return 6   # [P_hv, C_bil, C_amm, C_alb, reserve, P_portal]

    def get_initial_state(self):
        return np.array([
            self.P_hv0,
            self.C_bilirubin0,
            self.C_ammonia0,
            self.C_albumin0,
            0.0,                     # reserve — не используется
            self.P_portal0,
        ])

    # ------------------------------------------------------------------
    # Основной метод — производные
    # ------------------------------------------------------------------
    def get_derivatives(self, t, state, inputs):
        P_hv = state[0]
        C_bil = state[1]
        C_amm = state[2]
        C_alb = state[3]
        # state[4] — reserve (не используется)
        P_portal = state[5]

        P_sa = inputs['P_sa']
        P_sv = inputs['P_sv']
        C_bil_blood = inputs.get('C_bilirubin_blood', 0.0)
        C_amm_blood = inputs.get('C_ammonia_blood', 0.0)
        C_alb_blood = inputs.get('C_albumin_blood', 4.5)
        C_lac_blood = inputs.get('C_lactate_blood', 0.10)
        V_blood = inputs.get('V_blood', 5000.0)
        V_blood = max(V_blood, 1e-6)   # защита от деления на ноль

        # Связь с ЖКТ — приходит из whole_body.py
        Q_gut_out = inputs.get('Q_gut_out', inputs.get('Q_portal', 8.0))

        # --- Гемодинамика ---
        Q_pv  = max((P_portal - P_hv) / self.R_pv_base, 0.0)
        Q_ha  = max((P_sa - P_hv) / self.R_ha, 0.0)
        Q_out = max((P_hv - P_sv) / self.R_hv_base, 0.0)

        dP_hv = (Q_ha + Q_pv - Q_out) / self.C
        dP_portal = (Q_gut_out - Q_pv) / self.C_portal   # Windkessel портальной вены

        # --- Метаболизм: билирубин ---
        # Кровь теряет uptake_bil (то, что печень ЗАБИРАЕТ),
        # а не clearance_bil (то, что печень МЕТАБОЛИЗИРУЕТ).
        # Это обеспечивает корректный масс-баланс в переходных процессах.
        uptake_bil = self.k_uptake_bil * (C_bil_blood - C_bil)
        clearance_bil = self.bilirubin_clearance_base * C_bil
        dC_bil = uptake_bil - clearance_bil

        # --- Метаболизм: аммиак ---
        uptake_amm = self.k_uptake_amm * (C_amm_blood - C_amm)
        clearance_amm = self.ammonia_clearance_base * C_amm
        dC_amm = uptake_amm - clearance_amm

        # --- Метаболизм: альбумин ---
        synthesis_alb = self.albumin_prod_base
        degradation_alb = self.k_deg_alb * C_alb
        release_alb = self.k_release_alb * (C_alb - C_alb_blood)
        dC_alb = synthesis_alb - degradation_alb - release_alb

        # --- Вклады в BloodPool ---
        # uptake_* и release_* — скорости изменения ТКАНЕВОЙ концентрации
        # [мг/(мл·с)]. Масса, обмениваемая с кровью за секунду, = скорость·V_liver
        # [мг/с]. Деление на V_blood даёт скорость изменения концентрации в
        # крови [мг/(мл·с)]. Ср. brain.py: dC_amm_blood = (amm_out−amm_in)·V_tissue/V_blood.
        dC_bil_blood = -uptake_bil * self.V_liver / V_blood
        dC_amm_blood = -uptake_amm * self.V_liver / V_blood
        dC_alb_blood = +release_alb * self.V_liver / V_blood

        # Печёночный клиренс лактата — flow-limited. E_lac — безразмерная
        # доля экстракции за проход, ограниченная 0.95 (нельзя извлечь всю
        # лактатную нагрузку за один пасс).
        E_lac = float(np.clip(
            self.lactate_clearance_base * self.k_lac_clear, 0.0, 0.95
        ))
        dC_lac_blood = -Q_ha * E_lac * C_lac_blood / V_blood

        self._current_outputs = {
            'Q_liver_out': Q_out,
            'P_portal': P_portal,
            'dC_bilirubin': dC_bil_blood,
            'dC_ammonia': dC_amm_blood,
            'dC_albumin': dC_alb_blood,
            'dC_lactate': dC_lac_blood,
            'Q_ha': Q_ha,
            'Q_pv': Q_pv,
            'Q_gut_out': Q_gut_out,
            'functional': 1.0,
            'E_lac': float(E_lac),
        }

        return np.array([dP_hv, dC_bil, dC_amm, dC_alb, 0.0, dP_portal])

    def get_outputs(self, state):
        return self._current_outputs.copy()