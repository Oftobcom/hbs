# lungs.py
import numpy as np
from organ_base import OrganModel


class Lungs2Chamber(OrganModel):
    """
    Модель лёгких с тремя механизмами изменения сопротивления:

    1. Быстрая активная вазоконстрикция от потока (shear stress).
       При Q_pulm > Q_norm — рост R (линейный по flow_sensitivity).

    2. Пассивный recruitment + distension (нелинейный, по давлению).
       При росте P_pa раскрываются закрытые капилляры и расширяются
       уже открытые — эффективное R падает по сигмоиде до f_recruit_min.
       Это ключевой механизм, ограничивающий рост P_pa при нагрузке.

    3. Хроническое структурное ремоделирование от давления.
       Медленный рост R_remodel при P_pa > P_pa_threshold.

    Состояние: [P_prox, P_dist, R_remodel]
        P_prox     — давление в проксимальном сегменте (P_pa), мм рт. ст.
        P_dist     — давление в дистальном сегменте, мм рт. ст.
        R_remodel  — безразмерный множитель структурного ремоделирования,
                     стартует с 1.0, растёт до R_remodel_max.
    """

    # --- Санити-пороги для валидации конфигурации ---
    _K_FLOW_MIN = 1.0
    _K_FLOW_MAX = 100.0
    _R_MIN = 1e-3
    _R_MAX = 10.0
    _C_MIN = 0.1
    _C_MAX = 100.0
    _Q_NORM_MIN = 1.0
    _Q_NORM_MAX = 500.0
    _P50_MIN = 1.0
    _P50_MAX = 100.0
    _TAU_REMODEL_MIN = 1.0
    _TAU_REMODEL_MAX = 1e5
    _R_REMODEL_MAX_MIN = 1.0
    _R_REMODEL_MAX_LIMIT = 20.0

    def __init__(self,
                 R1=0.06, R2=0.04,
                 C1=3.0, C2=5.0,
                 # --- Быстрая вазоконстрикция от потока ---
                 flow_dependent_resistance=False,
                 flow_sensitivity=0.15,
                 Q_norm: float = 80.0,
                 k_flow: float = 15.0,
                 # --- Passive recruitment / distension ---
                 recruitment_enabled=True,
                 P_recruit_50=20.0,        # мм рт. ст., давление полу-рекруитмента
                 n_recruit=3.0,            # крутизна сигмоиды
                 f_recruit_min=0.55,       # R при полном рекруитменте (55% от базового)
                 # --- Хроническое структурное ремоделирование от давления ---
                 pressure_remodel=False,
                 P_pa_threshold=25.0,      # мм рт. ст., порог запуска
                 pressure_sensitivity=0.04,# прирост R_remodel на 1 мм рт. ст. превышения
                 R_remodel_max=5.0,
                 k_rarefaction: float = 0.5, 
                 tau_remodel=200.0):       # с, время выхода на R_target

        # =================================================================
        # Валидация конфигурации — fail-fast при инициализации.
        # Эти параметры приходят из YAML и не меняются во время симуляции;
        # ошибки в них должны ловиться один раз, а не в горячем пути RHS.
        # =================================================================

        def _check_range(name, v, lo, hi, typical=""):
            v = float(v)
            if not np.isfinite(v) or not (lo <= v <= hi):
                raise ValueError(
                    f"Lungs2Chamber: {name}={v} вне [{lo}, {hi}]. {typical}"
                )
            return v

        # --- k_flow (крутизна виртуального клапана) ---
        # Нижняя граница: δ=1/k_flow ≤ 1 мм рт.ст. — сглаживание не шире шкалы.
        # Верхняя граница: при k>100 профиль становится почти разрывным.
        self.k_flow = _check_range(
            "k_flow", k_flow, self._K_FLOW_MIN, self._K_FLOW_MAX,
            "типично 9–15 (ширина виртуального клапана ~1/k_flow мм рт.ст.). "
            "k_flow → 0 даёт нефизичную утечку v(0)=δ/(2R); "
            "k_flow > 100 делает клапан численно жёстким."
        )

        # --- Базовые сопротивления ---
        self.R1_base = _check_range(
            "R1", R1, self._R_MIN, self._R_MAX,
            "типично 0.02–0.10 мм рт.ст.·с/мл; проверьте единицы."
        )
        self.R2_base = _check_range(
            "R2", R2, self._R_MIN, self._R_MAX,
            "типично 0.02–0.10 мм рт.ст.·с/мл; проверьте единицы."
        )

        # --- Комплаенсы ---
        self.C1 = _check_range(
            "C1", C1, self._C_MIN, self._C_MAX,
            "типично 2–20 мл/мм рт.ст.; проверьте единицы."
        )
        self.C2 = _check_range(
            "C2", C2, self._C_MIN, self._C_MAX,
            "типично 2–20 мл/мм рт.ст.; проверьте единицы."
        )

        # --- Flow-зависимая вазоконстрикция ---
        self.flow_dependent_resistance = bool(flow_dependent_resistance)
        self.flow_sensitivity = float(flow_sensitivity)
        if self.flow_sensitivity < 0.0:
            raise ValueError(
                f"Lungs2Chamber: flow_sensitivity={flow_sensitivity} должно быть ≥ 0."
            )

        self.Q_norm = _check_range(
            "Q_norm", Q_norm, self._Q_NORM_MIN, self._Q_NORM_MAX,
            "типично 80, согласовано с target_CO в systemic."
        )

        # --- Passive recruitment ---
        self.recruitment_enabled = bool(recruitment_enabled)

        self.P_recruit_50 = _check_range(
            "P_recruit_50", P_recruit_50, self._P50_MIN, self._P50_MAX,
            "типично 20 мм рт.ст. — давление полу-рекруитмента."
        )

        n_rec = float(n_recruit)
        if not np.isfinite(n_rec) or n_rec <= 0.0:
            raise ValueError(
                f"Lungs2Chamber: n_recruit={n_recruit} должно быть > 0 (типично 3)."
            )
        self.n_recruit = n_rec

        fmin = float(f_recruit_min)
        if not np.isfinite(fmin) or not (0.0 < fmin < 1.0):
            raise ValueError(
                f"Lungs2Chamber: f_recruit_min={f_recruit_min} вне (0, 1). "
                f"Типично 0.55 (максимальный рекруитмент — 55% от базового R)."
            )
        self.f_recruit_min = fmin

        # --- Структурное ремоделирование ---
        self.pressure_remodel = bool(pressure_remodel)

        pthr = float(P_pa_threshold)
        if not np.isfinite(pthr):
            raise ValueError(
                f"Lungs2Chamber: P_pa_threshold={P_pa_threshold} не конечно."
            )
        self.P_pa_threshold = pthr

        ps = float(pressure_sensitivity)
        if not np.isfinite(ps) or ps < 0.0:
            raise ValueError(
                f"Lungs2Chamber: pressure_sensitivity={pressure_sensitivity} "
                f"должно быть ≥ 0."
            )
        self.pressure_sensitivity = ps

        self.R_remodel_max = _check_range(
            "R_remodel_max", R_remodel_max,
            self._R_REMODEL_MAX_MIN, self._R_REMODEL_MAX_LIMIT,
            "типично 3–10 (кратное превышение нормы PVR)."
        )

        self.k_rarefaction = _check_range(
            "k_rarefaction", k_rarefaction, 0.0, 2.0,
            "безразмерный, типично 0.5 — дополнительный рост PVR от запустевания капилляров."
        )

        self.tau_remodel = _check_range(
            "tau_remodel", tau_remodel,
            self._TAU_REMODEL_MIN, self._TAU_REMODEL_MAX,
            "типично 150–300 с."
        )

        self._current_outputs = {}

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
    def get_state_size(self):
        return 3   # [P_prox, P_dist, R_remodel]

    def get_initial_state(self):
        return np.array([16.0, 11.2, 1.0])

    # ------------------------------------------------------------------
    # 1. Быстрый активный отклик на поток
    # ------------------------------------------------------------------
    def _flow_factor(self, Q_pulm: float) -> float:
        """
        Активная вазоконстрикция в ответ на увеличенный поток.

        В норме при Q ≤ Q_norm — не активна (f_flow = 1).
        При Q > Q_norm — линейный рост: f_flow = 1 + s · (Q − Q_norm)/Q_norm.

        Верхний клип f_flow ≤ 3.0 — защита от нефизиологических потоков
        в переходных процессах.
        """
        if not self.flow_dependent_resistance:
            return 1.0

        Q = max(float(Q_pulm), 0.0)
        if Q <= self.Q_norm:
            return 1.0

        excess = (Q - self.Q_norm) / self.Q_norm
        f = 1.0 + self.flow_sensitivity * excess
        return float(np.clip(f, 1.0, 3.0))

    # ------------------------------------------------------------------
    # 1b. Гладкий односторонний поток (виртуальный клапан)
    # ------------------------------------------------------------------
    def _valve_flow(self, dP: float, R: float, P_operating: float = 15.0) -> float:
        """
        Гладкий односторонний клапан.

        v(dP) = max(0, (dP + hypot(dP, δ) − δ) / (2R))
        δ = (1/k_flow) · (1 + 0.04·max(P_operating − 15, 0))  — ширина
            сглаживания растёт с рабочим давлением, чтобы при высоких
            P_pa (ремоделирование, Эйзенменгер) клапан не «дребезжал»
            вблизи dP = 0.

        v(0) = 0  (δ вычитается, утечки через закрытый клапан нет)
        """
        # Мягкий клип R в runtime (без exception)
        if not np.isfinite(R):
            R_safe = 1e-6
        else:
            R_safe = max(float(R), 1e-6)

        delta = (1.0 / self.k_flow) * (1.0 + 0.04 * max(P_operating - 15.0, 0.0))
        q_raw = (float(dP) + np.hypot(float(dP), delta) - delta) / (2.0 * R_safe)
        return float(q_raw) if q_raw > 0.0 else 0.0

    # ------------------------------------------------------------------
    # 2. Пассивный recruitment + distension
    # ------------------------------------------------------------------
    def _recruit_factor(self, P_pa: float, R_remodel: float) -> float:
        """
        Множитель сопротивления от рекруитмента и рарефакции.

        R_remodel = 1     → f_healthy(P_pa) ∈ [f_recruit_min, 1]
        R_remodel = R_max → f_rarefaction = 1 + k_rarefaction (PVR выше структурного)
        Между             → линейная интерполяция по remodel_frac.
        """        
        if not self.recruitment_enabled:
            return 1.0
        P = float(np.clip(P_pa, 1.0, 200.0))
        ratio = (P / self.P_recruit_50) ** self.n_recruit
        f_sigmoid = 1.0 / (1.0 + ratio)
        f_healthy = self.f_recruit_min + (1.0 - self.f_recruit_min) * f_sigmoid

        remodel_frac = self._remodel_frac(R_remodel)

        # При полном фиброзе капилляры не только теряют способность
        # к дилатации, но и частично запустевают (rarefaction).
        # f_target > 1 → PVR выше «структурного» предсказания R_base · R_remodel.
        f_rarefaction = self._f_rarefaction(remodel_frac)

        return float(f_healthy * (1.0 - remodel_frac) + f_rarefaction * remodel_frac)

    # ------------------------------------------------------------------
    # 3. Медленное структурное ремоделирование
    # ------------------------------------------------------------------
    def _R_remodel_target(self, P_pa: float) -> float:
        if not self.pressure_remodel:
            return 1.0
        excess = max(float(P_pa) - self.P_pa_threshold, 0.0)
        if excess <= 0.0:
            return 1.0
        stimulus = self.pressure_sensitivity * excess * (1.0 + 0.03 * excess)
        R_target = 1.0 + (self.R_remodel_max - 1.0) * (1.0 - np.exp(-stimulus))
        return float(np.clip(R_target, 1.0, self.R_remodel_max))

    def _mode_from_R_remodel(self, R_remodel: float) -> str:
        if R_remodel < 1.5:
            return 'healthy'
        if R_remodel < 4.0:
            return 'compensated'
        return 'decompensated'

    def _remodel_frac(self, R_remodel: float) -> float:
        return float(np.clip(
            (R_remodel - 1.0) / max(self.R_remodel_max - 1.0, 1e-6),
            0.0, 1.0,
        ))

    def _f_rarefaction(self, remodel_frac: float) -> float:
        return 1.0 + self.k_rarefaction * remodel_frac
    
    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------
    def get_derivatives(self, t, state, inputs):
        P_prox, P_dist, R_remodel = state

        # Мягкие клипы (защита от solver retries, не от ошибок конфига)
        P_prox = max(float(P_prox), 0.0)
        P_dist = max(float(P_dist), 0.0)
        R_remodel = float(np.clip(R_remodel, 1.0, self.R_remodel_max))

        Q_pulm = inputs.get('Q_pulmonary', 0.0)
        P_pv = max(float(inputs.get('P_pv', 5.0)), 0.0)

        f_recruit = self._recruit_factor(P_prox, R_remodel)
        # Структурная компонента — ремоделирование
        R1_struct = self.R1_base * R_remodel
        R2_struct = self.R2_base * R_remodel
        f_flow = self._flow_factor(Q_pulm)
        remodel_frac = self._remodel_frac(R_remodel)
        f_flow_eff = 1.0 + (f_flow - 1.0) * (1.0 - remodel_frac) ** 2
        R1_eff = R1_struct * f_recruit * f_flow_eff
        R2_eff = R2_struct * f_recruit * f_flow_eff

        # Мягкий клип R в runtime без exception.
        # R1_eff/R2_eff в норме лежат в [0.002, 0.32] — клип не срабатывает.
        if not np.isfinite(R1_eff):
            R1_safe = 1e-6
        else:
            R1_safe = max(float(R1_eff), 1e-6)
        if not np.isfinite(R2_eff):
            R2_safe = 1e-6
        else:
            R2_safe = max(float(R2_eff), 1e-6)

        Q_int = (P_prox - P_dist) / R1_safe
        Q_out = self._valve_flow(P_dist - P_pv, R2_safe, P_operating=P_prox)

        dP_prox = (Q_pulm - Q_int) / self.C1
        dP_dist = (Q_int - Q_out) / self.C2

        R_target = self._R_remodel_target(P_prox)
        dR_remodel = max(R_target - R_remodel, 0.0) / self.tau_remodel

        # Диагностика
        self._current_outputs = {
            'P_pa': float(P_prox),
            'P_pa_dist': float(P_dist),
            'R1_eff': float(R1_eff),
            'R2_eff': float(R2_eff),
            'flow_factor': float(f_flow),
            'recruit_factor': float(f_recruit),
            'Q_int': float(Q_int),
            'Q_out': float(Q_out),
            'R_target': float(R_target),
            'dR_remodel': float(dR_remodel),
            'mode': self._mode_from_R_remodel(R_remodel),
            'R_pulm_total': float(R1_eff + R2_eff),
            'f_rarefaction': self._f_rarefaction(remodel_frac),
            'R_remodel': float(R_remodel),
        }
        return np.array([dP_prox, dP_dist, dR_remodel])

    def get_outputs(self, state):
        if self._current_outputs:
            return self._current_outputs.copy()
        # fallback: вычисляем из state без побочных эффектов на кэш
        P_prox = max(float(state[0]), 0.0)
        P_dist = max(float(state[1]), 0.0)
        R_rem = float(np.clip(state[2], 1.0, self.R_remodel_max))
        f_rec = self._recruit_factor(P_prox, R_rem)

        R_target = self._R_remodel_target(P_prox)
        dR = max(R_target - R_rem, 0.0) / self.tau_remodel
        remodel_frac = self._remodel_frac(R_rem)
        return {
            'P_pa': P_prox,
            'P_pa_dist': P_dist,
            'R_remodel': R_rem,
            'R1_eff': self.R1_base * R_rem * f_rec,
            'R2_eff': self.R2_base * R_rem * f_rec,
            'recruit_factor': f_rec,
            'flow_factor': 1.0,
            'R_pulm_total': (self.R1_base + self.R2_base) * R_rem * f_rec,
            'R_target': float(R_target),
            'dR_remodel': float(dR),
            'mode': self._mode_from_R_remodel(R_rem),
            'Q_int': 0.0, 'Q_out': 0.0,
            'f_rarefaction': self._f_rarefaction(remodel_frac),
        }