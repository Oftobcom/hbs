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

    Эффективное сопротивление:
        R_eff = R_base × f_recruit(P_pa) × f_flow(Q_pulm) × R_remodel

    В покое (P_pa ≈ 15, Q ≈ 83):      f_recruit ≈ 0.87, f_flow = 1.0
    При нагрузке (P_pa ≈ 30, Q ≈ 350): f_recruit ≈ 0.65, f_flow ≈ 1.5
    Произведение почти не меняется — лёгкие стабилизируют PVR.
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
                 C1=4.0, C2=8.0,
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
        # Верхняя граница: при k>100 профиль становится почти разрывным, LSODA
        # начинает тратить шаги на «дребезг» около dP=0.
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
    def _valve_flow(self, dP: float, R: float) -> float:
        """
        Гладкий односторонний клапан (сдвинутый smooth ReLU).

            v(dP) = max( 0, (dP + sqrt(dP² + δ²) − δ) / (2R) )
            где δ = 1/k_flow — ширина сглаживания.

        Свойства:
            dP = 0     →  v = 0              (клапан полностью закрыт, утечки нет)
            dP >> δ    →  v ≈ dP/R − δ/(2R)  (ламинарный поток)
            dP << −δ   →  v = 0              (обратного тока нет)

        Гарантированно ≥ 0 при любом dP. Гладкая C¹ (излом производной
        только в dP = 0). Монотонно не убывает.

        Параметр δ вычитается, чтобы v(0) = 0. Без него v(0) = δ/(2R) > 0 —
        постоянная «утечка» из P_dist в P_pv, которая при Q=0 и P_pv=0
        уводит P_dist в глубокий минус. Именно этот дефект был в старой
        версии и ловился в debug_lungs (TEST 9, TEST 11).

        Численная защита:
          • R клипуется к 1e-6 — от solver retries, не от ошибок конфига
            (конфиг валидируется в __init__).
          • R=NaN/Inf тоже отлавливается: max(NaN, 1e-6) в Python даёт NaN,
            поэтому сначала проверяем np.isfinite.
          • np.hypot(dP, δ) вместо sqrt(dP²+δ²) — не переполняется при
            больших |dP|.
        """
        # Мягкий клип R в runtime (без exception)
        if not np.isfinite(R):
            R_safe = 1e-6
        else:
            R_safe = max(float(R), 1e-6)

        delta = 1.0 / self.k_flow          # k_flow ≥ 1 (валидировано в __init__)
        q_raw = (float(dP) + np.hypot(float(dP), delta) - delta) / (2.0 * R_safe)
        return float(q_raw) if q_raw > 0.0 else 0.0

    # ------------------------------------------------------------------
    # 2. Пассивный recruitment + distension
    # ------------------------------------------------------------------
    def _recruit_factor(self, P_pa: float) -> float:
        """
        Passive recruitment + distension лёгочных сосудов.

        При росте P_pa раскрываются закрытые капилляры и расширяются
        уже открытые — эффективное сопротивление падает.

        Сигмоида:
            f(P) = f_min + (1 − f_min) / (1 + (P/P50)^n)

        Пределы:
            P << P50  →  f → 1.0          (только базовые капилляры)
            P >> P50  →  f → f_min         (полный рекруитмент)

        Типичные значения для здорового взрослого:
            P_pa = 10 → 0.95
            P_pa = 15 → 0.87
            P_pa = 20 → 0.78
            P_pa = 30 → 0.65
            P_pa = 40 → 0.60
        """
        if not self.recruitment_enabled:
            return 1.0

        # Клип давления в разумных пределах, чтобы exp/степень не взорвались
        P = float(np.clip(P_pa, 1.0, 200.0))

        ratio = (P / self.P_recruit_50) ** self.n_recruit
        f_sigmoid = 1.0 / (1.0 + ratio)

        return self.f_recruit_min + (1.0 - self.f_recruit_min) * f_sigmoid

    # ------------------------------------------------------------------
    # 3. Медленное структурное ремоделирование
    # ------------------------------------------------------------------
    def _R_remodel_target(self, P_pa: float) -> float:
        """
        Целевой множитель структурного ремоделирования.

        При pressure_remodel=False — всегда 1.0.
        При True — растёт линейно при P_pa > P_pa_threshold,
        ограничен R_remodel_max.
        """
        if not self.pressure_remodel:
            return 1.0
        excess_p = max(float(P_pa) - self.P_pa_threshold, 0.0)
        R_target = 1.0 + self.pressure_sensitivity * excess_p
        return float(min(R_target, self.R_remodel_max))

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------
    def get_derivatives(self, t, state, inputs):
        P_prox, P_dist, R_remodel = state

        # Мягкие клипы (защита от solver retries, не от ошибок конфига)
        P_prox = max(float(P_prox), 0.0)
        P_dist = max(float(P_dist), 0.0)
        R_remodel = float(np.clip(R_remodel, 0.1, self.R_remodel_max))

        Q_pulm = inputs.get('Q_pulmonary', 0.0)
        P_pv = max(float(inputs.get('P_pv', 5.0)), 0.0)

        # 1. Быстрый активный отклик на поток
        f_flow = self._flow_factor(Q_pulm)

        # 2. Пассивный recruitment / distension
        f_recruit = self._recruit_factor(P_prox)

        # 3. Медленный структурный отклик на давление
        R_target = self._R_remodel_target(P_prox)
        dR_remodel = (R_target - R_remodel) / self.tau_remodel

        # 4. Эффективное сопротивление:
        #    base × passive_recruit × active_flow × chronic_remodel
        R1_eff = self.R1_base * f_recruit * f_flow * R_remodel
        R2_eff = self.R2_base * f_recruit * f_flow * R_remodel

        # Мягкий клип R в runtime без exception.
        # R1_eff/R2_eff в норме лежат в [0.002, 0.32] — клип не срабатывает.
        # NaN-проверка страхует от численных сбоев LSODA.
        if not np.isfinite(R1_eff):
            R1_safe = 1e-6
        else:
            R1_safe = max(float(R1_eff), 1e-6)
        if not np.isfinite(R2_eff):
            R2_safe = 1e-6
        else:
            R2_safe = max(float(R2_eff), 1e-6)

        Q_int = (P_prox - P_dist) / R1_safe                          # внутри лёгких — без клапана
        Q_out = self._valve_flow(P_dist - P_pv, R2_safe)             # односторонний к P_pv

        dP_prox = (Q_pulm - Q_int) / self.C1
        dP_dist = (Q_int - Q_out) / self.C2

        # Диагностика
        self._current_outputs = {
            'P_pa': float(P_prox),
            'P_pa_dist': float(P_dist),
            'R1_eff': float(R1_eff),
            'R2_eff': float(R2_eff),
            'R_remodel': float(R_remodel),
            'flow_factor': float(f_flow),
            'recruit_factor': float(f_recruit),
            'Q_int': float(Q_int),
            'Q_out': float(Q_out),
        }
        return np.array([dP_prox, dP_dist, dR_remodel])

    def get_outputs(self, state):
        return self._current_outputs.copy()