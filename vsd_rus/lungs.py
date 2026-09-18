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

    def __init__(self,
                 R1=0.06, R2=0.04,
                 C1=4.0, C2=8.0,
                 # --- Быстрая вазоконстрикция от потока ---
                 flow_dependent_resistance=False,
                 flow_sensitivity=0.15,
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

        self.R1_base = R1
        self.R2_base = R2
        self.C1 = C1
        self.C2 = C2

        self.flow_dependent_resistance = flow_dependent_resistance
        self.flow_sensitivity = flow_sensitivity

        self.recruitment_enabled = bool(recruitment_enabled)
        self.P_recruit_50 = float(P_recruit_50)
        self.n_recruit = float(n_recruit)
        self.f_recruit_min = float(np.clip(f_recruit_min, 0.1, 1.0))

        self.pressure_remodel = pressure_remodel
        self.P_pa_threshold = P_pa_threshold
        self.pressure_sensitivity = pressure_sensitivity
        self.R_remodel_max = R_remodel_max
        self.tau_remodel = tau_remodel

        self._current_outputs = {}

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

        Q_norm = 80.0
        Q = max(float(Q_pulm), 0.0)
        if Q <= Q_norm:
            return 1.0

        excess = (Q - Q_norm) / Q_norm
        f = 1.0 + self.flow_sensitivity * excess
        return float(np.clip(f, 1.0, 3.0))

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
        Q_pulm = inputs.get('Q_pulmonary', 0.0)
        P_pv = inputs.get('P_pv', 5.0)

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

        # Защита от деления на почти-ноль (не должно случаться, но на всякий случай)
        R1_safe = max(R1_eff, 1e-6)
        R2_safe = max(R2_eff, 1e-6)

        dP_prox = (Q_pulm - (P_prox - P_dist) / R1_safe) / self.C1
        dP_dist = ((P_prox - P_dist) / R1_safe - (P_dist - P_pv) / R2_safe) / self.C2

        # Диагностика
        self._current_outputs = {
            'P_pa': float(P_prox),
            'P_pa_dist': float(P_dist),
            'R1_eff': float(R1_eff),
            'R2_eff': float(R2_eff),
            'R_remodel': float(R_remodel),
            'flow_factor': float(f_flow),
            'recruit_factor': float(f_recruit),   # <-- новый диагностический выход
        }
        return np.array([dP_prox, dP_dist, dR_remodel])

    def get_outputs(self, state):
        return self._current_outputs.copy()