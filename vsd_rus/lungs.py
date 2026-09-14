# lungs.py
import numpy as np
from organ_base import OrganModel


class Lungs2Chamber(OrganModel):
    """
    Модель лёгких с двумя механизмами роста сопротивления:

    1. Быстрая вазоконстрикция от потока (flow-dependent).
    2. Хроническое структурное ремоделирование от давления (pressure_remodel).

    Состояние: [P_prox, P_dist, R_remodel]
        P_prox     — давление в проксимальном сегменте (P_pa), мм рт. ст.
        P_dist     — давление в дистальном сегменте, мм рт. ст.
        R_remodel  — безразмерный множитель структурного ремоделирования,
                     стартует с 1.0, растёт до R_remodel_max.
    """

    def __init__(self,
                 R1=0.06, R2=0.04,
                 C1=4.0, C2=8.0,
                 # --- Быстрая вазоконстрикция от потока ---
                 flow_dependent_resistance=False,
                 flow_sensitivity=0.15,
                 # --- Хроническое ремоделирование от давления ---
                 pressure_remodel=False,
                 P_pa_threshold=25.0,        # мм рт. ст., порог запуска
                 pressure_sensitivity=0.04,  # прирост R_remodel на 1 мм Hg превышения
                 R_remodel_max=5.0,
                 tau_remodel=200.0):         # с, время выхода на R_target
        self.R1_base = R1
        self.R2_base = R2
        self.C1 = C1
        self.C2 = C2
        self.flow_dependent_resistance = flow_dependent_resistance
        self.flow_sensitivity = flow_sensitivity
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
    # Быстрый (алгебраический) множитель от потока
    # ------------------------------------------------------------------
    def _flow_factor(self, Q_pulm):
        if not self.flow_dependent_resistance:
            return 1.0
        Q_norm = 80.0
        if Q_pulm <= Q_norm:
            return 1.0
        excess = (Q_pulm - Q_norm) / Q_norm
        # Без потолка: линейный рост, но с малым коэффициентом
        return 1.0 + self.flow_sensitivity * excess

    # ------------------------------------------------------------------
    # Медленный (дифференциальный) множитель структурного ремоделирования
    # ------------------------------------------------------------------
    def _R_remodel_target(self, P_pa):
        if not self.pressure_remodel:
            return 1.0
        excess_p = max(P_pa - self.P_pa_threshold, 0.0)
        R_target = 1.0 + self.pressure_sensitivity * excess_p
        return min(R_target, self.R_remodel_max)

    # ------------------------------------------------------------------
    def get_derivatives(self, t, state, inputs):
        P_prox, P_dist, R_remodel = state
        Q_pulm = inputs.get('Q_pulmonary', 0.0)
        P_pv = inputs.get('P_pv', 5.0)

        # 1. Быстрый отклик на поток
        f_flow = self._flow_factor(Q_pulm)

        # 2. Медленный структурный отклик на давление
        R_target = self._R_remodel_target(P_prox)
        dR_remodel = (R_target - R_remodel) / self.tau_remodel

        # 3. Эффективные сопротивления: base × fast × slow
        R1_eff = self.R1_base * f_flow * R_remodel
        R2_eff = self.R2_base * f_flow * R_remodel

        dP_prox = (Q_pulm - (P_prox - P_dist) / R1_eff) / self.C1
        dP_dist = ((P_prox - P_dist) / R1_eff - (P_dist - P_pv) / R2_eff) / self.C2

        self._current_outputs = {
            'P_pa': P_prox,
            'P_pa_dist': P_dist,
            'R1_eff': R1_eff,
            'R2_eff': R2_eff,
            'R_remodel': R_remodel,
            'flow_factor': f_flow,
        }
        return np.array([dP_prox, dP_dist, dR_remodel])

    def get_outputs(self, state):
        return self._current_outputs.copy()