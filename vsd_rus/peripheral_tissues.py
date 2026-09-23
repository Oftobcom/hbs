# peripheral_tissues.py
"""
Модель периферических тканей (скелетные мышцы, кожа, соединительная ткань).

Заменяет статический R_sys_peripheral в WholeBodyModel на динамический
компартмент с:
    1. Метаболической ауторегуляцией (гипоксия → вазодилатация).
    2. Миогенной ауторегуляцией (высокое P_sa → вазоконстрикция).
    3. Потреблением O2 (добавляется к dC_oxygen в BloodPool).
    4. Продукцией лактата при гипоксии (новый маркер в BloodPool).

Состояние: [C_O2_local, C_lactate_local, R_eff]
    C_O2_local      — локальная концентрация O2 (мл O2 / мл крови-экв.)
    C_lactate_local — локальная концентрация лактата (мг/мл)
    R_eff           — эффективное периферическое сопротивление
                      (мм рт.ст.·с/мл)

Единицы:
    P          — мм рт. ст.
    Q          — мл/с
    R          — мм рт.ст.·с/мл
    C_O2       — мл O2 / мл крови
    C_lactate  — мг/мл  (≈ 0.09 мг/мл = 1 мМ)

Распределение VO2 (после централизации в whole_body.py):
    Периферия только РАПОРТУЕТ потребление через ключ
    'O2_consumption_periph' — это flow-зависимая величина VO2_eff.
    В dC_O2 крови периферия НЕ пишет: централизованный баланс O2/CO2
    собирается в whole_body._compute_organ_flows, который суммирует
    VO2_brain + VO2_periph + VO2_rest. Поле '_diagnostic_dC_O2_blood'
    оставлено только для отладки и в баланс не подмешивается.
"""

import numpy as np
from organ_base import OrganModel


class PeripheralTissues(OrganModel):
    # ------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------
    def __init__(self,
        # --- Гемодинамика ---
        R_base: float = 3.8,            # базовое сопротивление
        C_tissue: float = 8.0,          # (не используется напрямую, оставлено для расширения)
        P_tissue0: float = 15.0,        # (диагностика)

        # --- Метаболическая ауторегуляция ---
        O2_norm: float = 0.15,          # нормальная локальная O2
        k_O2_autoreg: float = 0.5,      # чувствительность к гипоксии
        R_min_factor: float = 0.75,      # максимальная вазодилатация
        tau_autoreg: float = 3.0,       # с, постоянная времени R_eff

        # --- Миогенная ауторегуляция ---
        k_P_myogenic: float = 0.002,    # чувствительность к P_sa
        # k_P_myogenic: float = 0.005,    # чувствительность к P_sa
        P_sa_norm: float = 90.0,        # норм. P_sa
        R_max_factor: float = 2.5,      # максимальная вазоконстрикция
        P_myogenic_deadband: float = 10.0,

        # --- Потребление O2 ---
        VO2_base: float = 1.5,          # мл O2/с (мышцы+кожа, ≈90 мл/мин)
        V_tissue_eff: float = 400.0,    # мл, эффективный объём тканевого O2
        C_a_O2_norm: float = 0.20,      # артериальная O2 (для автоинициализации)

        # --- Лактат ---
        C_O2_lactate_threshold: float = 0.08,  # порог гипоксии
        k_lactate_prod: float = 0.05,          # мг/(мл·с) при гипоксии
        k_lactate_clear: float = 0.0,         # 1/с
        k_lactate_release: float = 0.05,       # 1/с
        C_lactate0: float = 0.10,              # мг/мл
        C_O2_local0: float = 0.10):            # мл O2/мл
        
        # Гемодинамика
        self.R_base = float(R_base)
        self.C_tissue = float(C_tissue)
        self.P_tissue0 = float(P_tissue0)

        # Ауторегуляция
        self.O2_norm = float(O2_norm)
        self.k_O2_autoreg = float(k_O2_autoreg)
        self.R_min_factor = float(R_min_factor)
        self.tau_autoreg = float(tau_autoreg)

        self.k_P_myogenic = float(k_P_myogenic)
        self.P_sa_norm = float(P_sa_norm)
        self.R_max_factor = float(R_max_factor)
        self.P_myogenic_deadband = float(P_myogenic_deadband)

        # Метаболизм
        self.VO2_base = float(VO2_base)
        self.V_tissue_eff = float(V_tissue_eff)
        self.C_a_O2_norm = float(C_a_O2_norm)

        # Лактат
        self.C_O2_lactate_threshold = float(C_O2_lactate_threshold)
        self.k_lactate_prod = float(k_lactate_prod)
        self.k_lactate_clear = float(k_lactate_clear)
        self.k_lactate_release = float(k_lactate_release)

        # Инициализация
        self.C_lactate0 = float(C_lactate0)
        self.C_O2_local0 = float(C_O2_local0)

        # Кэш выходов
        self._current_outputs = {}

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
    def get_state_size(self) -> int:
        return 3   # [C_O2_local, C_lactate_local, R_eff]

    def get_initial_state(self) -> np.ndarray:
        return np.array([self.C_O2_local0, self.C_lactate0, self.R_base])

    # ------------------------------------------------------------------
    # Ауторегуляция
    # ------------------------------------------------------------------
    def _autoregulation_target(self, P_sa: float, C_O2_local: float) -> float:
        """
        Целевое R_eff — комбинация метаболической и миогенной регуляции.

        Метаболическая: гипоксия -> вазодилатация (f_O2 < 1)
        Миогенная:      высокое P_sa -> вазоконстрикция (f_P > 1)
        """
        # Метаболический фактор
        hypoxia_excess = max(self.O2_norm - C_O2_local, 0.0)
        f_O2 = 1.0 - self.k_O2_autoreg * hypoxia_excess
        f_O2 = float(np.clip(f_O2, self.R_min_factor, 1.0))

        # Миогенный фактор (только вазоконстрикция при высоком P)
        pressure_excess = P_sa - self.P_sa_norm
        if abs(pressure_excess) <= self.P_myogenic_deadband:
            f_P = 1.0
        else:
            signed = pressure_excess - np.sign(pressure_excess) * self.P_myogenic_deadband
            f_P = 1.0 + self.k_P_myogenic * signed
        f_P = float(np.clip(f_P, self.R_min_factor, self.R_max_factor))

        return self.R_base * f_O2 * f_P

    # ------------------------------------------------------------------
    # Производные
    # ------------------------------------------------------------------
    def get_derivatives(self, t, state, inputs):
        C_O2_loc_raw, C_lac_loc, R_eff = state

        # --- Входы ---
        P_sa         = float(inputs.get('P_sa', 85.0))
        P_sv         = float(inputs.get('P_sv', 5.0))
        C_a_O2       = float(inputs.get('C_a_O2', self.C_a_O2_norm))
        C_v_lactate  = float(inputs.get('C_v_lactate', 0.10))
        V_blood      = float(inputs.get('V_blood', 5000.0))
        V_blood      = max(V_blood, 1e-6)

        C_O2_loc = float(np.clip(C_O2_loc_raw, 0.0, max(C_a_O2, 0.0)))

        # --- 1. Ауторегуляция: R_eff релаксирует к целевому ---
        R_target = self._autoregulation_target(P_sa, C_O2_loc)
        dR_eff = (R_target - R_eff) / self.tau_autoreg

        # --- 2. Кровоток через периферию ---
        Q_periph = (P_sa - P_sv) / max(R_eff, 1e-6)
        Q_periph = max(Q_periph, 0.0)   # нет обратного тока

        # --- 3. Баланс O2 в ткани ---
        Q_factor = float(np.clip(Q_periph / 20.0, 0.1, 1.5))
        VO2_eff = self.VO2_base * Q_factor
        O2_delivery_rate = Q_periph * (C_a_O2 - C_O2_loc)
        dC_O2_loc = (O2_delivery_rate - VO2_eff) / self.V_tissue_eff
        if C_O2_loc <= 0.0 and dC_O2_loc < 0.0:
            dC_O2_loc = 0.0

        # --- 4. Лактат ---
        hypoxia_severity = max(self.C_O2_lactate_threshold - C_O2_loc, 0.0)
        lac_production = self.k_lactate_prod * hypoxia_severity            # мг/(мл·с)
        lac_clearance  = self.k_lactate_clear * C_lac_loc                  # мг/(мл·с)
        lac_release    = self.k_lactate_release * max(C_lac_loc - C_v_lactate, 0.0)
        dC_lac_loc = lac_production - lac_clearance - lac_release

        # --- 5. Вклады в BloodPool ---
        # Локальная скорость потребления O2 (диагностика).
        # В системный баланс НЕ подмешивается — это делает whole_body
        # через централизованный блок dC_O2_blood (см. fix_deepseek.md §2.6).
        dC_O2_blood = -VO2_eff / V_blood
        # Выделение лактата в кровь: lac_release * V_tissue_eff / V_blood (мг/мл/с)
        dC_lactate_blood = (lac_release * self.V_tissue_eff) / V_blood

        # --- 6. Кэш выходов ---
        self._current_outputs = {
            # Гемодинамика
            'Q_peripheral': float(Q_periph),
            'R_eff':        float(R_eff),
            'R_target':     float(R_target),
            'P_tissue':     float(self.P_tissue0),   # диагностика (не state)

            # Ауторегуляция
            'f_O2_autoreg': float(np.clip(
                1.0 - self.k_O2_autoreg * max(self.O2_norm - C_O2_loc, 0.0),
                self.R_min_factor, 1.0)),
            'f_P_myogenic': float(np.clip(
                1.0 + self.k_P_myogenic * (P_sa - self.P_sa_norm),
                1.0, self.R_max_factor)),

            # Метаболизм
            'C_O2_local':      float(C_O2_loc),
            'C_lactate_local': float(C_lac_loc),
            'O2_consumption_periph': float(VO2_eff),
            'lactate_production':    float(lac_production),
            'lactate_release_to_blood': float(lac_release),

            # Производные для BloodPool
            '_diagnostic_dC_O2_blood': float(dC_O2_blood),
            'dC_lactate_blood':        float(dC_lactate_blood),
        }

        return np.array([dC_O2_loc, dC_lac_loc, dR_eff])

    def get_outputs(self, state):
        return self._current_outputs.copy()


# =====================================================================
# Быстрый тест (python peripheral_tissues.py)
# =====================================================================
if __name__ == "__main__":
    pt = PeripheralTissues()
    print(f"State size: {pt.get_state_size()}")
    print(f"Initial state: {pt.get_initial_state()}")

    from scipy.integrate import solve_ivp

    def rhs(t, y):
        return pt.get_derivatives(t, y, {
            'P_sa': 85.0, 'P_sv': 5.0, 'C_a_O2': 0.20,
            'C_v_lactate': 0.10, 'V_blood': 5000.0,
        })

    sol = solve_ivp(rhs, (0, 60), pt.get_initial_state(),
                    method='LSODA', rtol=1e-4, atol=1e-5, max_step=0.07)
    y_end = sol.y[:, -1]
    pt.get_derivatives(sol.t[-1], y_end, {
        'P_sa': 85.0, 'P_sv': 5.0, 'C_a_O2': 0.20,
        'C_v_lactate': 0.10, 'V_blood': 5000.0,
    })
    out = pt.get_outputs(y_end)
    print("\nSteady state (здоровый):")
    for k in ('Q_peripheral', 'R_eff', 'C_O2_local', 'C_lactate_local',
              'O2_consumption_periph', 'lactate_production',
              '_diagnostic_dC_O2_blood'):
        print(f"  {k:28s} = {out[k]:+.6g}")

    print("\nГипоксия (P_sa=60, C_a_O2=0.12):")
    pt.get_derivatives(0.0, pt.get_initial_state(), {
        'P_sa': 60.0, 'P_sv': 5.0, 'C_a_O2': 0.12,
        'C_v_lactate': 0.10, 'V_blood': 5000.0,
    })
    out = pt.get_outputs(pt.get_initial_state())
    for k in ('Q_peripheral', 'R_eff', 'f_O2_autoreg', 'f_P_myogenic',
              'lactate_production'):
        print(f"  {k:28s} = {out[k]:+.6g}")