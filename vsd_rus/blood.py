# blood.py
"""
Модель крови как единого резервуара.

Состояние: [V_blood, C_0, C_1, ..., C_n]
    V_blood — общий объём циркулирующей крови (мл)
    C_i     — концентрация i-го вещества (масса/мл)

Ключевые механизмы:
    • dV = absorption + intake − urine − insensible  (собирается в whole_body)
    • Разбавление: при ↑V концентрации падают, даже если dC_contrib=0
      dC = dC_contrib − C · dV / V
    • Мягкий пол: при V < V_min и dV < 0 отток гасится до нуля

Единицы:
    V — мл
    C — масса/мл (интерпретация зависит от вещества)
"""

import numpy as np
from organ_base import OrganModel
from typing import List, Dict, Any


class BloodPool(OrganModel):
    """
    Модель крови как единого резервуара.

    Состояние: [V_blood, C_0, ..., C_n]
    """

    # --- Санити-пороги для валидации конфигурации ---
    _V0_MIN, _V0_MAX = 1000.0, 15000.0
    _V_MIN_MIN, _V_MIN_MAX = 100.0, 10000.0
    _CONC_MIN, _CONC_MAX = 0.0, 1e4

    def __init__(self,
                 substance_names: List[str],
                 V0: float = 5000.0,
                 V_min: float = 2000.0,
                 initial_concentrations: Dict[str, float] = None):

        # =================================================================
        # Валидация конфигурации — fail-fast при инициализации.
        # =================================================================

        # --- substance_names: list/tuple непустых уникальных строк ---
        if not isinstance(substance_names, (list, tuple)):
            raise ValueError(
                f"BloodPool: substance_names должен быть list/tuple, "
                f"получено {type(substance_names).__name__}."
            )
        if len(substance_names) == 0:
            raise ValueError("BloodPool: substance_names пуст.")
        if not all(isinstance(s, str) and s for s in substance_names):
            raise ValueError(
                "BloodPool: substance_names должен содержать непустые строки."
            )
        if len(set(substance_names)) != len(substance_names):
            raise ValueError(
                f"BloodPool: substance_names содержит дубликаты: "
                f"{substance_names}."
            )

        self.substance_names = list(substance_names)
        self.num_substances = len(self.substance_names)

        # --- V0: объём крови ---
        def _check_range(name, v, lo, hi, typical=""):
            v = float(v)
            if not np.isfinite(v) or not (lo <= v <= hi):
                raise ValueError(
                    f"BloodPool: {name}={v} вне [{lo}, {hi}]. {typical}"
                )
            return v

        self.V0 = _check_range(
            "V0", V0, self._V0_MIN, self._V0_MAX,
            "мл, типично 5000–6500 (нормоволемия у взрослого)."
        )

        # --- V_min: нижний мягкий пол ---
        self.V_min = _check_range(
            "V_min", V_min, self._V_MIN_MIN, self._V_MIN_MAX,
            "мл, типично 2000 — порог, ниже которого гасится отток."
        )
        if self.V_min >= self.V0:
            raise ValueError(
                f"BloodPool: V_min={self.V_min} должно быть < V0={self.V0} "
                f"(иначе мягкий пол срабатывает уже в начальном состоянии)."
            )

        # --- initial_concentrations ---
        if initial_concentrations is None:
            self.C0 = np.zeros(self.num_substances)
        else:
            if not isinstance(initial_concentrations, dict):
                raise ValueError(
                    f"BloodPool: initial_concentrations должен быть dict, "
                    f"получено {type(initial_concentrations).__name__}."
                )
            C0_list = []
            for name in self.substance_names:
                v = float(initial_concentrations.get(name, 0.0))
                if not np.isfinite(v) or not (self._CONC_MIN <= v <= self._CONC_MAX):
                    raise ValueError(
                        f"BloodPool: initial_concentrations[{name!r}]={v} "
                        f"вне [{self._CONC_MIN}, {self._CONC_MAX}]."
                    )
                C0_list.append(v)
            self.C0 = np.array(C0_list)

        # Кэш не используется, но оставлен для совместимости
        self._current_state = None

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
    def get_state_size(self) -> int:
        return 1 + self.num_substances

    def get_initial_state(self) -> np.ndarray:
        return np.concatenate(([self.V0], self.C0))

    # ------------------------------------------------------------------
    # Основной метод — производные
    # ------------------------------------------------------------------
    def get_derivatives(self, t: float, state_slice: np.ndarray,
                        inputs: Dict[str, Any]) -> np.ndarray:
        """
        Параметры
        ---------
        state_slice : [V, C_0, ..., C_n]
        inputs      : dict с ключами
            dV — суммарная скорость изменения объёма (мл/с), собирается
                 в whole_body как absorption + intake − urine − insensible
            dC — вектор скоростей изменения концентраций (масса/мл/с),
                 без учёта разбавления (contributions от органов)

        Возвращает
        ----------
        np.ndarray длины 1+n: [dV, dC_with_dilution]
        """
        # --- Разбор состояния с мягкими клипами (защита от LSODA retries) ---
        V_raw = float(state_slice[0])
        V = max(V_raw, 0.0)

        C_raw = np.asarray(state_slice[1:], dtype=float)
        C = np.maximum(C_raw, 0.0)

        # --- Входы ---
        dV = float(inputs.get('dV', 0.0))
        dC_input = inputs.get('dC', None)

        # --- Разбавление концентраций ---
        if dC_input is None:
            dC = np.zeros(self.num_substances)
        else:
            dC_input = np.asarray(dC_input, dtype=float)
            if dC_input.shape[0] != self.num_substances:
                raise ValueError(
                    f"BloodPool: dC_input имеет длину {dC_input.shape[0]}, "
                    f"ожидается {self.num_substances}."
                )
            if not np.all(np.isfinite(dC_input)):
                raise ValueError(
                    "BloodPool: dC_input содержит NaN/Inf."
                )
            # Правильный баланс с разбавлением:
            #   dM/dt = V·dC + C·dV, но мы храним C, а не M.
            #   dC_observed = dC_input − C·(dV/V)
            # Второй член — эффект разбавления/концентрации при изменении V.
            if V > 1e-6:
                dC = dC_input - C * dV / V
            else:
                dC = dC_input

        # --- Мягкий пол на объём ---
        # При V < V_min и dV < 0 гасим отток, чтобы V не ушёл в глубокий минус
        # за счёт численного сбоя LSODA.
        if V < self.V_min and dV < 0:
            dV = 0.0

        return np.concatenate(([dV], dC))

    # ------------------------------------------------------------------
    # Выходы
    # ------------------------------------------------------------------
    def get_outputs(self, state_slice: np.ndarray) -> Dict[str, float]:
        V = float(state_slice[0])
        concentrations = np.asarray(state_slice[1:], dtype=float)
        outputs = {'V_blood': V}
        for name, value in zip(self.substance_names, concentrations):
            outputs[f'C_{name}'] = float(value)
        return outputs