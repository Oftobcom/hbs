# blood.py
"""
Модель крови как единого резервуара.

Состояние — только концентрации, V_blood приходит извне для разбавления
"""

import numpy as np
from organ_base import OrganModel
from typing import List, Dict, Any


class BloodPool(OrganModel):
    """
    Модель крови как единого резервуара.
    """

    # --- Санити-пороги для валидации конфигурации ---
    _V0_MIN, _V0_MAX = 1000.0, 15000.0
    _CONC_MIN, _CONC_MAX = 0.0, 1e4

    def __init__(self,
                 substance_names: List[str],
                 V0: float = 5000.0,
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

    # ------------------------------------------------------------------
    # Обязательный интерфейс OrganModel
    # ------------------------------------------------------------------
    def get_state_size(self) -> int:
        return self.num_substances

    def get_initial_state(self) -> np.ndarray:
        return self.C0.copy()

    # ------------------------------------------------------------------
    # Основной метод — производные
    # ------------------------------------------------------------------
    def get_derivatives(self, t: float, state_slice: np.ndarray,
                        inputs: Dict[str, Any]) -> np.ndarray:
        """
        Параметры
        ---------
        state_slice : [C_0, ..., C_n] — только концентрации.
                      V_blood больше НЕ в состоянии (см. docstring модуля).
        inputs      : dict с ключами
            V_blood   — текущий полный объём крови (мл), derived в whole_body
                        как сумма всех физических V (heart, lungs, sys_art,
                        sys_ven, pul_ven, jugular, liver, gitract, brain).
            dV_blood  — суммарная скорость изменения объёма (мл/с) =
                        absorption + intake − urine − insensible.
            dC        — вектор скоростей изменения концентраций (масса/мл/с),
                        без учёта разбавления (contributions от органов).

        Возвращает
        ----------
        np.ndarray длины n: dC_with_dilution
        """
        # --- Разбор состояния с мягким клипом (защита от LSODA retries) ---
        C_raw = np.asarray(state_slice, dtype=float)
        C = np.maximum(C_raw, 0.0)

        # --- Входы ---
        V = max(float(inputs.get('V_blood', self.V0)), 1e-6)
        dV = float(inputs.get('dV_blood', 0.0))
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
                raise ValueError("BloodPool: dC_input содержит NaN/Inf.")
            # dM/dt = V·dC + C·dV, но храним C:
            #   dC_observed = dC_input − C·(dV/V)
            dC = dC_input - C * dV / V   # V ≥ 1e-6 → деление безопасно

        return dC

    # ------------------------------------------------------------------
    # Выходы
    # ------------------------------------------------------------------
    def get_outputs(self, state_slice):
        concentrations = np.asarray(state_slice, dtype=float)
        outputs = {}
        for name, value in zip(self.substance_names, concentrations):
            outputs[f'C_{name}'] = float(value)
        return outputs