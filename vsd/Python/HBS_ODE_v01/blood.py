# blood.py
import numpy as np
from organ_base import OrganModel
from typing import List, Dict, Any


class BloodPool(OrganModel):
    """
    Модель крови как единого резервуара.
    Состояние: [V_blood, C_0, C_1, ..., C_n] — объём крови (мл) и концентрации
    различных веществ (в у.е./мл или других единицах).

    Параметры:
        substance_names : список названий веществ, определяющий порядок концентраций.
        V0 : начальный объём крови (мл)
        initial_concentrations : словарь {имя: начальное значение} или список той же длины,
                                 что и substance_names. Если не задано, все концентрации = 0.
    """
    def __init__(self,
                 substance_names: List[str],
                 V0: float = 5000.0,
                 initial_concentrations: Dict[str, float] = None):
        self.substance_names = substance_names.copy()
        self.num_substances = len(substance_names)
        self.V0 = V0

        # Начальные концентрации
        if initial_concentrations is None:
            self.C0 = np.zeros(self.num_substances)
        else:
            self.C0 = np.array([initial_concentrations.get(name, 0.0)
                                for name in substance_names])

        self._current_state = None   # для возможного кэширования, не обязательно

    def get_state_size(self) -> int:
        """Возвращает 1 (объём) + число веществ."""
        return 1 + self.num_substances

    def get_initial_state(self) -> np.ndarray:
        """Начальное состояние: [V0, C0_0, C0_1, ...]."""
        return np.concatenate(([self.V0], self.C0))

    def get_derivatives(self, t: float, state_slice: np.ndarray,
                        inputs: Dict[str, Any]) -> np.ndarray:
        """
        Вычисляет производные состояния крови.
        Ожидаемые ключи в inputs:
            'dV' : скорость изменения объёма крови (мл/с)
            'dC' : список или массив скоростей изменения концентраций
                   (той же длины, что и число веществ). Если не задан, считается нулевым.
        """
        # state_slice: [V, C0, C1, ...]
        V = state_slice[0]
        # Производная объёма
        dV = inputs.get('dV', 0.0)

        # Производные концентраций
        dC_input = inputs.get('dC', None)
        if dC_input is None:
            dC = np.zeros(self.num_substances)
        else:
            dC = np.asarray(dC_input)
            if dC.shape[0] != self.num_substances:
                raise ValueError(f"dC должен быть длины {self.num_substances}, "
                                 f"получено {dC.shape[0]}")

        # Возвращаем [dV, dC0, dC1, ...]
        return np.concatenate(([dV], dC))

    def get_outputs(self, state_slice: np.ndarray) -> Dict[str, float]:
        """
        Возвращает словарь с текущими значениями объёма и концентраций.
        Ключи: 'V_blood' и для каждого вещества 'C_<имя>'.
        """
        V = state_slice[0]
        concentrations = state_slice[1:]
        outputs = {'V_blood': V}
        for name, value in zip(self.substance_names, concentrations):
            outputs[f'C_{name}'] = value
        return outputs