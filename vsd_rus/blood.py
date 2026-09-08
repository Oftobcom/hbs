# blood.py - ИСПРАВЛЕННАЯ ВЕРСИЯ P0
import numpy as np
from organ_base import OrganModel
from typing import List, Dict, Any

class BloodPool(OrganModel):
    """
    Модель крови как единого резервуара.
    Состояние: [V_blood, C_0, C_1, ..., C_n] — объём (мл) и концентрации (масса/мл).
    
    ИСПРАВЛЕНИЕ P0:
    - dV теперь = absorption + iv - urine - insensible (считается в whole_body)
    - Добавлен учет разбавления: при увеличении V концентрации падают, даже если dC=0
    - Добавлена защита от отрицательного объема
    """
    def __init__(self,
                 substance_names: List[str],
                 V0: float = 5000.0,
                 initial_concentrations: Dict[str, float] = None):
        self.substance_names = substance_names.copy()
        self.num_substances = len(substance_names)
        self.V0 = V0
        if initial_concentrations is None:
            self.C0 = np.zeros(self.num_substances)
        else:
            self.C0 = np.array([initial_concentrations.get(name, 0.0)
                                for name in substance_names])
        self._current_state = None

    def get_state_size(self) -> int:
        return 1 + self.num_substances

    def get_initial_state(self) -> np.ndarray:
        return np.concatenate(([self.V0], self.C0))

    def get_derivatives(self, t: float, state_slice: np.ndarray,
                        inputs: Dict[str, Any]) -> np.ndarray:
        V = state_slice[0]
        C = state_slice[1:]
        dV = inputs.get('dV', 0.0)
        dC_input = inputs.get('dC', None)
        
        if dC_input is None:
            dM_dt = np.zeros(self.num_substances)  # mass rate
        else:
            dC_raw = np.asarray(dC_input)
            if dC_raw.shape[0] != self.num_substances:
                raise ValueError(f"dC должен иметь длину {self.num_substances}, получено {dC_raw.shape[0]}")
            # dC_input из органов - это уже dC (концентрация/мл /с), но с учетом массы
            # Для корректного баланса: d(V*C)/dt = V*dC + C*dV = mass_rate
            # Органы отдают mass_rate / V = dC, поэтому:
            # dC_true = dC_input - C*dV/V  (разбавление)
            # Чтобы сохранить обратную совместимость, делаем опционально
            # Если V>0, применяем коррекцию разбавления
            dM_dt = dC_raw * max(V, 1.0)  # переводим dC в массу, если вход был как dC
            # На самом деле liver/kidney уже делят на V, поэтому оставляем как есть
            # и добавляем разбавление отдельно

        # Защита от отрицательного объема - не даем упасть ниже 1000 мл
        if V < 1500 and dV < 0:
            dV = max(dV, (1500 - V) / 0.1)  # мягкий барьер

        # Расчет dC с учетом разбавления
        # Если dC_input = mass_rate / V, то полный dC = mass_rate/V - C*dV/V
        if dC_input is None:
            dC = np.zeros(self.num_substances)
        else:
            dC_mass = np.asarray(dC_input)  # уже как dC
            # Коррекция разбавления: при росте V концентрация падает
            if V > 1e-6:
                dC_dilution = -C * dV / V
                dC = dC_mass + dC_dilution
            else:
                dC = dC_mass

        return np.concatenate(([dV], dC))

    def get_outputs(self, state_slice: np.ndarray) -> Dict[str, float]:
        V = state_slice[0]
        concentrations = state_slice[1:]
        outputs = {'V_blood': V}
        for name, value in zip(self.substance_names, concentrations):
            outputs[f'C_{name}'] = value
        return outputs
