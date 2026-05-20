# blood.py
import numpy as np
from organ_base import OrganModel
from typing import List, Dict, Any

class BloodPool(OrganModel):
    """
    Blood as a single reservoir.
    State: [V_blood, C_0, C_1, ..., C_n] – volume (ml) and concentrations.
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
        dV = inputs.get('dV', 0.0)
        dC_input = inputs.get('dC', None)
        if dC_input is None:
            dC = np.zeros(self.num_substances)
        else:
            dC = np.asarray(dC_input)
            if dC.shape[0] != self.num_substances:
                raise ValueError(f"dC length {self.num_substances} expected, got {dC.shape[0]}")
        return np.concatenate(([dV], dC))

    def get_outputs(self, state_slice: np.ndarray) -> Dict[str, float]:
        V = state_slice[0]
        concentrations = state_slice[1:]
        outputs = {'V_blood': V}
        for name, value in zip(self.substance_names, concentrations):
            outputs[f'C_{name}'] = value
        return outputs