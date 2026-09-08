# baroreflex.py
import numpy as np
from organ_base import OrganModel

class Baroreflex(OrganModel):
    """
    Модель барорефлекторной регуляции частоты сердечных сокращений.
    Состояние: [HR_current] — текущая ЧСС (уд/мин).
    """
    def __init__(self, P_set=90.0, HR_base=70.0, gain=0.01, tau=2.0):
        """
        Параметры:
            P_set   – заданное давление (мм рт. ст.), при котором ЧСС = HR_base
            HR_base – базовая ЧСС (уд/мин)
            gain    – коэффициент усиления (относительное изменение ЧСС на 1 мм рт. ст.)
            tau     – постоянная времени рефлекса (с)
        """
        self.P_set = P_set
        self.HR_base = HR_base
        self.gain = gain
        self.tau = tau
        self._current_outputs = {}

    def get_state_size(self) -> int:
        return 1   # только HR_current

    def get_initial_state(self) -> np.ndarray:
        return np.array([self.HR_base])

    def get_derivatives(self, t, state, inputs):
        HR = state[0]
        P_sa = inputs.get('P_sa', 90.0)

        # Целевая ЧСС (обратная зависимость от давления)
        HR_target = self.HR_base * (1.0 - self.gain * (P_sa - self.P_set))
        HR_target = np.clip(HR_target, 40.0, 180.0)

        # Производная (линейная динамика первого порядка)
        dHR = (HR_target - HR) / self.tau

        self._current_outputs = {
            'HR': HR,
            'HR_target': HR_target,
            'hr_factor': HR / self.HR_base
        }
        return np.array([dHR])

    def get_outputs(self, state):
        return self._current_outputs.copy()