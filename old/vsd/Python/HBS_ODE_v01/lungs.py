# lungs.py
import numpy as np
from organ_base import OrganModel


class Lungs2Chamber(OrganModel):
    """
    Двухкамерная модель лёгочного сосудистого русла с возможностью модуляции
    сопротивлений под влиянием повреждения печени (портопульмональная гипертензия)
    и простой оценкой эффективности оксигенации (гепатопульмональный синдром).

    Состояние: [P_prox, P_dist] — давление в проксимальном и дистальном отделах (мм рт. ст.)

    Входные данные (inputs):
        Q_pulmonary (float) : поток из правого желудочка (мл/с)
        P_pv (float) : давление в лёгочных венах (мм рт. ст.)
        liver_damage (float, optional) : степень повреждения печени (0..1), по умолчанию 0
        (может быть использовано для модуляции сопротивлений)

    Выходные данные (outputs):
        P_pa (float) : давление в лёгочной артерии (мм рт. ст.)
        oxygenation_index (float) : индекс эффективности оксигенации (0..1, 1 – норма)
        (опционально, зависит от shunt_fraction)

    Параметры:
        R1_base, R2_base : базовые сопротивления (мм рт. ст.·с/мл)
        C1, C2 : податливости (мл/мм рт. ст.)
        sensitivity_R1, sensitivity_R2 : чувствительность сопротивлений к повреждению печени
        shunt_base : базовая доля шунта (0..1)
        shunt_sensitivity : увеличение шунта при повреждении печени
    """
    def __init__(self,
                 R1_base=0.5, R2_base=0.5,
                 C1=2.0, C2=3.0,
                 sensitivity_R1=0.5,    # увеличение R1 на 50% при damage=1
                 sensitivity_R2=0.5,
                 shunt_base=0.02,        # нормальный шунт 2%
                 shunt_sensitivity=0.1): # при damage=1 шунт увеличивается на 10% (до 12%)
        self.R1_base = R1_base
        self.R2_base = R2_base
        self.C1 = C1
        self.C2 = C2
        self.sensitivity_R1 = sensitivity_R1
        self.sensitivity_R2 = sensitivity_R2
        self.shunt_base = shunt_base
        self.shunt_sensitivity = shunt_sensitivity
        self._current_outputs = {}

    def get_state_size(self):
        return 2

    def get_initial_state(self):
        return np.array([15.0, 10.0])   # P_prox, P_dist

    def _effective_resistance(self, base, damage, sensitivity):
        """Вычисляет сопротивление с учётом повреждения печени."""
        # линейное увеличение: R = base * (1 + sensitivity * damage)
        return base * (1.0 + sensitivity * damage)

    def _shunt_fraction(self, damage):
        """Доля крови, не участвующая в газообмене (внутрилёгочное шунтирование)."""
        # линейно растёт с повреждением
        shunt = self.shunt_base + self.shunt_sensitivity * damage
        return min(shunt, 0.5)   # ограничим разумным пределом

    def get_derivatives(self, t, state, inputs):
        P_prox, P_dist = state
        Q_pulm = inputs.get('Q_pulmonary', 0.0)
        P_pv = inputs.get('P_pv', 5.0)
        damage = inputs.get('liver_damage', 0.0)   # степень повреждения печени

        # Актуальные сопротивления
        R1 = self._effective_resistance(self.R1_base, damage, self.sensitivity_R1)
        R2 = self._effective_resistance(self.R2_base, damage, self.sensitivity_R2)

        # Уравнения гемодинамики
        dP_prox = (Q_pulm - (P_prox - P_dist) / R1) / self.C1
        dP_dist = ((P_prox - P_dist) / R1 - (P_dist - P_pv) / R2) / self.C2

        # Расчёт индекса оксигенации (упрощённо)
        shunt = self._shunt_fraction(damage)
        oxygenation_index = 1.0 - shunt   # чем больше шунт, тем хуже оксигенация

        # Сохраняем выходы
        self._current_outputs = {
            'P_pa': P_prox,
            'oxygenation_index': oxygenation_index,
            'shunt_fraction': shunt
        }

        return np.array([dP_prox, dP_dist])

    def get_outputs(self, state):
        return self._current_outputs.copy()