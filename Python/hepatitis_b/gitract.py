# gitract.py
import numpy as np
from organ_base import OrganModel


class GITract(OrganModel):
    """
    Модель желудочно-кишечного тракта с учётом портальной гемодинамики
    и влияния повышенного портального давления (например, при циррозе,
    вызванном гепатитом B) на всасывание и кровоток.

    Состояние: [P_art, P_cap] — давление в артериальном и капиллярном
                компартментах (мм рт. ст.)

    Входные данные (inputs):
        P_sa (float) : системное артериальное давление
        P_portal (float) : давление в воротной вене (мм рт. ст.)
        intake_water (float) : поступление воды в просвет (мл/с)
        intake_nutrients (float) : поступление питательных веществ (г/с)

    Выходные данные (outputs):
        Q_out (float) : поток из капилляров в портальную систему (мл/с)
        absorption_water (float) : скорость всасывания воды (мл/с)
        absorption_nutrients (float) : скорость всасывания питательных веществ (г/с)
    """
    def __init__(self,
                 R_art=0.3,          # сопротивление артериол (мм рт. ст.·с/мл)
                 R_cap=0.2,          # сопротивление капилляров (между артериолами и капиллярами)
                 R_venous=0.2,        # сопротивление венозного оттока в портальную систему
                 C_art=2.0,           # податливость артериального отдела (мл/мм рт. ст.)
                 C_cap=5.0,           # податливость капиллярного отдела
                 k_absorption_water=0.1,   # базовая константа скорости всасывания воды (1/с)
                 k_absorption_nutrients=0.05, # базовая константа скорости всасывания питательных веществ (1/с)
                 portal_pressure_sensitivity=0.02, # чувствительность всасывания к портальному давлению (1/мм рт. ст.)
                 P_art0=75.0, P_cap0=10.0): # начальные давления

        self.R_art = R_art
        self.R_cap = R_cap
        self.R_venous = R_venous
        self.C_art = C_art
        self.C_cap = C_cap
        self.k_abs_water = k_absorption_water
        self.k_abs_nutrients = k_absorption_nutrients
        self.portal_pressure_sensitivity = portal_pressure_sensitivity
        self.P_art0 = P_art0
        self.P_cap0 = P_cap0
        self._current_outputs = {}

    def get_state_size(self):
        return 2

    def get_initial_state(self):
        return np.array([self.P_art0, self.P_cap0])

    def _absorption_factor(self, P_portal):
        """
        Коэффициент снижения всасывания при повышении портального давления.
        Линейное уменьшение от 1 до минимального значения 0.2.
        """
        # пороговое давление считаем равным нормальному (около 8 мм рт. ст.)
        # если P_portal <= norm, factor = 1
        norm_p = 8.0
        if P_portal <= norm_p:
            return 1.0
        else:
            factor = 1.0 - self.portal_pressure_sensitivity * (P_portal - norm_p)
            return max(0.2, factor)

    def get_derivatives(self, t, state, inputs):
        P_art, P_cap = state
        P_sa = inputs.get('P_sa', 80.0)
        # Если портальное давление не передано, используем системное венозное (упрощение)
        P_portal = inputs.get('P_portal', inputs.get('P_sv', 5.0))
        intake_water = inputs.get('intake_water', 0.0)
        intake_nutrients = inputs.get('intake_nutrients', 0.0)

        # Потоки
        Q_in = (P_sa - P_art) / self.R_art                     # приток из системных артерий
        Q_cap = (P_art - P_cap) / self.R_cap                   # поток через капилляры
        Q_out = (P_cap - P_portal) / self.R_venous              # отток в портальную систему

        # Коэффициент влияния портальной гипертензии на всасывание
        abs_factor = self._absorption_factor(P_portal)

        # Абсорбция
        absorption_water = self.k_abs_water * intake_water * abs_factor
        absorption_nutrients = self.k_abs_nutrients * intake_nutrients * abs_factor

        # Изменение давлений
        dP_art = (Q_in - Q_cap) / self.C_art
        dP_cap = (Q_cap + absorption_water - Q_out) / self.C_cap   # добавка от всасывания воды

        # Сохраняем выходы
        self._current_outputs = {
            'Q_out': Q_out,
            'absorption_water': absorption_water,
            'absorption_nutrients': absorption_nutrients,
            'portal_pressure_factor': abs_factor   # для диагностики
        }

        return np.array([dP_art, dP_cap])

    def get_outputs(self, state):
        return self._current_outputs.copy()