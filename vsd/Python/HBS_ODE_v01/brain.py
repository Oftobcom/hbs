# brain.py
import numpy as np
from organ_base import OrganModel


class Brain(OrganModel):
    """
    Модель головного мозга с учётом гемодинамики, метаболизма и влияния
    токсических веществ (например, аммиака), характерных для печёночной
    энцефалопатии при гепатите B.

    Состояние: [P_br] — давление в мозговых сосудах (мм рт. ст.)

    Входные данные (inputs):
        P_sa (float) : системное артериальное давление
        P_sv (float) : системное венозное давление
        C_ammonia (float) : концентрация аммиака в крови (у.е./мл)
        (опционально) другие токсины могут быть добавлены аналогично

    Выходные данные (outputs):
        Q_br (float) : кровоток через мозг (мл/с)
        O2_consumption (float) : потребление кислорода (у.е./с)
        glucose_consumption (float) : потребление глюкозы (у.е./с)
        metabolic_inhibition (float) : коэффициент ингибирования метаболизма
    """
    def __init__(self,
                 R_base=0.8,                # базовое сопротивление (мм рт. ст.·с/мл)
                 C=2.0,                      # податливость (мл/мм рт. ст.)
                 autoreg_gain=0.02,          # чувствительность авторегуляции
                 P_autoreg=80.0,             # давление авторегуляции (мм рт. ст.)
                 O2_extraction=0.4,           # доля извлекаемого кислорода в норме
                 glucose_extraction=0.1,      # доля извлекаемой глюкозы в норме
                 ammonia_inhibition_factor=0.1,  # чувствительность к аммиаку (1/у.е.)
                 ammonia_threshold=0.5,        # пороговая концентрация аммиака (у.е./мл)
                 P0=60.0):                     # начальное давление (мм рт. ст.)

        self.R_base = R_base
        self.C = C
        self.autoreg_gain = autoreg_gain
        self.P_autoreg = P_autoreg
        self.O2_extraction = O2_extraction
        self.glucose_extraction = glucose_extraction
        self.ammonia_inhibition_factor = ammonia_inhibition_factor
        self.ammonia_threshold = ammonia_threshold
        self.P0 = P0
        self._current_outputs = {}

    def get_state_size(self):
        return 1

    def get_initial_state(self):
        return np.array([self.P0])

    def _autoregulation_resistance(self, P_sa):
        """Вычисляет эффективное сопротивление с учётом авторегуляции."""
        delta = P_sa - self.P_autoreg
        reg = 1.0 - self.autoreg_gain * delta   # при повышении давления сопротивление растёт
        reg = max(0.5, min(reg, 2.0))
        return self.R_base * reg

    def _ammonia_inhibition(self, C_ammonia):
        """
        Рассчитывает коэффициент ингибирования метаболизма из-за аммиака.
        При превышении порога метаболизм линейно снижается.
        """
        if C_ammonia <= self.ammonia_threshold:
            return 1.0
        else:
            # ингибирование: factor = 1 - k * (C - threshold)
            inhibition = 1.0 - self.ammonia_inhibition_factor * (C_ammonia - self.ammonia_threshold)
            return max(0.2, inhibition)   # не опускаемся ниже 20% нормы

    def get_derivatives(self, t, state, inputs):
        P_br = state[0]
        P_sa = inputs.get('P_sa', 80.0)
        P_sv = inputs.get('P_sv', 5.0)
        C_ammonia = inputs.get('C_ammonia', 0.0)   # концентрация аммиака

        # Сопротивление с авторегуляцией
        R_eff = self._autoregulation_resistance(P_sa)

        # Кровоток
        Q_br = (P_sa - P_br) / R_eff
        Q_out = (P_br - P_sv) / R_eff

        # Изменение давления в сосудах мозга
        dP_br = (Q_br - Q_out) / self.C

        # Влияние аммиака на метаболизм
        metab_factor = self._ammonia_inhibition(C_ammonia)

        # Потребление кислорода и глюкозы с учётом ингибирования
        O2_consumption = Q_br * self.O2_extraction * metab_factor
        glucose_consumption = Q_br * self.glucose_extraction * metab_factor

        # Сохраняем выходы для get_outputs
        self._current_outputs = {
            'Q_br': Q_br,
            'O2_consumption': O2_consumption,
            'glucose_consumption': glucose_consumption,
            'metabolic_inhibition': metab_factor   # степень угнетения
        }

        return np.array([dP_br])

    def get_outputs(self, state):
        return self._current_outputs.copy()