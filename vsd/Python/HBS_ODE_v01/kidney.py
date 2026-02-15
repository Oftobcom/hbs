# kidney.py
import numpy as np
from organ_base import OrganModel


class KidneyHemodynamic(OrganModel):
    """
    Модель почек с авторегуляцией GFR, влиянием на объём крови и клиренс токсинов.
    Модель расширена для учёта гепаторенального синдрома при гепатите B:
    - Снижение GFR и увеличение сосудистого сопротивления при высоком уровне
      билирубина или аммиака.
    - Возможность модуляции реабсорбции (опционально).

    Не имеет собственных состояний (статические вычисления).
    """
    def __init__(self,
                 GFR_base=120.0,                 # базовая СКФ (мл/мин) в норме
                 P_autoreg=95.0,                  # давление авторегуляции (мм рт. ст.)
                 autoreg_gain=0.02,                # чувствительность авторегуляции
                 toxin_clearance_frac=0.2,         # доля отфильтрованного токсина, выводимая с мочой
                 volume_reabsorption_frac=0.99,    # доля реабсорбции воды (0.99 -> 1% мочи)
                 renal_resistance=0.02,             # базовое сопротивление почечных сосудов (мм рт. ст.·с/мл)
                 # Параметры гепаторенального синдрома
                 bilirubin_threshold=2.0,           # порог билирубина (у.е./мл)
                 bilirubin_effect_on_gfr=0.2,       # снижение GFR на 20% за каждую единицу выше порога
                 ammonia_threshold=1.0,              # порог аммиака (у.е./мл)
                 ammonia_effect_on_gfr=0.15,         # снижение GFR на 15% за каждую единицу выше порога
                 max_impairment=0.8):                 # максимальное снижение (до 20% от нормы)

        self.GFR_base = GFR_base / 2.0   # на одну почку (суммарный GFR = GFR_base)
        self.P_autoreg = P_autoreg
        self.autoreg_gain = autoreg_gain
        self.toxin_clearance_frac = toxin_clearance_frac
        self.volume_reabsorption_frac = volume_reabsorption_frac
        self.renal_resistance = renal_resistance

        self.bilirubin_threshold = bilirubin_threshold
        self.bilirubin_effect = bilirubin_effect_on_gfr
        self.ammonia_threshold = ammonia_threshold
        self.ammonia_effect = ammonia_effect_on_gfr
        self.max_impairment = max_impairment

        self._current_outputs = {}

    def get_state_size(self):
        return 0   # нет внутренних состояний

    def get_initial_state(self):
        return np.array([])

    def get_derivatives(self, t, state, inputs):
        # Нет производных собственных состояний
        return np.array([])

    def get_outputs(self, state):
        return self._current_outputs.copy()

    def _impairment_factor(self, C_bilirubin, C_ammonia):
        """
        Вычисляет общий коэффициент снижения функции почек (от 1 до max_impairment)
        на основе концентраций билирубина и аммиака.
        """
        factor = 1.0
        if C_bilirubin > self.bilirubin_threshold:
            excess = C_bilirubin - self.bilirubin_threshold
            factor *= max(1.0 - self.bilirubin_effect * excess, self.max_impairment)
        if C_ammonia > self.ammonia_threshold:
            excess = C_ammonia - self.ammonia_threshold
            factor *= max(1.0 - self.ammonia_effect * excess, self.max_impairment)
        return factor

    def _compute_gfr(self, P_art, impairment_factor):
        """
        СКФ с учётом авторегуляции и гепаторенального повреждения.
        """
        # Авторегуляция
        delta = P_art - self.P_autoreg
        reg = 1.0 + self.autoreg_gain * delta
        reg = max(0.2, min(reg, 1.8))
        # Применяем фактор повреждения
        return 2 * self.GFR_base * reg * impairment_factor   # суммарный GFR обеих почек

    def compute_effects(self, P_sa, P_sv, C_tox, V_blood, C_bilirubin=0.0, C_ammonia=0.0):
        """
        Вычисляет влияние почек на объём крови и концентрации веществ.

        Параметры:
            P_sa : системное артериальное давление (мм рт. ст.)
            P_sv : системное венозное давление (мм рт. ст.)
            C_tox : концентрация общего токсина (у.е./мл)
            V_blood : объём крови (мл)
            C_bilirubin : концентрация билирубина (у.е./мл) – опционально
            C_ammonia : концентрация аммиака (у.е./мл) – опционально

        Возвращает словарь:
            'dC_tox' : производная концентрации токсина (у.е./мл/с)
            'dV_blood' : производная объёма крови (мл/с)
            'Q_renal' : почечный кровоток (мл/с)
            'impairment_factor' : коэффициент снижения функции (для диагностики)
            'GFR' : текущая СКФ (мл/с)
        """
        # Коэффициент повреждения
        impair = self._impairment_factor(C_bilirubin, C_ammonia)

        # Эффективное сопротивление (может расти при повреждении, например, из-за вазоконстрикции)
        # Для простоты увеличиваем сопротивление пропорционально снижению функции:
        # чем выше impair (ближе к 1), тем меньше сопротивление; при impair < 1 сопротивление растёт.
        # Используем зависимость R_eff = renal_resistance / impair
        R_eff = self.renal_resistance / max(impair, 0.2)

        # Почечный кровоток
        Q_renal = (P_sa - P_sv) / R_eff

        # GFR с учётом повреждения
        GFR = self._compute_gfr(P_sa, impair)

        # Клиренс токсина
        toxin_filtered = GFR * C_tox
        toxin_excreted = self.toxin_clearance_frac * toxin_filtered
        dC_tox = -toxin_excreted / max(V_blood, 1e-6)

        # Объём мочи и изменение объёма крови
        urine_output = GFR * (1 - self.volume_reabsorption_frac)
        dV_blood = -urine_output

        # Сохраняем для get_outputs (можно использовать при необходимости)
        self._current_outputs = {
            'Q_renal': Q_renal,
            'GFR': GFR,
            'impairment_factor': impair
        }

        return {
            'dC_tox': dC_tox,
            'dV_blood': dV_blood,
            'Q_renal': Q_renal,
            'GFR': GFR,
            'impairment_factor': impair
        }