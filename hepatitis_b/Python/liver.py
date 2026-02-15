# liver.py
import numpy as np
from organ_base import OrganModel
from typing import Dict, Any


class Liver(OrganModel):
    """
    Модель печени, включающая гемодинамику (артерия, воротная вена, печёночные вены)
    и метаболические функции: синтез альбумина, метаболизм билирубина,
    детоксикация аммиака, продукция факторов свёртывания (агрегированно).
    Моделируется влияние гепатита B через повреждение гепатоцитов (параметр damage)
    и вирусную нагрузку (C_virus). Повреждение влияет на метаболические скорости
    и сосудистые сопротивления (портальная гипертензия).

    Состояние (8 переменных):
        P_hv           : давление в печёночных венах (мм рт. ст.)
        C_bilirubin    : концентрация билирубина в печени (у.е./мл) – отражает количество,
                         которое предстоит конъюгировать/экскретировать
        C_ammonia      : концентрация аммиака в печени (у.е./мл)
        C_albumin      : концентрация альбумина в печени (для синтеза)
        C_virus        : вирусная нагрузка (у.е./мл) – внутрипечёночная
        damage         : степень повреждения (0..1, 0 – норма, 1 – тяжёлое повреждение)
        reserve_1      : запас для будущих расширений
        reserve_2      : запас

    Входы (inputs):
        P_sa           : системное артериальное давление (мм рт. ст.)
        P_sv           : системное венозное давление (мм рт. ст.)
        C_bilirubin_blood : концентрация билирубина в крови (у.е./мл)
        C_ammonia_blood   : концентрация аммиака в крови (у.е./мл)
        C_albumin_blood   : концентрация альбумина в крови (у.е./мл)
        (другие субстраты могут быть добавлены аналогично)

    Выходы (outputs):
        Q_liver_out    : поток из печени в системные вены (мл/с)
        P_portal       : давление в воротной вене (мм рт. ст.) – для ЖКТ
        dC_bilirubin   : скорость изменения концентрации билирубина в крови (у.е./мл/с)
        dC_ammonia     : скорость изменения концентрации аммиака в крови
        dC_albumin     : скорость изменения концентрации альбумина в крови
        dC_virus       : скорость изменения вирусной нагрузки (для обратной связи)
        damage_output  : текущее повреждение (для диагностики)
    """

    def __init__(self,
                 # Гемодинамические параметры
                 R_ha=0.5,          # сопротивление печёночной артерии (мм рт. ст.·с/мл)
                 R_pv_base=0.3,      # базовое сопротивление воротной вены
                 R_hv_base=0.2,      # базовое сопротивление печёночных вен
                 C=5.0,               # податливость печёночного сосудистого русла (мл/мм рт. ст.)
                 P_hv0=5.0,           # начальное давление в печёночных венах
                 # Метаболические параметры
                 albumin_prod_base=0.1,   # базовая скорость синтеза альбумина (у.е./с)
                 bilirubin_clearance_base=0.2, # базовая скорость клиренса билирубина (1/с)
                 ammonia_clearance_base=0.15,   # базовая скорость клиренса аммиака (1/с)
                 virus_growth_rate=0.05,         # скорость размножения вируса (1/с)
                 virus_killing_rate=0.02,        # скорость уничтожения вируса иммунитетом (1/с)
                 damage_progression_rate=0.01,    # скорость нарастания повреждения при высокой вирусной нагрузке
                 damage_regression_rate=0.005,    # скорость восстановления при низкой нагрузке
                 # Влияние повреждения на сопротивления
                 resistance_increase_factor=2.0,   # максимальное увеличение сопротивления при damage=1
                 # Начальные значения
                 C_bilirubin0=0.0,
                 C_ammonia0=0.0,
                 C_albumin0=1.0,
                 C_virus0=0.0,
                 damage0=0.0):

        self.R_ha = R_ha
        self.R_pv_base = R_pv_base
        self.R_hv_base = R_hv_base
        self.C = C
        self.P_hv0 = P_hv0

        self.albumin_prod_base = albumin_prod_base
        self.bilirubin_clearance_base = bilirubin_clearance_base
        self.ammonia_clearance_base = ammonia_clearance_base
        self.virus_growth_rate = virus_growth_rate
        self.virus_killing_rate = virus_killing_rate
        self.damage_progression_rate = damage_progression_rate
        self.damage_regression_rate = damage_regression_rate
        self.resistance_increase_factor = resistance_increase_factor

        # Начальные состояния
        self.C_bilirubin0 = C_bilirubin0
        self.C_ammonia0 = C_ammonia0
        self.C_albumin0 = C_albumin0
        self.C_virus0 = C_virus0
        self.damage0 = damage0

        self._current_outputs = {}

    def get_state_size(self):
        return 8   # P_hv + 5 основных + 2 резерва

    def get_initial_state(self):
        return np.array([
            self.P_hv0,
            self.C_bilirubin0,
            self.C_ammonia0,
            self.C_albumin0,
            self.C_virus0,
            self.damage0,
            0.0,   # reserve 1
            0.0    # reserve 2
        ])

    def _effective_resistance(self, base_resistance, damage):
        """
        Сопротивление увеличивается с повреждением (портальная гипертензия).
        """
        factor = 1.0 + damage * (self.resistance_increase_factor - 1.0)
        return base_resistance * factor

    def get_derivatives(self, t, state, inputs: Dict[str, Any]):
        P_hv = state[0]
        C_bil = state[1]
        C_amm = state[2]
        C_alb = state[3]
        C_virus = state[4]
        damage = state[5]

        # Входные данные (давления и концентрации в крови)
        P_sa = inputs['P_sa']
        P_sv = inputs['P_sv']
        C_bil_blood = inputs.get('C_bilirubin_blood', 0.0)
        C_amm_blood = inputs.get('C_ammonia_blood', 0.0)
        C_alb_blood = inputs.get('C_albumin_blood', 1.0)  # норма

        # --- Гемодинамика ---
        # Давление в мезентериальных сосудах (упрощённо)
        P_mes = (P_sa + P_sv) / 2

        # Сопротивления с учётом повреждения
        R_pv = self._effective_resistance(self.R_pv_base, damage)
        R_hv = self._effective_resistance(self.R_hv_base, damage)

        Q_ha = (P_sa - P_hv) / self.R_ha
        Q_pv = (P_mes - P_hv) / R_pv
        Q_out = (P_hv - P_sv) / R_hv

        dP_hv = (Q_ha + Q_pv - Q_out) / self.C

        # Давление в воротной вене (на входе в печень) – аппроксимируем как P_mes
        P_portal = P_mes   # можно уточнить

        # --- Метаболизм и повреждение ---
        # Коэффициент функциональности (1 – повреждение)
        functional = max(0.2, 1.0 - damage)

        # Билирубин: захват из крови, конъюгация и экскреция
        # Упрощённо: скорость изменения билирубина в печени = поступление из крови (пропорционально разности) - клиренс
        uptake_bil = 0.1 * (C_bil_blood - C_bil)   # простой обмен
        clearance_bil = self.bilirubin_clearance_base * functional * C_bil
        dC_bil = uptake_bil - clearance_bil

        # Аммиак: аналогично
        uptake_amm = 0.1 * (C_amm_blood - C_amm)
        clearance_amm = self.ammonia_clearance_base * functional * C_amm
        dC_amm = uptake_amm - clearance_amm

        # Альбумин: синтез (зависит от функциональности) и потребление (упрощённо)
        synthesis_alb = self.albumin_prod_base * functional
        degradation_alb = 0.01 * C_alb   # базовая деградация
        # Переход альбумина в кровь (пропорционально разности)
        release_alb = 0.05 * (C_alb - C_alb_blood)
        dC_alb = synthesis_alb - degradation_alb - release_alb

        # Вирусная динамика (очень упрощённая модель)
        # Рост пропорционален текущей нагрузке и функциональности? Вирус размножается в гепатоцитах,
        # поэтому при повреждении может размножаться хуже? Для простоты: рост пропорционален C_virus,
        # а уничтожение пропорционально C_virus и, возможно, функциональности (иммунитет).
        # В реальности сложнее, но для иллюстрации:
        dC_virus = self.virus_growth_rate * C_virus * (1 - damage) - self.virus_killing_rate * C_virus

        # Повреждение: растёт при высокой вирусной нагрузке, снижается при низкой
        if C_virus > 0.1:
            d_damage = self.damage_progression_rate * (1 - damage) * C_virus
        else:
            d_damage = -self.damage_regression_rate * damage

        # Ограничиваем damage в [0,1]
        d_damage = np.clip(d_damage, -0.01, 0.01)   # предотвращаем резкие скачки

        # --- Влияние на кровь (скорости изменения концентраций в крови) ---
        # Для каждого вещества вычисляем чистый поток из печени в кровь (или наоборот)
        # Билирубин: в кровь поступает конъюгированный билирубин (clearance_bil) – считаем, что он выводится в желчь,
        # но для простоты предположим, что клиренс удаляет билирубин из крови, т.е. dC_bil_blood = -clearance_bil / V_blood?
        # Однако мы не знаем V_blood здесь. Эти производные будут суммироваться в BloodPool.
        # Поэтому вернём скорости изменения концентраций в крови (положительные – добавление в кровь).
        # Обычно печень выделяет альбумин в кровь, удаляет аммиак и билирубин.
        # Примем, что:
        # - Билирубин удаляется из крови со скоростью clearance_bil (но clearance_bil – это внутрипечёночный клиренс,
        #   который должен уменьшать концентрацию в крови). Лучше считать, что скорость удаления из крови
        #   пропорциональна разности концентраций и функциональности.
        # Упростим: пусть dC_bil_blood = - k_bil * functional * C_bil_blood   (линейное удаление)
        # Аналогично для аммиака.
        # Для альбумина: dC_alb_blood = + k_alb * functional * (C_alb - C_alb_blood)   (выделение)
        # Для простоты оставим эти расчёты здесь, но тогда нужно знать объём крови? Нет, мы возвращаем именно производные концентрации,
        # которые будут напрямую добавлены в BloodPool. BloodPool ожидает dC (скорость изменения концентрации).
        # Поэтому здесь мы можем вычислить эти скорости, используя текущие концентрации и параметры.
        # Для этого нам нужен доступ к объёму крови? Нет, объём крови уже учтён в BloodPool при интегрировании.
        # Мы возвращаем именно dC/dt (изменение концентрации в единицу времени).
        # Например, dC_bil_blood = - (uptake_bil) / V_blood? Но V_blood неизвестно.
        # Лучше перенести эти расчёты в BloodPool, а здесь возвращать только внутрипечёночные потоки (массы в секунду),
        # а BloodPool уже поделит на объём. Но интерфейс OrganModel не предусматривает возврат масс.
        # В текущей архитектуре BloodPool сам интегрирует концентрации и принимает в inputs dC (скорости изменения концентрации).
        # Эти dC должны быть рассчитаны органами. Для этого орган должен знать объём крови или передавать массу.
        # Более последовательно: пусть Liver возвращает в outputs массы веществ, поступающих в кровь или удаляемых из неё за секунду,
        # а BloodPool, зная свой объём, преобразует массу в изменение концентрации.
        # Но текущий интерфейс get_outputs возвращает только словарь скаляров, не обязательно dC.
        # Можно договориться, что в outputs мы возвращаем массовые потоки, а BloodPool их использует.
        # Однако BloodPool ожидает в inputs dC (концентрационные), а не массовые. Значит, нужно изменить BloodPool, чтобы он принимал массовые потоки и делил на объём.
        # Но мы уже модифицировали BloodPool в предыдущем ответе – он принимает dC напрямую. Значит, Liver должен вычислять именно dC.
        # Для этого ему нужно знать объём крови. Передавать V_blood через inputs? Это возможно.
        # Добавим в inputs ключ 'V_blood' (текущий объём крови) от BloodPool.

        # Будем считать, что в inputs передаётся V_blood.
        V_blood = inputs.get('V_blood', 5000.0)

        # Теперь можем рассчитать dC для крови:
        # Билирубин: удаление из крови пропорционально разности (C_bil_blood - C_bil) и функциональности?
        # Но лучше использовать клиренс на основе внутрипечёночной концентрации.
        # Для простоты используем простую модель:
        dC_bil_blood = - clearance_bil / V_blood   # удаление билирубина (clearance_bil – масса/с)
        dC_amm_blood = - clearance_amm / V_blood   # удаление аммиака
        dC_alb_blood = + release_alb / V_blood     # добавление альбумина

        # Сохраняем выходы
        self._current_outputs = {
            'Q_liver_out': Q_out,
            'P_portal': P_portal,
            'dC_bilirubin': dC_bil_blood,
            'dC_ammonia': dC_amm_blood,
            'dC_albumin': dC_alb_blood,
            'dC_virus': dC_virus,          # для возможного мониторинга
            'damage': damage,
            'functional': functional,
            'Q_ha': Q_ha,
            'Q_pv': Q_pv
        }

        # Возвращаем производные состояния печени
        return np.array([
            dP_hv,
            dC_bil,
            dC_amm,
            dC_alb,
            dC_virus,
            d_damage,
            0.0,   # reserve 1
            0.0    # reserve 2
        ])

    def get_outputs(self, state):
        return self._current_outputs.copy()