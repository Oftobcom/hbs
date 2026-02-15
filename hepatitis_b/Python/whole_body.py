# whole_body.py

import numpy as np
from scipy.integrate import solve_ivp

from organ_base import OrganModel
from heart import Heart4Chambers
from lungs import Lungs2Chamber
from liver import Liver
from kidney import KidneyHemodynamic
from blood import BloodPool
from gitract import GITract
from brain import Brain


# ----------------------------------------------------------------------
# Вспомогательный класс для сосудистого компартмента (Windkessel)
# ----------------------------------------------------------------------
class WindkesselVessel(OrganModel):
    """
    Двухэлементная модель Windkessel для одного сосудистого компартмента.
    Состояние: [P] — давление (мм рт. ст.)
    Входы: Q_in, Q_out (мл/с) — приток и отток
    Выходы: P
    """
    def __init__(self, C, P0):
        self.C = C          # податливость (мл/мм рт. ст.)
        self.P0 = P0        # начальное давление
        self._current_outputs = {}

    def get_state_size(self):
        return 1

    def get_initial_state(self):
        return np.array([self.P0])

    def get_derivatives(self, t, state, inputs):
        P = state[0]
        Q_in = inputs.get('Q_in', 0.0)
        Q_out = inputs.get('Q_out', 0.0)
        dP = (Q_in - Q_out) / self.C
        self._current_outputs = {'P': P}
        return np.array([dP])

    def get_outputs(self, state):
        return self._current_outputs.copy()


# ----------------------------------------------------------------------
# Главный класс, объединяющий все органы
# ----------------------------------------------------------------------
class WholeBodyModel:
    """
    Полная модель организма, включающая сердце, лёгкие, печень, почки,
    кровь, желудочно-кишечный тракт, головной мозг и три сосудистых
    компартмента: системные артерии (P_sa), системные вены (P_sv),
    лёгочные вены (P_pv).

    Переменные состояния (в порядке следования):
        0–3 : объёмы камер сердца (V_la, V_lv, V_ra, V_rv)
        4–5 : давления в лёгких (P_prox, P_dist)
        6–13: состояние печени (8 переменных: P_hv, C_bilirubin, C_ammonia, C_albumin, C_virus, damage, reserve1, reserve2)
        14–(14+N-1): состояние крови (1 объём + N концентраций)
        далее: состояния ЖКТ (P_art, P_cap), мозга (P_br), системных артерий, системных вен, лёгочных вен.

    Всего состояний: 4 + 2 + 8 + (1+N) + 2 + 1 + 1 + 1 + 1 = 21 + N,
    где N — число веществ в крови (задаётся через blood_params).
    По умолчанию N = 5 (tox, bilirubin, ammonia, albumin, virus).
    """

    def __init__(self,
                 heart_params=None,
                 lungs_params=None,
                 liver_params=None,
                 kidney_params=None,
                 blood_params=None,
                 gitract_params=None,
                 brain_params=None,
                 vsd_resistance=np.inf,
                 R_sys_peripheral=1.0,       # общее периферическое сопротивление (мм рт. ст.·с/мл)
                 C_sys_art=2.0,               # податливость системных артерий (мл/мм рт. ст.)
                 C_sys_ven=10.0,              # податливость системных вен
                 C_pul_ven=5.0,                # податливость лёгочных вен
                 P_sa0=80.0, P_sv0=5.0, P_pv0=8.0,   # начальные давления
                 substance_names=None):        # список веществ в крови

        # Список веществ по умолчанию для гепатита B
        if substance_names is None:
            substance_names = ['tox', 'bilirubin', 'ammonia', 'albumin', 'virus']
        self.substance_names = substance_names

        # Параметры для крови (объём и начальные концентрации)
        blood_init = {'V0': 5000.0}
        if blood_params:
            blood_init.update(blood_params)
        # Начальные концентрации можно задать через blood_params['initial_concentrations']
        initial_concentrations = blood_init.get('initial_concentrations', {})
        # Убедимся, что для всех веществ есть значение (по умолчанию 0)
        for name in substance_names:
            if name not in initial_concentrations:
                initial_concentrations[name] = 0.0
        # Для вируса можно задать ненулевое начальное значение, чтобы имитировать инфекцию
        # Например, в параметрах liver_params можно передать C_virus0, но для крови тоже нужно
        # Если хотим начать с инфицированного состояния, зададим здесь.
        # По умолчанию все нули.

        # Инициализация органов
        self.heart = Heart4Chambers(**(heart_params or {}), R_vsd=vsd_resistance)
        self.lungs = Lungs2Chamber(**(lungs_params or {}))
        self.liver = Liver(**(liver_params or {}))
        self.kidney = KidneyHemodynamic(**(kidney_params or {}))
        self.blood = BloodPool(substance_names=substance_names,
                                V0=blood_init['V0'],
                                initial_concentrations=initial_concentrations)
        self.gitract = GITract(**(gitract_params or {}))
        self.brain = Brain(**(brain_params or {}))

        # Сосудистые компартменты
        self.sys_art = WindkesselVessel(C=C_sys_art, P0=P_sa0)
        self.sys_ven = WindkesselVessel(C=C_sys_ven, P0=P_sv0)
        self.pul_ven = WindkesselVessel(C=C_pul_ven, P0=P_pv0)

        # Периферическое сопротивление (используется в derivatives)
        self.R_sys_peripheral = R_sys_peripheral

        # Список органов в порядке, определяющем глобальный вектор состояния
        self.organ_list = [
            self.heart,      # 4
            self.lungs,      # 2
            self.liver,      # 8
            self.blood,      # 1 + N
            self.gitract,    # 2
            self.brain,      # 1
            self.sys_art,    # 1
            self.sys_ven,    # 1
            self.pul_ven     # 1
        ]

        # Построение отображения индексов (слайсов) для каждого органа
        self.state_slices = []
        start = 0
        for org in self.organ_list:
            size = org.get_state_size()
            self.state_slices.append(slice(start, start + size))
            start += size
        self.total_states = start

    def get_initial_state(self):
        """Возвращает начальный вектор состояния как конкатенацию начальных состояний всех органов."""
        y0 = []
        for org in self.organ_list:
            y0.extend(org.get_initial_state())
        return np.array(y0)

    def derivatives(self, t, y):
        """
        Вычисляет производные состояния в момент времени t.
        """
        # Извлекаем состояния по слайсам
        V_heart = y[self.state_slices[0]]   # 4
        V_lungs = y[self.state_slices[1]]   # 2
        V_liver = y[self.state_slices[2]]   # 8
        V_blood = y[self.state_slices[3]]   # 1+N
        V_gitract = y[self.state_slices[4]] # 2
        V_brain = y[self.state_slices[5]]   # 1
        P_sa = y[self.state_slices[6]][0]   # системное артериальное давление
        P_sv = y[self.state_slices[7]][0]   # системное венозное давление
        P_pv = y[self.state_slices[8]][0]   # лёгочное венозное давление

        # Распаковка состояния печени
        P_hv = V_liver[0]
        C_bilirubin_liver = V_liver[1]
        C_ammonia_liver = V_liver[2]
        C_albumin_liver = V_liver[3]
        C_virus_liver = V_liver[4]
        damage_liver = V_liver[5]

        # Распаковка состояния крови
        Vb = V_blood[0]                # объём крови
        # Концентрации веществ в крови (индекс 1..N)
        C_blood = V_blood[1:]           # массив длиной N
        # Сопоставим с именами для удобства
        blood_concentrations = dict(zip(self.substance_names, C_blood))
        C_tox = blood_concentrations.get('tox', 0.0)
        C_bilirubin = blood_concentrations.get('bilirubin', 0.0)
        C_ammonia = blood_concentrations.get('ammonia', 0.0)
        C_albumin = blood_concentrations.get('albumin', 0.0)
        C_virus = blood_concentrations.get('virus', 0.0)

        P_pa = V_lungs[0]                # давление в лёгочной артерии

        # --- Сердце ---
        heart_inputs = {
            'P_sa': P_sa,
            'P_sv': P_sv,
            'P_pa': P_pa,
            'P_pv': P_pv,
            # можно добавить факторы, если нужно
            'hr_factor': 1.0,
            'inotropy_factor': 1.0
        }
        d_heart = self.heart.get_derivatives(t, V_heart, heart_inputs)
        heart_out = self.heart.get_outputs(V_heart)

        # --- Лёгкие ---
        lungs_inputs = {
            'Q_pulmonary': heart_out['Q_pulmonary'],
            'P_pv': P_pv,
            'liver_damage': damage_liver      # для портопульмональных эффектов
        }
        d_lungs = self.lungs.get_derivatives(t, V_lungs, lungs_inputs)
        lungs_out = self.lungs.get_outputs(V_lungs)

        # --- Печень ---
        liver_inputs = {
            'P_sa': P_sa,
            'P_sv': P_sv,
            'C_bilirubin_blood': C_bilirubin,
            'C_ammonia_blood': C_ammonia,
            'C_albumin_blood': C_albumin,
            'V_blood': Vb
        }
        d_liver = self.liver.get_derivatives(t, V_liver, liver_inputs)
        liver_out = self.liver.get_outputs(V_liver)

        # --- Почки (нет собственного состояния) ---
        kidney_effects = self.kidney.compute_effects(
            P_sa, P_sv, C_tox, Vb,
            C_bilirubin=C_bilirubin,
            C_ammonia=C_ammonia
        )
        Q_renal = kidney_effects['Q_renal']
        dV_kidney = kidney_effects['dV_blood']
        dC_kidney = kidney_effects['dC_tox']   # это изменение tox, но почки также могут влиять на другие вещества?
        # В текущей модели почки влияют только на tox, но можно расширить.
        # Для других веществ пока вклад почек нулевой.

        # --- ЖКТ ---
        gitract_inputs = {
            'P_sa': P_sa,
            'P_sv': P_sv,
            'P_portal': liver_out.get('P_portal', P_sv),   # портальное давление из печени
            'intake_water': 0.0,          # можно задать через параметры
            'intake_nutrients': 0.0
        }
        d_gitract = self.gitract.get_derivatives(t, V_gitract, gitract_inputs)
        gitract_out = self.gitract.get_outputs(V_gitract)

        # --- Мозг ---
        brain_inputs = {
            'P_sa': P_sa,
            'P_sv': P_sv,
            'C_ammonia': C_ammonia
        }
        d_brain = self.brain.get_derivatives(t, V_brain, brain_inputs)
        brain_out = self.brain.get_outputs(V_brain)

        # --- Кровь ---
        # Собираем все вклады в изменение концентраций (dC) от разных органов
        # Создаём массив dC такой же длины, как число веществ, изначально нулевой
        dC_blood = np.zeros(len(self.substance_names))

        # Вклад печени (возвращает dC для каждого вещества)
        # Из liver_out ожидаем ключи: 'dC_bilirubin', 'dC_ammonia', 'dC_albumin'
        # Индексы для веществ:
        idx_map = {name: i for i, name in enumerate(self.substance_names)}
        if 'bilirubin' in idx_map:
            dC_blood[idx_map['bilirubin']] += liver_out.get('dC_bilirubin', 0.0)
        if 'ammonia' in idx_map:
            dC_blood[idx_map['ammonia']] += liver_out.get('dC_ammonia', 0.0)
        if 'albumin' in idx_map:
            dC_blood[idx_map['albumin']] += liver_out.get('dC_albumin', 0.0)
        if 'tox' in idx_map:
            dC_blood[idx_map['tox']] += dC_kidney   # почки влияют на tox
        # Вирус пока не меняется (можно добавить позже)

        # Изменение объёма крови
        dV_total = dV_kidney + gitract_out.get('absorption_water', 0.0)

        blood_inputs = {
            'dV': dV_total,
            'dC': dC_blood
        }
        d_blood = self.blood.get_derivatives(t, V_blood, blood_inputs)

        # --- Системные артерии ---
        # Поток в остальные периферические ткани (не учтённые отдельно)
        Q_peripheral = (P_sa - P_sv) / self.R_sys_peripheral
        # Потоки в органы
        Q_ha = liver_out.get('Q_ha', 0.0)   # печёночная артерия (если есть в выходах)
        Q_gitract_in = (P_sa - V_gitract[0]) / self.gitract.R_art
        Q_brain = brain_out['Q_br']

        # Суммарный отток из артерий
        Q_art_out = Q_peripheral + Q_ha + Q_renal + Q_gitract_in + Q_brain

        sys_art_inputs = {
            'Q_in': heart_out['Q_aortic'],
            'Q_out': Q_art_out
        }
        d_sys_art = self.sys_art.get_derivatives(t, np.array([P_sa]), sys_art_inputs)

        # --- Системные вены ---
        Q_ven_in = Q_peripheral + liver_out['Q_liver_out'] + Q_renal + gitract_out['Q_out'] + Q_brain
        Q_ven_out = heart_out['Q_sv_to_ra']

        sys_ven_inputs = {
            'Q_in': Q_ven_in,
            'Q_out': Q_ven_out
        }
        d_sys_ven = self.sys_ven.get_derivatives(t, np.array([P_sv]), sys_ven_inputs)

        # --- Лёгочные вены ---
        Q_from_lungs = (V_lungs[1] - P_pv) / self.lungs.R2   # P_dist -> P_pv
        Q_pul_ven_out = heart_out['Q_pv_to_la']

        pul_ven_inputs = {
            'Q_in': Q_from_lungs,
            'Q_out': Q_pul_ven_out
        }
        d_pul_ven = self.pul_ven.get_derivatives(t, np.array([P_pv]), pul_ven_inputs)

        # Собираем все производные
        dydt = np.concatenate([
            d_heart,
            d_lungs,
            d_liver,
            d_blood,
            d_gitract,
            d_brain,
            d_sys_art,
            d_sys_ven,
            d_pul_ven
        ])
        return dydt

    def compute_outputs(self, t, y):
        """
        Вычисляет физиологические показатели по вектору состояния y в момент времени t.
        """
        slices = self.state_slices
        V_heart = y[slices[0]]
        V_lungs = y[slices[1]]
        V_liver = y[slices[2]]
        V_blood = y[slices[3]]
        V_gitract = y[slices[4]]
        V_brain = y[slices[5]]
        P_sa = y[slices[6]][0]
        P_sv = y[slices[7]][0]
        P_pv = y[slices[8]][0]
        P_pa = V_lungs[0]

        # Распаковка печени
        P_hv = V_liver[0]
        C_bilirubin_liver = V_liver[1]
        C_ammonia_liver = V_liver[2]
        C_albumin_liver = V_liver[3]
        C_virus_liver = V_liver[4]
        damage_liver = V_liver[5]

        # Распаковка крови
        Vb = V_blood[0]
        C_blood = V_blood[1:]
        blood_dict = dict(zip(self.substance_names, C_blood))

        # --- Сердце ---
        heart_inputs = {
            'P_sa': P_sa,
            'P_sv': P_sv,
            'P_pa': P_pa,
            'P_pv': P_pv
        }
        self.heart.get_derivatives(t, V_heart, heart_inputs)
        heart_out = self.heart.get_outputs(V_heart)

        # --- Лёгкие ---
        lungs_inputs = {
            'Q_pulmonary': heart_out['Q_pulmonary'],
            'P_pv': P_pv,
            'liver_damage': damage_liver
        }
        self.lungs.get_derivatives(t, V_lungs, lungs_inputs)
        lungs_out = self.lungs.get_outputs(V_lungs)

        # --- Печень ---
        liver_inputs = {
            'P_sa': P_sa,
            'P_sv': P_sv,
            'C_bilirubin_blood': blood_dict.get('bilirubin', 0.0),
            'C_ammonia_blood': blood_dict.get('ammonia', 0.0),
            'C_albumin_blood': blood_dict.get('albumin', 0.0),
            'V_blood': Vb
        }
        self.liver.get_derivatives(t, V_liver, liver_inputs)
        liver_out = self.liver.get_outputs(V_liver)

        # --- Почки ---
        kidney_effects = self.kidney.compute_effects(
            P_sa, P_sv,
            blood_dict.get('tox', 0.0),
            Vb,
            C_bilirubin=blood_dict.get('bilirubin', 0.0),
            C_ammonia=blood_dict.get('ammonia', 0.0)
        )

        # --- ЖКТ ---
        gitract_inputs = {
            'P_sa': P_sa,
            'P_sv': P_sv,
            'P_portal': liver_out.get('P_portal', P_sv),
            'intake_water': 0.0,
            'intake_nutrients': 0.0
        }
        self.gitract.get_derivatives(t, V_gitract, gitract_inputs)
        gitract_out = self.gitract.get_outputs(V_gitract)

        # --- Мозг ---
        brain_inputs = {
            'P_sa': P_sa,
            'P_sv': P_sv,
            'C_ammonia': blood_dict.get('ammonia', 0.0)
        }
        self.brain.get_derivatives(t, V_brain, brain_inputs)
        brain_out = self.brain.get_outputs(V_brain)

        # GFR
        reabs_frac = self.kidney.volume_reabsorption_frac
        GFR = -kidney_effects['dV_blood'] / (1 - reabs_frac) if reabs_frac < 1.0 else 0.0

        # Формируем выходной словарь
        outputs = {
            'P_sa': P_sa,
            'P_sv': P_sv,
            'P_pa': P_pa,
            'P_pv': P_pv,
            'V_la': V_heart[0],
            'V_lv': V_heart[1],
            'V_ra': V_heart[2],
            'V_rv': V_heart[3],
            'Q_aortic': heart_out['Q_aortic'],
            'Q_pulmonary': heart_out['Q_pulmonary'],
            'Q_vsd': heart_out['Q_vsd'],
            'Q_liver_out': liver_out['Q_liver_out'],
            'Q_renal': kidney_effects['Q_renal'],
            'Q_gitract_out': gitract_out['Q_out'],
            'absorption_water': gitract_out['absorption_water'],
            'Q_brain': brain_out['Q_br'],
            'O2_consumption': brain_out['O2_consumption'],
            'glucose_consumption': brain_out['glucose_consumption'],
            'V_blood': Vb,
            'GFR': GFR,
            'liver_damage': damage_liver,
            'C_virus_blood': blood_dict.get('virus', 0.0),
            'C_bilirubin_blood': blood_dict.get('bilirubin', 0.0),
            'C_ammonia_blood': blood_dict.get('ammonia', 0.0),
            'C_albumin_blood': blood_dict.get('albumin', 0.0),
            'C_tox_blood': blood_dict.get('tox', 0.0),
            'oxygenation_index': lungs_out.get('oxygenation_index', 1.0),
            'shunt_fraction': lungs_out.get('shunt_fraction', 0.0),
            'metabolic_inhibition': brain_out.get('metabolic_inhibition', 1.0),
        }
        # Добавляем внутрипечёночные показатели по желанию
        outputs['C_virus_liver'] = C_virus_liver
        outputs['C_bilirubin_liver'] = C_bilirubin_liver
        outputs['C_ammonia_liver'] = C_ammonia_liver
        outputs['C_albumin_liver'] = C_albumin_liver
        outputs['liver_functional'] = liver_out.get('functional', 1.0)

        return outputs

    def simulate(self, t_span, t_eval=None, method='RK45', **kwargs):
        """
        Запускает численное интегрирование системы.
        """
        y0 = self.get_initial_state()
        sol = solve_ivp(
            self.derivatives,
            t_span,
            y0,
            t_eval=t_eval,
            method=method,
            **kwargs
        )
        return sol