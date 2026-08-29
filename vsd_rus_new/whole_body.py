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

class WindkesselVessel(OrganModel):
    """Двухэлементная модель Windkessel для сосудистого компартмента."""
    def __init__(self, C, P0):
        self.C = C
        self.P0 = P0
        self._current_outputs = {}
    def get_state_size(self): return 1
    def get_initial_state(self): return np.array([self.P0])
    def get_derivatives(self, t, state, inputs):
        P = state[0]
        Q_in = inputs.get('Q_in', 0.0)
        Q_out = inputs.get('Q_out', 0.0)
        dP = (Q_in - Q_out) / self.C
        self._current_outputs = {'P': P}
        return np.array([dP])
    def get_outputs(self, state): return self._current_outputs.copy()

class WholeBodyModel:
    """
    Полная модель организма здорового человека или с ДМЖП.
    Параметр vsd_resistance задаёт сопротивление дефекта (Ом).
    """
    def __init__(self,
                 heart_params=None,
                 lungs_params=None,
                 liver_params=None,
                 kidney_params=None,
                 blood_params=None,
                 gitract_params=None,
                 brain_params=None,
                 vsd_resistance=np.inf,            # сопротивление ДМЖП
                 flow_dependent_lungs=False,       # учитывать рост сопротивления лёгких при перегрузке
                 R_sys_peripheral=1.0,
                 C_sys_art=2.0, C_sys_ven=10.0, C_pul_ven=5.0,
                 P_sa0=80.0, P_sv0=5.0, P_pv0=8.0,
                 substance_names=None):
        if substance_names is None:
            substance_names = ['tox', 'bilirubin', 'ammonia', 'albumin']
        self.substance_names = substance_names

        blood_init = {'V0': 5000.0}
        if blood_params:
            blood_init.update(blood_params)
        initial_concentrations = blood_init.get('initial_concentrations', {})
        for name in substance_names:
            if name not in initial_concentrations:
                initial_concentrations[name] = 0.0

        # Передаём сопротивление ДМЖП в сердце
        heart_params = heart_params or {}
        heart_params['R_vsd'] = vsd_resistance
        self.heart = Heart4Chambers(**heart_params)

        lungs_params = lungs_params or {}
        if flow_dependent_lungs:
            lungs_params['flow_dependent_resistance'] = True
        self.lungs = Lungs2Chamber(**lungs_params)

        self.liver = Liver(**(liver_params or {}))
        self.kidney = KidneyHemodynamic(**(kidney_params or {}))
        self.blood = BloodPool(substance_names=substance_names,
                               V0=blood_init['V0'],
                               initial_concentrations=initial_concentrations)
        self.gitract = GITract(**(gitract_params or {}))
        self.brain = Brain(**(brain_params or {}))

        self.sys_art = WindkesselVessel(C=C_sys_art, P0=P_sa0)
        self.sys_ven = WindkesselVessel(C=C_sys_ven, P0=P_sv0)
        self.pul_ven = WindkesselVessel(C=C_pul_ven, P0=P_pv0)
        self.R_sys_peripheral = R_sys_peripheral

        self.organ_list = [
            self.heart,   # 4
            self.lungs,   # 2
            self.liver,   # 5
            self.blood,   # 1+4 =5
            self.gitract, # 2
            self.brain,   # 1
            self.sys_art, # 1
            self.sys_ven, # 1
            self.pul_ven  # 1
        ]
        self.state_slices = []
        start = 0
        for org in self.organ_list:
            size = org.get_state_size()
            self.state_slices.append(slice(start, start+size))
            start += size
        self.total_states = start

    def get_initial_state(self):
        y0 = []
        for org in self.organ_list:
            y0.extend(org.get_initial_state())
        return np.array(y0)

    def derivatives(self, t, y):
        s = self.state_slices
        V_heart = y[s[0]]
        V_lungs = y[s[1]]
        V_liver = y[s[2]]
        V_blood = y[s[3]]
        V_gitract = y[s[4]]
        V_brain = y[s[5]]
        P_sa = y[s[6]][0]
        P_sv = y[s[7]][0]
        P_pv = y[s[8]][0]

        Vb = V_blood[0]
        C_blood = V_blood[1:]
        conc = dict(zip(self.substance_names, C_blood))
        C_tox = conc.get('tox', 0.0)
        C_bil = conc.get('bilirubin', 0.0)
        C_amm = conc.get('ammonia', 0.0)
        C_alb = conc.get('albumin', 0.0)

        P_pa = V_lungs[0]

        # Сердце
        heart_inputs = {'P_sa': P_sa, 'P_sv': P_sv, 'P_pa': P_pa, 'P_pv': P_pv}
        d_heart = self.heart.get_derivatives(t, V_heart, heart_inputs)
        heart_out = self.heart.get_outputs(V_heart)

        # Лёгкие
        lungs_inputs = {'Q_pulmonary': heart_out['Q_pulmonary'], 'P_pv': P_pv}
        d_lungs = self.lungs.get_derivatives(t, V_lungs, lungs_inputs)
        lungs_out = self.lungs.get_outputs(V_lungs)

        # Печень
        liver_inputs = {'P_sa': P_sa, 'P_sv': P_sv,
                        'C_bilirubin_blood': C_bil, 'C_ammonia_blood': C_amm,
                        'C_albumin_blood': C_alb, 'V_blood': Vb}
        d_liver = self.liver.get_derivatives(t, V_liver, liver_inputs)
        liver_out = self.liver.get_outputs(V_liver)

        # Почки
        kidney_effects = self.kidney.compute_effects(P_sa, P_sv, C_tox, Vb)
        dV_kidney = kidney_effects['dV_blood']
        dC_kidney = kidney_effects['dC_tox']

        # ЖКТ
        gitract_inputs = {'P_sa': P_sa, 'P_sv': P_sv,
                          'P_portal': liver_out.get('P_portal', P_sv),
                          'intake_water': 0.0, 'intake_nutrients': 0.0}
        d_gitract = self.gitract.get_derivatives(t, V_gitract, gitract_inputs)
        gitract_out = self.gitract.get_outputs(V_gitract)

        # Мозг
        brain_inputs = {'P_sa': P_sa, 'P_sv': P_sv, 'C_ammonia': C_amm}
        d_brain = self.brain.get_derivatives(t, V_brain, brain_inputs)
        brain_out = self.brain.get_outputs(V_brain)

        # Кровь
        dC_blood_arr = np.zeros(len(self.substance_names))
        idx = {name:i for i,name in enumerate(self.substance_names)}
        if 'bilirubin' in idx: dC_blood_arr[idx['bilirubin']] += liver_out.get('dC_bilirubin',0)
        if 'ammonia' in idx:   dC_blood_arr[idx['ammonia']]   += liver_out.get('dC_ammonia',0)
        if 'albumin' in idx:   dC_blood_arr[idx['albumin']]   += liver_out.get('dC_albumin',0)
        if 'tox' in idx:       dC_blood_arr[idx['tox']]       += dC_kidney
        # Исправлено: убран двойной учёт воды из ЖКТ (absorption_water уже входит через венозный возврат)
        dV_total = dV_kidney   # только диурез
        blood_inputs = {'dV': dV_total, 'dC': dC_blood_arr}
        d_blood = self.blood.get_derivatives(t, V_blood, blood_inputs)

        # Системные артерии
        Q_peripheral = (P_sa - P_sv) / self.R_sys_peripheral
        Q_ha = liver_out.get('Q_ha',0)
        Q_gitract_in = (P_sa - V_gitract[0]) / self.gitract.R_art
        Q_renal = kidney_effects['Q_renal']
        Q_brain = brain_out['Q_br']
        Q_art_out = Q_peripheral + Q_ha + Q_renal + Q_gitract_in + Q_brain
        d_sys_art = self.sys_art.get_derivatives(t, np.array([P_sa]), {'Q_in': heart_out['Q_aortic'], 'Q_out': Q_art_out})

        # Системные вены
        Q_ven_in = Q_peripheral + liver_out['Q_liver_out'] + Q_renal + gitract_out['Q_out'] + Q_brain
        Q_ven_out = heart_out['Q_sv_to_ra']
        d_sys_ven = self.sys_ven.get_derivatives(t, np.array([P_sv]), {'Q_in': Q_ven_in, 'Q_out': Q_ven_out})

        # Лёгочные вены – исправлено: используем эффективное сопротивление из lungs_out
        Q_from_lungs = (V_lungs[1] - P_pv) / lungs_out['R2_eff']
        Q_pul_ven_out = heart_out['Q_pv_to_la']
        d_pul_ven = self.pul_ven.get_derivatives(t, np.array([P_pv]), {'Q_in': Q_from_lungs, 'Q_out': Q_pul_ven_out})

        dydt = np.concatenate([d_heart, d_lungs, d_liver, d_blood, d_gitract, d_brain, d_sys_art, d_sys_ven, d_pul_ven])
        return dydt

    def compute_outputs(self, t, y):
        s = self.state_slices
        V_heart = y[s[0]]
        V_lungs = y[s[1]]
        V_liver = y[s[2]]
        V_blood = y[s[3]]
        V_gitract = y[s[4]]
        V_brain = y[s[5]]
        P_sa = y[s[6]][0]
        P_sv = y[s[7]][0]
        P_pv = y[s[8]][0]
        P_pa = V_lungs[0]

        Vb = V_blood[0]
        C_blood = V_blood[1:]
        conc = dict(zip(self.substance_names, C_blood))

        heart_inputs = {'P_sa': P_sa, 'P_sv': P_sv, 'P_pa': P_pa, 'P_pv': P_pv}
        self.heart.get_derivatives(t, V_heart, heart_inputs)
        heart_out = self.heart.get_outputs(V_heart)

        lungs_inputs = {'Q_pulmonary': heart_out['Q_pulmonary'], 'P_pv': P_pv}
        self.lungs.get_derivatives(t, V_lungs, lungs_inputs)
        lungs_out = self.lungs.get_outputs(V_lungs)

        liver_inputs = {'P_sa': P_sa, 'P_sv': P_sv,
                        'C_bilirubin_blood': conc.get('bilirubin',0),
                        'C_ammonia_blood': conc.get('ammonia',0),
                        'C_albumin_blood': conc.get('albumin',0),
                        'V_blood': Vb}
        self.liver.get_derivatives(t, V_liver, liver_inputs)
        liver_out = self.liver.get_outputs(V_liver)

        kidney_effects = self.kidney.compute_effects(P_sa, P_sv, conc.get('tox',0), Vb)

        gitract_inputs = {'P_sa': P_sa, 'P_sv': P_sv,
                          'P_portal': liver_out.get('P_portal', P_sv),
                          'intake_water':0,'intake_nutrients':0}
        self.gitract.get_derivatives(t, V_gitract, gitract_inputs)
        gitract_out = self.gitract.get_outputs(V_gitract)

        brain_inputs = {'P_sa': P_sa, 'P_sv': P_sv, 'C_ammonia': conc.get('ammonia',0)}
        self.brain.get_derivatives(t, V_brain, brain_inputs)
        brain_out = self.brain.get_outputs(V_brain)

        reabs_frac = self.kidney.volume_reabsorption_frac
        GFR = -kidney_effects['dV_blood'] / (1 - reabs_frac) if reabs_frac < 1.0 else 0.0

        # Соотношение лёгочного и системного кровотока (Qp/Qs)
        Qp = heart_out['Q_pulmonary']
        Qs = heart_out['Q_aortic']
        Qp_Qs = Qp / Qs if Qs > 0 else np.nan

        outputs = {
            'P_sa': P_sa, 'P_sv': P_sv, 'P_pa': P_pa, 'P_pv': P_pv,
            'V_la': V_heart[0], 'V_lv': V_heart[1], 'V_ra': V_heart[2], 'V_rv': V_heart[3],
            'Q_aortic': Qs, 'Q_pulmonary': Qp,
            'Q_vsd': heart_out['Q_vsd'],
            'Qp_Qs': Qp_Qs,
            'Q_liver_out': liver_out['Q_liver_out'],
            'Q_renal': kidney_effects['Q_renal'],
            'Q_gitract_out': gitract_out['Q_out'],
            'absorption_water': gitract_out['absorption_water'],
            'Q_brain': brain_out['Q_br'],
            'O2_consumption': brain_out['O2_consumption'],
            'glucose_consumption': brain_out['glucose_consumption'],
            'V_blood': Vb,
            'GFR': GFR,
            'C_bilirubin_blood': conc.get('bilirubin',0),
            'C_ammonia_blood': conc.get('ammonia',0),
            'C_albumin_blood': conc.get('albumin',0),
            'C_tox_blood': conc.get('tox',0),
            'oxygenation_index': lungs_out.get('oxygenation_index',1.0),
            'metabolic_inhibition': brain_out.get('metabolic_inhibition',1.0),
            'liver_functional': liver_out.get('functional',1.0),
            'R1_lungs': lungs_out.get('R1_eff', self.lungs.R1_base),
            'R2_lungs': lungs_out.get('R2_eff', self.lungs.R2_base)
        }
        return outputs

    def simulate(self, t_span, t_eval=None, method='RK45', **kwargs):
        y0 = self.get_initial_state()
        return solve_ivp(self.derivatives, t_span, y0, t_eval=t_eval, method=method, **kwargs)