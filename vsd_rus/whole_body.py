# hbs
# HBS – Human Body Simulation is a modular Python framework
# for multi-organ physiological modeling.
# whole_body.py
from pprint import pp
import warnings

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
from baroreflex import Baroreflex
from gas_exchange import GasExchange
from peripheral_tissues import PeripheralTissues

class WindkesselVessel(OrganModel):
    """
    Двухэлементная модель Windkessel.

    Два режима работы (выбирается через mode):
      mode='P'  — состояние это давление P, dP/dt = (Qin − Qout)/C.
                  Используется для артерий и лёгочных вен.
      mode='V'  — состояние это объём V, dV/dt = Qin − Qout.
                  Давление выводится: P = P0 + (V − V0)/C.
                  Используется для системных вен — «буфера» крови.
    """
    def __init__(self, C, P0, mode='P', V0=None,
                 target_fraction=None, tau_target=200.0):
        self.C = C
        self.P0 = P0
        self.mode = mode
        # V0 = объём при P = P0. Для mode='V' это стартовое V.
        # Если не задан — считаем V0 = C * P0 (согласованность).
        self.V0 = float(V0) if V0 is not None else C * P0

        # --- Масс-баланс (только для mode='V') ---
        # Если target_fraction задан, то V_sv медленно релаксирует
        # к target_fraction · V_blood. Иначе — чистый Windkessel.
        self.target_fraction = float(target_fraction) if target_fraction is not None else None
        self.tau_target = float(tau_target)

        self._current_outputs = {}

    def get_state_size(self):
        return 1

    def get_initial_state(self):
        if self.mode == 'P':
            return np.array([self.P0])
        else:  # mode == 'V'
            return np.array([self.V0])

    def get_derivatives(self, t, state, inputs):
        Q_in = inputs.get('Q_in', 0.0)
        Q_out = inputs.get('Q_out', 0.0)

        if self.mode == 'P':
            P = state[0]
            dP = (Q_in - Q_out) / self.C
            self._current_outputs = {'P': P}
            return np.array([dP])
        else:  # mode == 'V'
            V = state[0]
            dV = Q_in - Q_out

            # --- Масс-баланс: медленная релаксация к целевой доле V_blood ---
            # Если target_fraction задан, добавляем член, который тянет
            # V_sv к target_fraction · V_blood. Релаксация медленная
            # (tau_target), чтобы не ломать быструю гемодинамику.
            if self.target_fraction is not None:
                V_blood = inputs.get('V_blood', None)
                if V_blood is not None:
                    V_target = self.target_fraction * float(V_blood)
                    dV += (V_target - V) / self.tau_target

            # --- Мягкий пол: не позволяем V уйти ниже 50% от V0 ---
            if V < 0.5 * self.V0 and dV < 0:
                softness = (V - 0.5 * self.V0) / (0.5 * self.V0)
                softness = float(np.clip(softness, 0.0, 1.0))
                dV *= softness

            # P выводится из V
            P = self.P0 + (V - self.V0) / self.C
            P = max(P, 0.0)
            self._current_outputs = {'P': P, 'V': V}
            return np.array([dV])
        
    def get_outputs(self, state):
        return self._current_outputs.copy()

class WholeBodyModel:
    """
    Полная модель организма здорового человека или с ДМЖП.
    """
    def __init__(self,
        heart_params=None,
        lungs_params=None,
        liver_params=None,
        kidney_params=None,
        blood_params=None,
        gitract_params=None,
        brain_params=None,
        baroreflex_params=None,
        gas_exchange_params=None,
        flow_dependent_lungs=False,
        R_sys_peripheral=None,
        target_MAP=85.0, target_CO=83.0,
        C_sys_art=1.5, C_pul_ven=15.0,
        P_sa0=85.0, P_sv0=12.0, P_pv0=12.0,
        SYS_VEN_FRACTION=0.5,
        C_sys_ven_eff=400.0,
        tau_target=200.0,
        fluid_intake_rate=0.0,
        insensible_loss_rate=0.0,
        peripheral_params=None,
        substance_names=None):

        # Если substance_names не передан, берём из blood_params или дефолтный список
        if substance_names is None:
            if blood_params and 'initial_concentrations' in blood_params:
                substance_names = list(blood_params['initial_concentrations'].keys())
            else:
                substance_names = ['tox', 'bilirubin', 'ammonia', 'albumin',
                    'glucose', 'oxygen', 'co2', 'lactate',]

        self.substance_names = substance_names
        self._substance_idx = {name: i for i, name in enumerate(substance_names)}

        # Инициализация органов и их параметров
        blood_init = {'V0': 5800.0}
        if blood_params:
            blood_init.update(blood_params)
        initial_concentrations = dict(blood_init.get('initial_concentrations', {}))

        DEFAULT_CONC = {
            'tox': 0.0, 'bilirubin': 0.5, 'ammonia': 0.3,
            'albumin': 4.5, 'glucose': 5.0, 'oxygen': 0.15, 
            'co2': 0.52, 'lactate': 0.10,
        }
        for name in substance_names:
            if name not in initial_concentrations:
                initial_concentrations[name] = DEFAULT_CONC.get(name, 0.0)

        heart_params = dict(heart_params or {})
        if heart_params.get('R_vsd') is None:      # <-- ДОБАВИТЬ
            heart_params['R_vsd'] = np.inf
        self.heart = Heart4Chambers(**heart_params)

        lungs_params = dict(lungs_params or {})
        lungs_params['flow_dependent_resistance'] = bool(flow_dependent_lungs)
        self.lungs = Lungs2Chamber(**lungs_params)
        
        self.liver = Liver(**(liver_params or {}))
        self.kidney = KidneyHemodynamic(**(kidney_params or {}))
        self.blood = BloodPool(substance_names=substance_names,
                               V0=blood_init['V0'],
                               initial_concentrations=initial_concentrations)
        self.gitract = GITract(**(gitract_params or {}))
        self.brain = Brain(**(brain_params or {}))

        # ОЦЕНКА других проводимостей для здорового
        # R = dP / Q_target
        R_renal_est = 3.75
        R_brain_est = 7.0
        R_ha_est = 17.0
        R_gitract_est = 4.5

        if R_sys_peripheral is None:
            R_total_target = target_MAP / target_CO
            sum_cond_other = (1/R_renal_est + 1/R_brain_est + 1/R_ha_est + 1/R_gitract_est)
            cond_per_needed = 1/R_total_target - sum_cond_other
            cond_per_needed = max(cond_per_needed, 0.1)
            R_sys_peripheral = 1.0 / cond_per_needed

        # --- Периферические ткани ---
        # R_base периферии должен совпасть с R_sys_peripheral,
        # который был вычислен выше под target_MAP / target_CO
        pp = dict(peripheral_params or {})
        if pp.get('R_base') is None:
            pp.pop('R_base', None)
        pp.setdefault('R_base', R_sys_peripheral)
        self.peripheral = PeripheralTissues(**pp)
            
        self.target_MAP = target_MAP
        self.target_CO = target_CO

        # Создаём барорефлекс с параметрами по умолчанию или переданными
        baroreflex_params = baroreflex_params or {}
        self.baroreflex = Baroreflex(**baroreflex_params)

        # Создаём сосудистые компартменты Windkessel
        # sys_art и pul_ven работают в P-mode (давление как состояние)
        self.sys_art = WindkesselVessel(C=C_sys_art, P0=P_sa0, mode='P')
        self.pul_ven = WindkesselVessel(C=C_pul_ven, P0=P_pv0, mode='P')

        # sys_ven — БУФЕР объёма. Работает в V-mode.
        V_sv0 = SYS_VEN_FRACTION * blood_init['V0']

        self.sys_ven = WindkesselVessel(
            C=C_sys_ven_eff, P0=P_sv0, mode='V', V0=V_sv0,
            target_fraction=SYS_VEN_FRACTION,
            tau_target=tau_target,
        )

        self.R_sys_peripheral = R_sys_peripheral
        self.fluid_intake_rate = fluid_intake_rate
        self.insensible_loss_rate = insensible_loss_rate

        ge = dict(gas_exchange_params or {})
        if 'VO2_base' not in ge:
            ge['VO2_base'] = 4.2 - self.peripheral.VO2_base
        self.gas_exchange = GasExchange(**ge)

        # Порядок органов определяет структуру вектора состояния
        self.organ_list = [
            self.heart, self.lungs, self.liver, self.blood, self.gitract,
            self.brain, self.peripheral, self.baroreflex,
            self.sys_art, self.sys_ven, self.pul_ven,
        ]
        # Имена органов в том же порядке, что и organ_list
        ORGAN_NAMES = [
            'heart', 'lungs', 'liver', 'blood', 'gitract',
            'brain', 'peripheral', 'baroreflex',
            'sys_art', 'sys_ven', 'pul_ven',
        ]

        assert len(ORGAN_NAMES) == len(self.organ_list), \
            "organ_list и ORGAN_NAMES рассинхронизированы"

        # Строим срезы состояния и именованный индекс в одном цикле
        self.state_slices = []
        self.idx = {}
        start = 0
        for name, org in zip(ORGAN_NAMES, self.organ_list):
            size = org.get_state_size()
            slc = slice(start, start + size)
            self.state_slices.append(slc)
            self.idx[name] = slc
            start += size
        self.total_states = start

    def calibrate_initial_state(self, t_calib=600.0, t_eval=None,
                                rtol=1e-4, atol=1e-5,
                                p_sa_lo=50.0, p_sa_hi=150.0,
                                rel_tol_cycle=0.25):
        """
        Калибровка начального состояния.

        Возвращает либо y_steady (если прогрев сошёлся на предельный цикл),
        либо аналитическое y0 (если что-то пошло не так).
        """
        y0 = self.get_initial_state()
        heart_slc = self.idx['heart']
        V0_arr = np.array([self.heart.V0[c] for c in ('LA','LV','RA','RV')])
        # страховка от вырожденного y0 (обычно no-op)
        y0[heart_slc] = np.maximum(y0[heart_slc], 1.5 * V0_arr)

        # --- Санитизация t_eval ---
        # solve_ivp при t_eval is not None требует строго t_eval ⊂ (t_span[0], t_span[1]).
        # Ошибки округления np.arange могут выкинуть последний элемент за t_calib,
        # поэтому отрезаем всё, что не попало в открытый интервал.
        if t_eval is not None:
            t_eval = np.asarray(t_eval, dtype=float)
            t_eval = t_eval[(t_eval > 0.0) & (t_eval < t_calib)]
            if t_eval.size == 0:
                t_eval = None

        try:
            sol = solve_ivp(self.derivatives, (0.0, t_calib), y0,
                            t_eval=t_eval, method='LSODA', rtol=rtol, atol=atol, max_step=0.07)
        except Exception as e:
            warnings.warn(f"calibrate: solver failed ({e}); using analytic y0")
            return y0

        if sol.y.shape[1] < 2 or not np.all(np.isfinite(sol.y[:, -1])):
            warnings.warn("calibrate: non-finite solution; using analytic y0")
            return y0

        y_steady = sol.y[:, -1].copy()

        # Проверка сходимости: P_sa должна попасть в физиологическое окно
        P_sa_end = y_steady[self.idx['sys_art']][0]
        if not (p_sa_lo < P_sa_end < p_sa_hi):
            warnings.warn(f"calibrate: P_sa={P_sa_end:.1f} вне [{p_sa_lo},{p_sa_hi}]; using analytic y0")
            return y0

        # Проверка цикличности — только если есть траектория
        if sol.t.size >= 3:
            P_sa_traj = sol.y[self.idx['sys_art'].start, :]     # ← исправлено
            HR_end = y_steady[self.idx['baroreflex']][0]
            T = 60.0 / max(HR_end, 1e-6)
            mask = sol.t >= (sol.t[-1] - T)
            if mask.sum() >= 3:
                ps_cycle = P_sa_traj[mask]
                mean_ps = np.mean(ps_cycle)
                if mean_ps > 0 and np.std(ps_cycle) / mean_ps > rel_tol_cycle:
                    warnings.warn(f"calibrate: P_sa вариация за цикл "
                                f"{np.std(ps_cycle)/mean_ps:.2f} > {rel_tol_cycle}; "
                                f"using analytic y0")
                    return y0

        # Опционально: проверить, что состояние на предельном цикле
        # (по вариации P_sa за последний цикл). Если t_eval задан —
        # можно сравнить y_steady с y_steady_cycle_ago.
        # В упрощённом варианте пропускаем.

        return y_steady
    
    def get_initial_state(self, calibrated=False):
        y0 = []
        for org in self.organ_list:
            y0.extend(org.get_initial_state())
        y0 = np.array(y0)
        if calibrated:
            # если хочешь сразу калиброванный
            return self.calibrate_initial_state()
        return y0

    def _compute_organ_flows(self, t, y):
        """
        Единая точка расчёта всех межагентных потоков и промежуточных величин.

        Вызывается из derivatives() и compute_outputs() — гарантирует, что
        оба метода видят одну и ту же физику (нет рассинхрона).

        Побочный эффект: вызывает get_derivatives() у органов, что обновляет
        их внутренний кэш _current_flows / _current_outputs. Поэтому
        последующие get_outputs() возвращают согласованные значения.

        Возвращает dict со всеми состояниями, выходами органов, локальными
        потоками, накопленным dC_blood_arr и dV_total.
        """
        sl = self.idx

        # --- Состояния ---
        V_heart      = y[sl['heart']]
        V_lungs      = y[sl['lungs']]
        V_liver      = y[sl['liver']]
        V_blood      = y[sl['blood']]
        V_gitract    = y[sl['gitract']]
        V_brain      = y[sl['brain']]
        V_periph     = y[sl['peripheral']]
        V_baroreflex = y[sl['baroreflex']]
        P_sa         = y[sl['sys_art']][0]
        V_sv         = y[sl['sys_ven']][0]      # объём системных вен (V-mode)
        P_pv         = y[sl['pul_ven']][0]
        # Извлекаем P_sv из V_sv (линейная зависимость)
        P_sv = self.sys_ven.P0 + (V_sv - self.sys_ven.V0) / self.sys_ven.C
        P_sv = max(P_sv, 0.0)

        Vb           = V_blood[0]
        C_blood      = V_blood[1:]
        conc         = dict(zip(self.substance_names, C_blood))
        P_pa         = V_lungs[0]

        # --- Барорефлекс ---
        baroreflex_inputs = {'P_sa': P_sa}
        d_baroreflex = self.baroreflex.get_derivatives(t, V_baroreflex, baroreflex_inputs)
        baroreflex_out = self.baroreflex.get_outputs(V_baroreflex)
        HR = baroreflex_out['HR']
        hr_factor = HR / self.baroreflex.HR_base

        # --- Сердце ---
        heart_inputs = {'P_sa': P_sa, 'P_sv': P_sv, 'P_pa': P_pa, 'P_pv': P_pv,
                        'hr_factor': hr_factor,
                        'baro_activation': baroreflex_out['baro_activation']}
        d_heart = self.heart.get_derivatives(t, V_heart, heart_inputs)
        heart_out = self.heart.get_outputs(V_heart)

        # --- Лёгкие ---
        lungs_inputs = {'Q_pulmonary': heart_out['Q_pulmonary'], 'P_pv': P_pv}
        d_lungs = self.lungs.get_derivatives(t, V_lungs, lungs_inputs)
        lungs_out = self.lungs.get_outputs(V_lungs)

        # --- ЖКТ ---
        P_portal_state = V_liver[5] if len(V_liver) > 5 else 8.0
        gitract_inputs = {'P_sa': P_sa, 'P_sv': P_sv,
                        'P_portal': P_portal_state,
                        'intake_water': 0.0, 'intake_nutrients': 0.0}
        d_gitract = self.gitract.get_derivatives(t, V_gitract, gitract_inputs)
        gitract_out = self.gitract.get_outputs(V_gitract)

        # --- Печень ---
        liver_inputs = {'P_sa': P_sa, 'P_sv': P_sv,
                        'C_bilirubin_blood': conc.get('bilirubin', 0.0),
                        'C_ammonia_blood':   conc.get('ammonia',   0.0),
                        'C_albumin_blood':   conc.get('albumin',   0.0),
                        'C_lactate_blood':   conc.get('lactate',   0.10),
                        'V_blood': Vb,
                        'Q_gut_out': gitract_out['Q_out']}
        d_liver = self.liver.get_derivatives(t, V_liver, liver_inputs)
        liver_out = self.liver.get_outputs(V_liver)

        # --- Почки ---
        kidney_effects = self.kidney.compute_effects(P_sa, P_sv,
                                                    conc.get('tox', 0.0), Vb)

        # --- Газообмен ---
        gas_ex = self.gas_exchange.compute_effects(
            C_v_O2  = conc.get('oxygen', 0.15),
            C_v_CO2 = conc.get('co2',    0.52),
            Q_p     = heart_out['Q_pulmonary'],
            Q_shunt = heart_out['Q_vsd'],
            V_blood = Vb,
        )

        # --- Мозг ---
        brain_inputs = {'P_sa': P_sa, 'P_sv': P_sv,
                        'C_a_O2': gas_ex['C_a_O2'],
                        'C_ammonia': conc.get('ammonia', 0.0)}
        d_brain = self.brain.get_derivatives(t, V_brain, brain_inputs)
        brain_out = self.brain.get_outputs(V_brain)

        # --- Периферия ---
        peripheral_inputs = {'P_sa': P_sa, 'P_sv': P_sv,
                            'C_a_O2': gas_ex['C_a_O2'],
                            'C_v_lactate': conc.get('lactate', 0.10),
                            'V_blood': Vb}
        d_peripheral = self.peripheral.get_derivatives(t, V_periph, peripheral_inputs)
        periph_out = self.peripheral.get_outputs(V_periph)

        # --- Локальные потоки (единые для derivatives и compute_outputs) ---
        Q_peripheral = periph_out['Q_peripheral']
        Q_ha         = liver_out.get('Q_ha', 0.0)
        Q_renal      = kidney_effects['Q_renal']
        Q_gitract_in = (P_sa - V_gitract[0]) / self.gitract.R_art
        Q_brain      = brain_out['Q_br']
        Q_art_out    = Q_peripheral + Q_ha + Q_renal + Q_gitract_in + Q_brain
        Q_ven_in     = Q_peripheral + liver_out['Q_liver_out'] + Q_renal + Q_brain
        Q_ven_out    = heart_out['Q_sv_to_ra']

        # --- Накопление dC_blood_arr (все вклады в концентрации) ---
        dC_blood_arr = np.zeros(len(self.substance_names))
        idx = self._substance_idx
        if 'bilirubin' in idx: dC_blood_arr[idx['bilirubin']] += liver_out.get('dC_bilirubin', 0.0)
        if 'ammonia'   in idx: dC_blood_arr[idx['ammonia']]   += liver_out.get('dC_ammonia',   0.0)
        if 'albumin'   in idx: dC_blood_arr[idx['albumin']]   += liver_out.get('dC_albumin',   0.0)
        if 'lactate'   in idx: dC_blood_arr[idx['lactate']]   += liver_out.get('dC_lactate',   0.0)
        if 'tox'       in idx: dC_blood_arr[idx['tox']]       += kidney_effects['dC_tox']
        if 'oxygen'    in idx: dC_blood_arr[idx['oxygen']]    += gas_ex['dC_O2']
        if 'co2'       in idx: dC_blood_arr[idx['co2']]       += gas_ex['dC_CO2']
        if 'oxygen'    in idx: dC_blood_arr[idx['oxygen']]    += periph_out['dC_O2_blood']
        if 'lactate'   in idx: dC_blood_arr[idx['lactate']]   += periph_out['dC_lactate_blood']

        # --- Баланс жидкости ---
        dV_total = (gitract_out['absorption_water']
                    + self.fluid_intake_rate
                    - kidney_effects['urine_output']
                    - self.insensible_loss_rate)

        return {
            # Состояния
            'V_heart': V_heart, 'V_lungs': V_lungs, 'V_liver': V_liver,
            'V_blood': V_blood, 'V_gitract': V_gitract, 'V_brain': V_brain,
            'V_periph': V_periph, 'V_baroreflex': V_baroreflex,
            'P_sa': P_sa, 'P_sv': P_sv, 'P_pv': P_pv, 'P_pa': P_pa,
            'V_sv': V_sv, 'Vb': Vb, 'C_blood': C_blood, 'conc': conc,
            'V_sv_target': self.sys_ven.target_fraction * Vb if self.sys_ven.target_fraction else V_sv,
            'V_sv_fraction': V_sv / max(Vb, 1e-6) if Vb > 0 else 0.0,
            # Барорефлекс
            'HR': HR, 'hr_factor': hr_factor, 'baroreflex_out': baroreflex_out,
            'baro_activation': baroreflex_out['baro_activation'],
            # Органные производные
            'd_heart': d_heart, 'd_lungs': d_lungs, 'd_liver': d_liver,
            'd_gitract': d_gitract, 'd_brain': d_brain,
            'd_peripheral': d_peripheral, 'd_baroreflex': d_baroreflex,
            # Органные выходы
            'heart_out': heart_out, 'lungs_out': lungs_out,
            'gitract_out': gitract_out, 'liver_out': liver_out,
            'kidney_effects': kidney_effects, 'brain_out': brain_out,
            'gas_ex': gas_ex, 'periph_out': periph_out,
            # Локальные потоки
            'Q_peripheral': Q_peripheral, 'Q_ha': Q_ha, 'Q_renal': Q_renal,
            'Q_gitract_in': Q_gitract_in, 'Q_brain': Q_brain,
            'Q_art_out': Q_art_out, 'Q_ven_in': Q_ven_in, 'Q_ven_out': Q_ven_out,
            # Кровь
            'dC_blood_arr': dC_blood_arr, 'dV_total': dV_total,
        }

    def derivatives(self, t, y):
        f = self._compute_organ_flows(t, y)

        # --- Кровь ---
        blood_inputs = {'dV': f['dV_total'], 'dC': f['dC_blood_arr']}
        d_blood = self.blood.get_derivatives(t, f['V_blood'], blood_inputs)

        # --- Системные артерии ---
        d_sys_art = self.sys_art.get_derivatives(
            t, np.array([f['P_sa']]),
            {'Q_in': f['heart_out']['Q_aortic'], 'Q_out': f['Q_art_out']},
        )

        # --- Системные вены ---
        # В V-mode состояние = V_sv, а не P_sv.
        # Передаём V_blood, чтобы sys_ven мог тянуть V_sv к target_fraction·V_blood.
        d_sys_ven = self.sys_ven.get_derivatives(
            t, np.array([f['V_sv']]),
            {'Q_in': f['Q_ven_in'], 'Q_out': f['Q_ven_out'], 'V_blood': f['Vb']},
        )       

        # --- Лёгочные вены ---
        Q_from_lungs = (f['V_lungs'][1] - f['P_pv']) / f['lungs_out']['R2_eff']
        Q_pul_ven_out = f['heart_out']['Q_pv_to_la']
        d_pul_ven = self.pul_ven.get_derivatives(
            t, np.array([f['P_pv']]),
            {'Q_in': Q_from_lungs, 'Q_out': Q_pul_ven_out},
        )

        return np.concatenate([
            f['d_heart'], f['d_lungs'], f['d_liver'], d_blood,
            f['d_gitract'], f['d_brain'], f['d_peripheral'], f['d_baroreflex'],
            d_sys_art, d_sys_ven, d_pul_ven,
        ])

    def compute_outputs(self, t, y):
        f = self._compute_organ_flows(t, y)

        heart_out = f['heart_out']
        Qp = heart_out['Q_pulmonary']
        Qs = heart_out['Q_aortic']
        Qp_Qs = Qp / max(Qs, 1e-6)
        shunt_fraction_LR = max(heart_out['Q_vsd'], 0.0) / max(Qp, 1e-6)

        return {
            'P_sa': f['P_sa'], 'P_sv': f['P_sv'], 'P_pa': f['P_pa'], 'P_pv': f['P_pv'],
            'V_la': f['V_heart'][0], 'V_lv': f['V_heart'][1],
            'V_sv': f['V_sv'], 'V_sv_target': f['V_sv_target'],
            'V_sv_fraction': f['V_sv_fraction'],
            'V_ra': f['V_heart'][2], 'V_rv': f['V_heart'][3],
            'Q_aortic': Qs, 'Q_pulmonary': Qp,
            'Q_vsd': heart_out['Q_vsd'],
            'P_lv': heart_out['P_lv'], 'P_rv': heart_out['P_rv'],
            'P_la': heart_out['P_la'], 'P_ra': heart_out['P_ra'],
            # --- Дополнительные потоки для диагностики ---
            'Q_ven_in':   f['Q_ven_in'],                     # суммарный венозный возврат
            'Q_ven_out':  f['Q_ven_out'],                    # отток в ПП
            'Q_sv_to_ra': heart_out['Q_sv_to_ra'],           # системные вены → ПП
            'Q_pv_to_la': heart_out['Q_pv_to_la'],           # лёгочные вены → ЛП
            'Qp_Qs': Qp_Qs,
            'shunt_fraction_LR': shunt_fraction_LR,
            'shunt_fraction_R2L': f['gas_ex']['shunt_fraction_R2L'],
            'Q_liver_out': f['liver_out']['Q_liver_out'],
            'Q_renal': f['kidney_effects']['Q_renal'],
            'urine_output': f['kidney_effects']['urine_output'],
            'Q_gitract_out': f['gitract_out']['Q_out'],
            'absorption_water': f['gitract_out']['absorption_water'],
            'Q_brain': f['brain_out']['Q_br'],
            'O2_consumption': f['brain_out']['O2_consumption'],
            'glucose_consumption': f['brain_out']['glucose_consumption'],
            'V_blood': f['Vb'],
            'GFR': f['kidney_effects']['GFR'],
            'C_bilirubin_blood': f['conc'].get('bilirubin', 0),
            'C_ammonia_blood':   f['conc'].get('ammonia',   0),
            'C_albumin_blood':   f['conc'].get('albumin',   0),
            'C_tox_blood':       f['conc'].get('tox',       0),
            'oxygenation_index': f['gas_ex']['SaO2'],
            'metabolic_inhibition': f['brain_out'].get('metabolic_inhibition', 1.0),
            'liver_functional':     f['liver_out'].get('functional', 1.0),
            'R1_lungs': f['lungs_out'].get('R1_eff', self.lungs.R1_base),
            'R2_lungs': f['lungs_out'].get('R2_eff', self.lungs.R2_base),
            'HR': f['HR'],
            'HR_target': f['baroreflex_out']['HR_target'],
            'SaO2':   f['gas_ex']['SaO2'],
            'C_a_O2': f['gas_ex']['C_a_O2'],
            'C_v_O2': f['gas_ex']['C_v_O2'],
            'P_a_O2': f['gas_ex']['P_a_O2'],
            'P_v_O2': f['gas_ex']['P_v_O2'],
            'P_v_CO2': f['gas_ex']['P_v_CO2'],
            'O2_uptake':   f['gas_ex']['O2_uptake'],
            'CO2_removal': f['gas_ex']['CO2_removal'],
            'Q_peripheral': f['Q_peripheral'],
            'Q_art_out':    f['Q_art_out'],
            'R_sys_peripheral': self.R_sys_peripheral,
            'R_eff_peripheral':      f['periph_out'].get('R_eff', None),
            'C_O2_local_periph':     f['periph_out'].get('C_O2_local', None),
            'C_lactate_local_periph': f['periph_out'].get('C_lactate_local', None),
            'C_lactate_blood':      f['conc'].get('lactate', None),
            'dC_lactate_liver':     f['liver_out'].get('dC_lactate', 0.0),
            'lactate_production':   f['periph_out'].get('lactate_production', 0.0),
            'O2_consumption_periph': f['periph_out'].get('O2_consumption_periph', 0.0),
            'f_O2_autoreg':         f['periph_out'].get('f_O2_autoreg', 1.0),
            'f_P_myogenic':         f['periph_out'].get('f_P_myogenic', 1.0),
            'dC_lactate_periph_to_blood': f['periph_out'].get('dC_lactate_blood', 0.0),
            'Q_mitral':    heart_out['Q_mitral'],
            'Q_tricuspid': heart_out['Q_tricuspid'],
        }

    # def simulate(self, t_span, t_eval=None, y0=None, method='RK45', **kwargs):
    # def simulate(self, t_span, t_eval=None, y0=None, method='BDF', **kwargs):
    def simulate(self, t_span, t_eval=None, y0=None, method='LSODA', **kwargs):
        if y0 is None:
            y0 = self.calibrate_initial_state()
        kwargs.setdefault('max_step', 0.05)
        return solve_ivp(self.derivatives, t_span, y0, 
                         t_eval=t_eval, method=method, **kwargs)
     