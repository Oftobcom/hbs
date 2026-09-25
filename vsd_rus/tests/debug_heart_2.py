#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_heart_2.py — изолированная проверка Heart4Chambers с сохранением в txt + строгий тест.
"""
from __future__ import annotations
import sys
from pathlib import Path
import datetime
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

# --- Совместимость NumPy 1.x / 2.x: np.trapz → np.trapezoid ---
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz

from heart import Heart4Chambers
from physio_config import load_physiology
RESULT_FILE = Path(__file__).resolve().parent / "results_debug_heart_2.txt"
class TeeLogger:
    def __init__(self, file_path: Path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()
tee = TeeLogger(RESULT_FILE)
sys.stdout = tee
sys.stderr = tee
def log(msg=""):
    print(msg)
def load_heart_cfg():
    cfg = load_physiology()
    return dict(cfg['heart']), dict(cfg['systemic'])
HEART_CFG, SYS_CFG = load_heart_cfg()
def make_heart(R_vsd=np.inf):
    return Heart4Chambers(
        hr=HEART_CFG['hr'],
        E_max_la=HEART_CFG['E_max_la'], E_min_la=HEART_CFG['E_min_la'],
        E_max_ra=HEART_CFG['E_max_ra'], E_min_ra=HEART_CFG['E_min_ra'],
        E_max_lv=HEART_CFG['E_max_lv'], E_min_lv=HEART_CFG['E_min_lv'],
        E_max_rv=HEART_CFG['E_max_rv'], E_min_rv=HEART_CFG['E_min_rv'],
        V0_la=HEART_CFG['V0_la'], V0_lv=HEART_CFG['V0_lv'],
        V0_ra=HEART_CFG['V0_ra'], V0_rv=HEART_CFG['V0_rv'],
        EDV_la=HEART_CFG['EDV_la'], EDV_lv=HEART_CFG['EDV_lv'],
        EDV_ra=HEART_CFG['EDV_ra'], EDV_rv=HEART_CFG['EDV_rv'],
        R_mitral=HEART_CFG['R_mitral'], R_aortic=HEART_CFG['R_aortic'],
        R_tricuspid=HEART_CFG['R_tricuspid'], R_pulmonary=HEART_CFG['R_pulmonary'],
        R_venous_sys=HEART_CFG['R_venous_sys'], R_venous_pulm=HEART_CFG['R_venous_pulm'],
        R_vsd=R_vsd,
        hr_min=HEART_CFG['hr_min'], hr_max=HEART_CFG['hr_max'],
        k_valve=HEART_CFG['k_valve'],
    )
def banner():
    log("="*70)
    log(f"Heart4Chambers: {datetime.datetime.now()} | physiology.yaml")
    log("="*70)
    for k,v in HEART_CFG.items():
        log(f"  {k:20s} = {v}")
    log(f"  P_sa0={SYS_CFG['P_sa0']} P_sv0={SYS_CFG['P_sv0']} P_pv0={SYS_CFG['P_pv0']} target_CO={SYS_CFG['target_CO']}")
    log(f"  → {RESULT_FILE}")
    log("="*70)
def simulate_beats(R_vsd=np.inf, P_sa=85, P_sv=12, P_pa=15, P_pv=12, hr=70, t_beats=10, return_full=False):
    from scipy.integrate import solve_ivp
    h = make_heart(R_vsd=R_vsd)
    h.hr_base = hr
    h.T_base = 60/hr
    h._current_T = 60/hr
    y0 = h.get_initial_state()
    inputs = {'P_sa':P_sa,'P_sv':P_sv,'P_pa':P_pa,'P_pv':P_pv,'hr_factor':1.0,'inotropy_factor':1.0,'baro_activation':1.0}
    T = 60/hr
    t_span = (0, t_beats*T)
    def rhs(t,y):
        return h.get_derivatives(t,y,inputs)
    sol = solve_ivp(rhs, t_span, y0, method='LSODA', rtol=1e-4, atol=1e-5, max_step=0.01)
    t = sol.t
    flows = {k: [] for k in ['Q_aortic','Q_pulmonary','Q_mitral','Q_tricuspid','Q_vsd','Q_sv_to_ra','Q_pv_to_la']}
    pressures = {k: [] for k in ['P_la','P_lv','P_ra','P_rv']}
    for ti, yi in zip(t, sol.y.T):
        h.get_derivatives(ti, yi, inputs)
        out = h.get_outputs(yi)
        for k in flows:
            flows[k].append(out.get(k,0.0))
        for k in pressures:
            pressures[k].append(out.get(k,0.0))
    for k in flows:
        flows[k]=np.array(flows[k])
    for k in pressures:
        pressures[k]=np.array(pressures[k])
    idx_last = t >= t[-1]-T
    SV = _trapz(flows['Q_aortic'][idx_last], t[idx_last])
    CO = SV * hr / 60.0
    V_lv = sol.y[1]
    EDV = np.max(V_lv[idx_last])
    ESV = np.min(V_lv[idx_last])
    P_lv_max = np.max(pressures['P_lv'])
    Q_vsd_mean = np.mean(flows['Q_vsd'][idx_last])
    res = {"SV":SV,"CO":CO,"EDV":EDV,"ESV":ESV,"EF":SV/EDV if EDV>0 else 0,
           "P_lv_max":P_lv_max,"Q_vsd_mean":Q_vsd_mean,"nfev":sol.nfev,"sol":sol,
           "flows":flows,"pressures":pressures,"t":t,"V":sol.y}
    return res
def test_interface():
    log("\n"+"="*70)
    log("TEST 1: Интерфейс OrganModel")
    log("="*70)
    h = make_heart()
    sz = h.get_state_size()
    y0 = h.get_initial_state()
    d = h.get_derivatives(0.0, y0, {'P_sa':85,'P_sv':12,'P_pa':15,'P_pv':12})
    log(f"state_size={sz} ожидание 4, y0={y0}")
    ok = sz==4 and len(y0)==4 and len(d)==4
    log(f"[{'OK' if ok else 'FAIL'}] интерфейс")
    return {"ok":ok}
def test_healthy():
    log("\n"+"="*70)
    log("TEST 2: Здоровый HR 70 P_sa 85")
    log("="*70)
    res = simulate_beats(t_beats=10)
    log(f"SV={res['SV']:.1f} [60-110] CO={res['CO']:.1f} [70-130] ({res['CO']*60:.0f} мл/мин)")
    log(f"EDV={res['EDV']:.1f} [100-150] ESV={res['ESV']:.1f} [20-60] EF={res['EF']:.2f} [0.5-0.8]")
    log(f"P_lv_max={res['P_lv_max']:.1f} [100-180] Q_vsd={res['Q_vsd_mean']:.3f} nfev={res['nfev']}")
    ok = (60<=res['SV']<=110) and (70<=res['CO']<=130) and (100<=res['EDV']<=150) and (100<=res['P_lv_max']<=180)
    log(f"[{'OK' if ok else 'FAIL'}] здоровое сердце")
    return {"ok":ok, **res}
def test_frank_starling():
    log("\n"+"="*70)
    log("TEST 3: Frank-Starling P_sv 4 vs 12 vs 20 → SV растет")
    log("="*70)
    vals=[]
    for P_sv in [4,12,20]:
        r = simulate_beats(P_sv=P_sv, t_beats=10)
        vals.append(r['SV'])
        log(f"P_sv {P_sv:2d} → SV {r['SV']:.1f} EDV {r['EDV']:.1f}")
    ok = vals[0]<=vals[1]<=vals[2]
    log(f"[{'OK' if ok else 'FAIL'}] Frank-Starling {vals}")
    return {"ok":ok}
def test_afterload():
    log("\n"+"="*70)
    log("TEST 4: Afterload P_sa 60 vs 85 vs 120 → SV падает")
    log("="*70)
    vals=[]
    for P_sa in [60,85,120]:
        r=simulate_beats(P_sa=P_sa,t_beats=10)
        vals.append(r['SV'])
        log(f"P_sa {P_sa} → SV {r['SV']:.1f}")
    ok = vals[0]>vals[1]>vals[2]
    log(f"[{'OK' if ok else 'FAIL'}] afterload")
    return {"ok":ok}
def test_vsd():
    log("\n"+"="*70)
    log("TEST 5: VSD inf vs 0.5 vs 0.1")
    log("="*70)
    for R in [np.inf,0.5,0.1]:
        r=simulate_beats(R_vsd=R,t_beats=10)
        log(f"R_vsd {R} → Q_vsd {r['Q_vsd_mean']:.2f} SV {r['SV']:.1f}")
    r_inf=simulate_beats(R_vsd=np.inf,t_beats=10)
    r_small=simulate_beats(R_vsd=0.1,t_beats=10)
    ok = abs(r_inf['Q_vsd_mean'])<0.5 and r_small['Q_vsd_mean']>1.0
    log(f"[{'OK' if ok else 'FAIL'}] VSD L→R")
    return {"ok":ok}
def test_hr():
    log("\n"+"="*70)
    log("TEST 6: HR sweep")
    log("="*70)
    for hr in [60,70,100,120]:
        r=simulate_beats(hr=hr,t_beats=10)
        log(f"HR {hr} → SV {r['SV']:.1f} CO {r['CO']:.1f} ({r['CO']*60:.0f})")
    log("[OK] HR sweep")
    return {"ok":True}
def test_drift():
    log("\n"+"="*70)
    log("TEST 7: Drift 600с t_calib")
    log("="*70)
    from scipy.integrate import solve_ivp
    h = make_heart()
    y0 = h.get_initial_state()
    inputs={'P_sa':85,'P_sv':12,'P_pa':15,'P_pv':12}
    def rhs(t,y): return h.get_derivatives(t,y,inputs)
    sol=solve_ivp(rhs,(0,600),y0,method='LSODA',rtol=1e-4,atol=1e-5,max_step=0.1)
    V_lv=sol.y[1]
    mean_580=np.mean(V_lv[sol.t>=580])
    mean_590=np.mean(V_lv[sol.t>=590])
    drift=abs(mean_590-mean_580)
    log(f"mean 580-600 {mean_580:.2f} 590-600 {mean_590:.2f} drift 10с {drift:.4f} nfev {sol.nfev}")
    ok = drift<2.0
    log(f"[{'OK' if ok else 'FAIL'}] drift 600с <2 мл")
    return {"ok":ok,"nfev":sol.nfev}
def test_strict_invariants():
    log("\n"+"="*70)
    log("TEST 8: Строгий — масса, клапаны, soft_clamp, PV-петля (НОВЫЙ)")
    log("="*70)
    res = simulate_beats(t_beats=10, return_full=True)
    t = res['t']
    V = res['V']
    flows = res['flows']
    pressures = res['pressures']
    idx_last = t >= t[-1] - (60/70)
    V0 = np.array([HEART_CFG['V0_la'],HEART_CFG['V0_lv'],HEART_CFG['V0_ra'],HEART_CFG['V0_rv']])[:,None]
    minV = np.min(V, axis=1)
    soft_clamp_ok = np.all(minV > (V0[:,0]*0.99))
    log(f"min V_la {minV[0]:.1f} V0 {V0[0,0]} | V_lv {minV[1]:.1f} V0 {V0[1,0]} | V_ra {minV[2]:.1f} | V_rv {minV[3]:.1f} → soft_clamp {soft_clamp_ok}")
    def regurg_frac(Q):
        pos = np.sum(Q[Q>0])
        neg = np.sum(np.abs(Q[Q<0]))
        return neg/(pos+1e-12)
    reg_aortic = regurg_frac(flows['Q_aortic'][idx_last])
    reg_pulm = regurg_frac(flows['Q_pulmonary'][idx_last])
    log(f"Регургитация аорта {reg_aortic*100:.3f}% <1% → {reg_aortic<0.01} | пульм {reg_pulm*100:.3f}% <1% → {reg_pulm<0.01}")
    valves_ok = reg_aortic<0.01 and reg_pulm<0.01
    mass_in = _trapz(flows['Q_sv_to_ra'][idx_last] + flows['Q_pv_to_la'][idx_last], t[idx_last])
    mass_out = _trapz(flows['Q_aortic'][idx_last] + flows['Q_pulmonary'][idx_last], t[idx_last])
    log(f"Mass ∫in {mass_in:.1f} vs ∫out {mass_out:.1f} Δ {mass_in-mass_out:.2f} мл/удар (<5 мл)")
    mass_ok = abs(mass_in-mass_out) < 5.0
    P_la_mean = np.mean(pressures['P_la'][idx_last])
    P_lv_min = np.min(pressures['P_lv'][idx_last])
    pv_ok1 = res['P_lv_max'] > 90
    pv_ok2 = P_lv_min < (P_la_mean+15)
    edv_esv = res['EDV']-res['ESV']
    sv_match = abs(edv_esv - res['SV']) / (res['SV']+1e-12)
    log(f"PV: P_lv_max {res['P_lv_max']:.1f} >90 {pv_ok1} | P_lv_min {P_lv_min:.1f} < P_la {P_la_mean:.1f}+15 {pv_ok2}")
    log(f"  EDV-ESV {edv_esv:.1f} vs SV {res['SV']:.1f} err {sv_match*100:.1f}% <5% → {sv_match<0.05}")
    pv_ok = pv_ok1 and pv_ok2 and sv_match<0.05
    no_nan = not (np.any(np.isnan(V)) or np.any(np.isinf(V)))
    log(f"  No NaN/Inf {no_nan}")
    ok = soft_clamp_ok and valves_ok and mass_ok and pv_ok and no_nan
    log(f"[{'OK' if ok else 'FAIL'}] строгие инварианты")
    return {"ok":ok}
def summary(results):
    log("\n"+"="*70)
    log("СВОДКА: Heart4Chambers?")
    log("="*70)
    names=["Интерфейс","Здоровый HR70","Frank-Starling","Afterload","VSD L→R","HR sweep","Drift 600с","Строгий инварианты (NEW)"]
    for n,r in zip(names,results):
        log(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    if all(r['ok'] for r in results):
        log("\nВЫВОД: Heart корректен, не причина P_sa 65, EDV 69.")
    else:
        log("\nВЫВОД: Требует правки — см. строгий тест.")
if __name__=="__main__":
    banner()
    r1=test_interface()
    r2=test_healthy()
    r3=test_frank_starling()
    r4=test_afterload()
    r5=test_vsd()
    r6=test_hr()
    r7=test_drift()
    r8=test_strict_invariants()
    summary([r1,r2,r3,r4,r5,r6,r7,r8])
    log("\nВход: P_sa,P_sv,P_pa,P_pv, hr_factor, inotropy_factor, baro_activation")
    log("Выход: Q_aortic (CO), Q_pulmonary, Q_vsd, P_la/lv/ra/rv")
    log(f"\nРезультаты в: {RESULT_FILE}")
    tee.close()
    print(f"\n[Сохранено в {RESULT_FILE}]")
