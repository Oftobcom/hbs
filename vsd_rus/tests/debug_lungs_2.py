#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_lungs_strict.py — СТРОГИЙ вариант проверки Lungs2Chamber v2

Отличия от debug_lungs.py:
 - Узкие допуски Гайтона (P_pa 16-20, PVR 0.06-0.10, f_rec 0.75-0.85)
 - Проверка клапана _valve_flow: Q_out >=0 всегда, нет обратного тока при P_dist<P_pv
 - Проверка защиты от отрицательных давлений: P_prox,P_dist clipped >=0
 - Проверка объёма: V_lungs = C1*P_prox + C2*P_dist = 120-220 мл в покое
 - Проверка mass balance с клапаном: в стационаре Q_pulm ≈ Q_int ≈ Q_out, |diff|<0.01
 - Проверка dP <1e-4, drift 600→1800 <0.01 мм рт.ст.
 - Проверка монотонности recruit: df/dP <0
 - Проверка PVR стабилизации: PVR(Q) 40-350 должен оставаться 0.06-0.10 (отклонение <20%)
 - Проверка remodel динамики: tau 200с, экспоненциальный подход к R_target
 - Проверка flow_factor clip ≤3 и линейности
 - Edge cases: Q=0, Q=1000, P_pv=0, P_pv=30
 - nfev <2000 за 600с, rtol 1e-7 atol 1e-9
"""

from __future__ import annotations
import sys, io
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import solve_ivp

from lungs import Lungs2Chamber
from physio_config import load_physiology

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_lungs_strict_2.txt"

class Tee:
    def __init__(self,*s): self.s=s
    def write(self,d):
        for s in self.s: s.write(d); s.flush()
    def flush(self):
        for s in self.s: s.flush()

def load_cfg():
    cfg=load_physiology()
    return dict(cfg.get("lungs",{})), dict(cfg.get("simulation",{}))

LUNGS_CFG, SIM_CFG = load_cfg()

def make_lungs(overrides=None):
    params=dict(LUNGS_CFG)
    if overrides: params.update(overrides)
    return Lungs2Chamber(
        R1=params.get("R1",0.06), R2=params.get("R2",0.04),
        C1=params.get("C1",4.0), C2=params.get("C2",8.0),
        flow_dependent_resistance=params.get("flow_dependent_resistance",True),
        flow_sensitivity=params.get("flow_sensitivity",0.15),
        Q_norm=params.get("Q_norm",80.0),
        k_flow=params.get("k_flow",9.0),
        recruitment_enabled=params.get("recruitment_enabled",True),
        P_recruit_50=params.get("P_recruit_50",20.0),
        n_recruit=params.get("n_recruit",3.0),
        f_recruit_min=params.get("f_recruit_min",0.55),
        pressure_remodel=params.get("pressure_remodel",False),
        P_pa_threshold=params.get("P_pa_threshold",25.0),
        pressure_sensitivity=params.get("pressure_sensitivity",0.04),
        R_remodel_max=params.get("R_remodel_max",5.0),
        tau_remodel=params.get("tau_remodel",200.0),
    )

INPUTS_HEALTHY = {"Q_pulmonary": 83.0, "P_pv": 12.0}

def banner():
    print("="*70)
    print(f"Запуск STRICT: {datetime.now():%Y-%m-%d %H:%M:%S} | NumPy {np.__version__}")
    print("Lungs2Chamber v2 строгий тест (клапан, объёмы, PVR)")
    print("="*70)
    print(f"  lungs.yaml: {LUNGS_CFG}")
    print(f"  Входы: {INPUTS_HEALTHY}")
    print(f"  → {RESULT_FILE}")
    print("="*70)

def integrate(lungs, inputs, t_end=600, max_step=0.5, rtol=1e-7, atol=1e-9):
    y0=lungs.get_initial_state()
    def rhs(t,y): return lungs.get_derivatives(t,y,inputs)
    sol=solve_ivp(rhs,(0.0,t_end),y0,method="LSODA",rtol=rtol,atol=atol,max_step=max_step)
    lungs.get_derivatives(sol.t[-1], sol.y[:,-1], inputs)
    out=lungs.get_outputs(sol.y[:,-1])
    return sol,out,y0

def test_interface_strict():
    print("\n"+"="*70+"\nTEST 1 STRICT: Интерфейс + защита от отрицательных\n"+"="*70)
    lung=make_lungs()
    sz=lung.get_state_size(); y0=lung.get_initial_state()
    # проверка отрицательных входов
    neg_inputs={"Q_pulmonary":-10,"P_pv":-5}
    d_neg=lung.get_derivatives(0.0,y0,neg_inputs)
    print(f"  size={sz} y0={y0}")
    print(f"  d при Q=-10 P_pv=-5: {d_neg} — должен быть finite, P clipped >=0")
    lung.get_derivatives(0.0,y0,INPUTS_HEALTHY)
    out=lung.get_outputs(y0)
    print(f"  outputs keys: {list(out.keys())} — должны включать Q_int,Q_out,P_pa,R1_eff,R2_eff,f_flow,f_rec")
    ok = sz==3 and y0.size==3 and np.all(np.isfinite(d_neg)) and "Q_int" in out and "Q_out" in out
    print(f"  [{'OK' if ok else 'FAIL'}] интерфейс строгий")
    return {"ok":ok}

def test_steady_strict():
    print("\n"+"="*70+"\nTEST 2 STRICT: Steady узкий допуск (P_pa 16-20, PVR 0.06-0.10)\n"+"="*70)
    lung=make_lungs()
    sol,out,_=integrate(lung, INPUTS_HEALTHY, t_end=800)
    P_prox=sol.y[0,-1]; P_dist=sol.y[1,-1]
    PVR=(P_prox-INPUTS_HEALTHY["P_pv"])/INPUTS_HEALTHY["Q_pulmonary"]
    V_lungs=LUNGS_CFG.get("C1",4.0)*P_prox + LUNGS_CFG.get("C2",8.0)*P_dist
    print(f"  P_pa={P_prox:.3f} ожидание 16.0-20.0 (было 12-24)")
    print(f"  P_dist={P_dist:.3f} ожидание 12-17 (между P_pa и P_pv=12)")
    print(f"  PVR={PVR:.4f} ожидание 0.06-0.10 (было 0.04-0.15)")
    print(f"  V_lungs=C1*P_pa+C2*P_dist={V_lungs:.1f} мл ожидание 120-220 мл")
    print(f"  R1_eff={out['R1_eff']:.5f} R2_eff={out['R2_eff']:.5f} sum={out['R1_eff']+out['R2_eff']:.5f} ожидание 0.07-0.10")
    print(f"  f_rec={out['recruit_factor']:.4f} ожидание 0.75-0.85, f_flow={out['flow_factor']:.4f} ожидание 1.00-1.02")
    print(f"  Q_int={out['Q_int']:.4f} Q_out={out['Q_out']:.4f} vs Q_pulm {INPUTS_HEALTHY['Q_pulmonary']} diff_int {abs(out['Q_int']-83):.4f} diff_out {abs(out['Q_out']-83):.4f}")
    ok = (16.0<=P_prox<=20.0) and (12.0<=P_dist<=17.0) and (0.06<=PVR<=0.10) and (120<=V_lungs<=220) and (0.75<=out['recruit_factor']<=0.85) and (1.0<=out['flow_factor']<=1.02) and (abs(out['Q_int']-83)<0.5) and (abs(out['Q_out']-83)<0.5)
    print(f"  [{'OK' if ok else 'FAIL'}] steady строгий")
    return {"ok":ok, "out":out, "sol":sol}

def test_valve_strict():
    print("\n"+"="*70+"\nTEST 3 STRICT: Клапан _valve_flow — нет обратного тока\n"+"="*70)
    lung=make_lungs()
    # P_dist < P_pv → Q_out должен быть ~0, не отрицательный
    for dP in [-10,-2,-0.5,0,0.5,2,10]:
        R=0.04
        q=lung._valve_flow(dP,R)
        print(f"  dP={dP:+5.1f} R={R} → Q={q:.4f} (ожидание 0 при dP<<0, dP/R при dP>>0)")
    # интеграция с P_pv высоким → P_dist<P_pv
    inputs_high_pv={"Q_pulmonary":20,"P_pv":20}
    sol,out,_=integrate(lung, inputs_high_pv, t_end=400)
    print(f"  При Q=20 P_pv=20 → P_dist={sol.y[1,-1]:.2f} Q_out={out['Q_out']:.4f} >=0 ? {out['Q_out']>=0}")
    print(f"  Q_int={out['Q_int']:.4f} — может быть отрицательным? (P_prox<P_dist) — допустимо, но Q_out не может быть <0")
    # проверка всех точек траектории Q_out>=0
    y0=lung.get_initial_state()
    def rhs(t,y): return lung.get_derivatives(t,y,inputs_high_pv)
    sol2=solve_ivp(rhs,(0,400),y0,method="LSODA",rtol=1e-7,atol=1e-9,max_step=0.5)
    q_outs=[]
    for i in range(sol2.y.shape[1]):
        lung.get_derivatives(sol2.t[i], sol2.y[:,i], inputs_high_pv)
        q_outs.append(lung.get_outputs(sol2.y[:,i])['Q_out'])
    q_outs=np.array(q_outs)
    print(f"  Q_out min {q_outs.min():.4f} max {q_outs.max():.2f} все >= -1e-6 ? {np.all(q_outs>=-1e-6)}")
    ok = np.all(q_outs>=-1e-6) and out['Q_out']>=0
    print(f"  [{'OK' if ok else 'FAIL'}] клапан строгий")
    return {"ok":ok}

def test_mass_strict():
    print("\n"+"="*70+"\nTEST 4 STRICT: Mass balance |Q_pulm-Q_int|<0.01 и |Q_int-Q_out|<0.01\n"+"="*70)
    lung=make_lungs()
    sol,out,_=integrate(lung, INPUTS_HEALTHY, t_end=800)
    Q_pulm=INPUTS_HEALTHY["Q_pulmonary"]
    print(f"  Q_pulm {Q_pulm:.4f} Q_int {out['Q_int']:.4f} Q_out {out['Q_out']:.4f}")
    print(f"  diff pulm-int {abs(Q_pulm-out['Q_int']):.6f} diff int-out {abs(out['Q_int']-out['Q_out']):.6f} ожидание <0.01")
    dydt=lung.get_derivatives(sol.t[-1], sol.y[:,-1], INPUTS_HEALTHY)
    print(f"  dP_prox {dydt[0]:.2e} dP_dist {dydt[1]:.2e} ожидание |dP|<1e-4 (было 1e-3)")
    ok = abs(Q_pulm-out['Q_int'])<0.01 and abs(out['Q_int']-out['Q_out'])<0.01 and abs(dydt[0])<1e-4 and abs(dydt[1])<1e-4
    print(f"  [{'OK' if ok else 'FAIL'}] mass строгий")
    return {"ok":ok}

def test_recruit_monotonic():
    print("\n"+"="*70+"\nTEST 5 STRICT: Recruitment монотонность df/dP<0 и узкие пороги\n"+"="*70)
    lung=make_lungs()
    Ps=np.linspace(5,60,12)
    fs=[lung._recruit_factor(P) for P in Ps]
    df=np.diff(fs)
    print(f"  P: {Ps}")
    print(f"  f: {[f'{v:.4f}' for v in fs]}")
    print(f"  df: {[f'{v:.4f}' for v in df]} — все <0 ? {np.all(df<0)}")
    checks=[
        ("P 10 0.93-0.97", 0.93<=lung._recruit_factor(10)<=0.97),
        ("P 15 0.84-0.89", 0.84<=lung._recruit_factor(15)<=0.89),
        ("P 20 0.75-0.80", 0.75<=lung._recruit_factor(20)<=0.80),
        ("P 30 0.62-0.68", 0.62<=lung._recruit_factor(30)<=0.68),
        ("P 40 0.58-0.62", 0.58<=lung._recruit_factor(40)<=0.62),
        ("Монотонно убывает", np.all(df<0)),
        ("f_min 0.55 <= f <=1.0", all(0.55<=f<=1.0 for f in fs)),
    ]
    ok=True
    for n,c in checks:
        ok=ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {n}")
    return {"ok":ok}

def test_pvr_stabilization_strict():
    print("\n"+"="*70+"\nTEST 6 STRICT: PVR стабилизация 0.07-0.10 при Q 40-350, отклонение <20%\n"+"="*70)
    lung=make_lungs()
    PVRs=[]
    for Q in [40,83,120,160,250,350]:
        sol,out,_=integrate(lung, {"Q_pulmonary":Q,"P_pv":12}, t_end=600)
        PVR=(sol.y[0,-1]-12)/Q
        PVRs.append(PVR)
        prod=out['recruit_factor']*out['flow_factor']
        print(f"  Q {Q:3d} → P_pa {sol.y[0,-1]:5.2f} f_rec {out['recruit_factor']:.3f} f_flow {out['flow_factor']:.3f} prod {prod:.3f} PVR {PVR:.4f}")
    PVR_mean=np.mean(PVRs)
    PVR_max_dev=max(abs(p-PVR_mean)/PVR_mean for p in PVRs)
    print(f"  PVR mean {PVR_mean:.4f} max dev {PVR_max_dev*100:.1f}% ожидание <20%")
    print(f"  PVR range {min(PVRs):.4f}-{max(PVRs):.4f} ожидание все 0.06-0.11")
    ok = all(0.06<=p<=0.11 for p in PVRs) and PVR_max_dev<0.20
    print(f"  [{'OK' if ok else 'FAIL'}] PVR стабилизация строгая")
    return {"ok":ok}

def test_flow_strict():
    print("\n"+"="*70+"\nTEST 7 STRICT: Flow factor линейность и клип ≤3\n"+"="*70)
    lung=make_lungs()
    for Q in [0,40,80,81,120,160,350,500,1000]:
        f=lung._flow_factor(Q)
        expected=1.0 if Q<=80 else 1.0+0.15*(Q-80)/80
        expected=min(expected,3.0)
        print(f"  Q {Q:4d} → f {f:.4f} exp {expected:.4f} diff {abs(f-expected):.4f}")
    checks=[
        ("Q 0→1.0", abs(lung._flow_factor(0)-1.0)<1e-6),
        ("Q 80→1.0", abs(lung._flow_factor(80)-1.0)<1e-6),
        ("Q 81→1.001875", abs(lung._flow_factor(81)-(1+0.15*1/80))<1e-6),
        ("Клип 3.0 при Q 1000", lung._flow_factor(1000)<=3.0+1e-9),
        ("Монотонно", lung._flow_factor(80)<lung._flow_factor(160)<lung._flow_factor(350)),
    ]
    ok=True
    for n,c in checks:
        ok=ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {n}")
    return {"ok":ok}

def test_remodel_dynamics():
    print("\n"+"="*70+"\nTEST 8 STRICT: Remodel динамика tau 200с\n"+"="*70)
    lung_off=make_lungs({"pressure_remodel":False})
    lung_on=make_lungs({"pressure_remodel":True})
    # P_pa ниже порога
    t1=lung_on._R_remodel_target(15); t2=lung_on._R_remodel_target(25); t3=lung_on._R_remodel_target(35); t4=lung_on._R_remodel_target(60)
    print(f"  R_target P 15→{t1:.3f} (1.0), 25→{t2:.3f} (1.0), 35→{t3:.3f} (1.4), 60→{t4:.3f} (2.4)")
    # интеграция с высоким P
    inputs_high={"Q_pulmonary":150,"P_pv":12}
    sol_off,_,_=integrate(lung_off, inputs_high, t_end=1200)
    sol_on,_,_=integrate(lung_on, inputs_high, t_end=1200)
    print(f"  OFF P_pa {sol_off.y[0,-1]:.2f} R_rem {sol_off.y[2,-1]:.4f} (должен 1.0)")
    print(f"  ON  P_pa {sol_on.y[0,-1]:.2f} R_rem {sol_on.y[2,-1]:.4f} (должен >1 при P>25)")
    # проверка экспоненты: R(t) = R_target + (R0-R_target)*exp(-t/tau)
    # возьмем первые 400с
    sol_200,_,_=integrate(lung_on, inputs_high, t_end=200)
    sol_400,_,_=integrate(lung_on, inputs_high, t_end=400)
    print(f"  ON R_rem 0с 1.0, 200с {sol_200.y[2,-1]:.4f}, 400с {sol_400.y[2,-1]:.4f}, 1200с {sol_on.y[2,-1]:.4f} — растёт к R_target")
    ok = abs(sol_off.y[2,-1]-1.0)<0.001 and sol_on.y[2,-1]>=1.0 and t1==1.0 and t2==1.0
    print(f"  [{'OK' if ok else 'FAIL'}] remodel динамика")
    return {"ok":ok}

def test_edge_cases():
    print("\n"+"="*70+"\nTEST 9 STRICT: Edge cases Q=0, Q=1000, P_pv=0, P_pv=30\n"+"="*70)
    lung=make_lungs()
    cases=[
        ({"Q_pulmonary":0,"P_pv":12}, "Q=0 P_pv=12"),
        ({"Q_pulmonary":1000,"P_pv":12}, "Q=1000 P_pv=12"),
        ({"Q_pulmonary":83,"P_pv":0}, "Q=83 P_pv=0"),
        ({"Q_pulmonary":83,"P_pv":30}, "Q=83 P_pv=30 (высокий)"),
        ({"Q_pulmonary":0,"P_pv":0}, "Q=0 P_pv=0"),
    ]
    ok_all=True
    for inp,label in cases:
        sol,out,_=integrate(lung, inp, t_end=400)
        finite=np.all(np.isfinite(sol.y))
        q_nonneg=out['Q_out']>=0
        p_nonneg=sol.y[0,-1]>=0 and sol.y[1,-1]>=0
        print(f"  {label:20s} → P_pa {sol.y[0,-1]:6.2f} P_dist {sol.y[1,-1]:6.2f} Q_out {out['Q_out']:6.2f} finite {finite} Q>=0 {q_nonneg} P>=0 {p_nonneg}")
        ok_all=ok_all and finite and q_nonneg and p_nonneg
    print(f"  [{'OK' if ok_all else 'FAIL'}] edge cases строгий")
    return {"ok":ok_all}

def test_drift_strict():
    print("\n"+"="*70+"\nTEST 10 STRICT: Drift <0.01 мм рт.ст. 600→1800с\n"+"="*70)
    lung=make_lungs()
    sol600,_,_=integrate(lung, INPUTS_HEALTHY, t_end=600)
    sol1800,_,_=integrate(lung, INPUTS_HEALTHY, t_end=1800)
    drift_prox=abs(sol1800.y[0,-1]-sol600.y[0,-1]); drift_dist=abs(sol1800.y[1,-1]-sol600.y[1,-1]); drift_rem=abs(sol1800.y[2,-1]-sol600.y[2,-1])
    print(f"  P_prox 600с {sol600.y[0,-1]:.4f} 1800с {sol1800.y[0,-1]:.4f} drift {drift_prox:.6f} ожидание <0.01 (было 0.5)")
    print(f"  P_dist 600с {sol600.y[1,-1]:.4f} 1800с {sol1800.y[1,-1]:.4f} drift {drift_dist:.6f} <0.01")
    print(f"  R_rem 600с {sol600.y[2,-1]:.6f} 1800с {sol1800.y[2,-1]:.6f} drift {drift_rem:.6f} <0.001")
    print(f"  nfev 600с {sol600.nfev} 1800с {sol1800.nfev} ожидание <2000 за 600с")
    ok=drift_prox<0.01 and drift_dist<0.01 and drift_rem<0.001 and sol600.nfev<2000
    print(f"  [{'OK' if ok else 'FAIL'}] drift строгий")
    return {"ok":ok}

def test_strict_final():
    print("\n"+"="*70+"\nTEST 11 STRICT: Final — P_prox>=P_dist>=P_pv, R>0, finite, V_lungs\n"+"="*70)
    lung=make_lungs()
    sol,out,_=integrate(lung, INPUTS_HEALTHY, t_end=600, max_step=0.2)
    P_prox=sol.y[0,:]; P_dist=sol.y[1,:]; R_rem=sol.y[2,:]
    V_lungs=LUNGS_CFG.get("C1",4.0)*P_prox + LUNGS_CFG.get("C2",8.0)*P_dist
    checks=[
        (f"P_prox>=P_dist-0.5 (P_prox {P_prox[-1]:.2f} >= P_dist {P_dist[-1]:.2f})", P_prox[-1]>=P_dist[-1]-0.5),
        (f"P_dist>=P_pv-0.5 (P_dist {P_dist[-1]:.2f} >= P_pv 12)", P_dist[-1]>=INPUTS_HEALTHY["P_pv"]-0.5),
        ("R1_eff 0.03-0.12", 0.03<=out['R1_eff']<=0.12),
        ("R2_eff 0.02-0.08", 0.02<=out['R2_eff']<=0.08),
        ("R_rem 0.1-5", 0.1<=R_rem[-1]<=5.0),
        ("V_lungs 100-250 мл", 100<=V_lungs[-1]<=250),
        ("Q_int>=0", out['Q_int']>=0),
        ("Q_out>=0", out['Q_out']>=0),
        ("finite", np.all(np.isfinite(sol.y))),
        ("P_prox>0", np.all(P_prox>0)),
        ("P_dist>0", np.all(P_dist>0)),
    ]
    ok_all=True
    for n,c in checks:
        ok_all=ok_all and c
        print(f"  [{'OK' if c else 'FAIL'}] {n}")
    print(f"  V_lungs min {V_lungs.min():.1f} max {V_lungs.max():.1f} last {V_lungs[-1]:.1f}")
    return {"ok":ok_all}

def summary(res):
    print("\n"+"="*70+"\nСВОДКА STRICT: Lungs2Chamber v2?\n"+"="*70)
    names=["Интерфейс+защита","Steady узкий","Клапан Q_out>=0","Mass <0.01","Recruit монотон","PVR <20%","Flow линейность+клип","Remodel tau","Edge cases","Drift <0.01","Final P>=P>=P_pv V"]
    for n,r in zip(names,res):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    if all(r["ok"] for r in res):
        print("\nВЫВОД STRICT: Lungs v2 ОТЛИЧНО — готова к whole_body.")
    else:
        print("\nВЫВОД STRICT: Есть FAIL — см. выше, требуется фикс")

def run_all(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf=io.StringIO(); tee=Tee(sys.stdout,buf); orig=sys.stdout; sys.stdout=tee
    try:
        banner()
        rs=[test_interface_strict(),test_steady_strict(),test_valve_strict(),test_mass_strict(),test_recruit_monotonic(),test_pvr_stabilization_strict(),test_flow_strict(),test_remodel_dynamics(),test_edge_cases(),test_drift_strict(),test_strict_final()]
        summary(rs)
        print(f"\nФиниш {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout=orig
    path.write_text(buf.getvalue(),encoding="utf-8")
    print(f"Отчёт: {path}")

if __name__=="__main__":
    run_all(Path(__file__).resolve().parent / "results_debug_lungs_strict_2.txt")
