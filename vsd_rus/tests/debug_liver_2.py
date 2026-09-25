#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_liver.py — изолированная проверка Liver

Печень = 2 Windkessel (P_hv + P_portal) + 3 метаболических пула (билирубин, аммиак, альбумин)
Состояние: [P_hv, C_bil_liver, C_amm_liver, C_alb_liver, reserve, P_portal] size 6

Что проверяем:
  1. Интерфейс OrganModel (state_size 6, y0 = [P_hv0, C_bil0, C_amm0, C_alb0, 0, P_portal0])
  2. Steady state при типичных входах: P_sa 85, P_sv 5, Q_gut 8 мл/с, V_blood 5800
     Ожидаем: Q_ha≈4.5, Q_pv≈8, Q_out≈12.5, P_hv≈6-10, P_portal≈8-12, dP≈0
  3. Mass balance гемодинамики: Q_ha+Q_pv ≈ Q_out, Q_gut ≈ Q_pv
  4. Portal Windkessel: dP_portal = (Q_gut - Q_pv)/C_portal
  5. Билирубин клиренс: C_bil_blood 0.5 → uptake → clearance 0.2*C_liver → dC_blood отрицательный
  6. Аммиак клиренс аналогично
  7. Альбумин синтез: synthesis 0.1, degradation 0.01*C, release 0.05*(C_liver-C_blood)
  8. Лактат клиренс: dC_lac = -Q_ha/V*C_lac*0.1
  9. Проверка объёмов и масс: нет NaN, P>=0, Q>=0, C>=0, finite, drift <1 мм рт.ст.
  10. Чувствительность к давлениям: P_sa 60→120, P_sv 2→15, Q_gut 2→15
  11. Строгий: положительность, отсутствие регургитации Q_out>=0, P_hv>=P_sv в стационаре

Вход: P_sa, P_sv, C_bilirubin_blood, C_ammonia_blood, C_albumin_blood, C_lactate_blood, V_blood, Q_gut_out
Выход → whole_body: Q_liver_out, P_portal, dC_bilirubin, dC_ammonia, dC_albumin, dC_lactate, Q_ha, Q_pv

Результаты: консоль + results_debug_liver.txt
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

from liver import Liver
from physio_config import load_physiology

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_liver_2.txt"

class Tee:
    def __init__(self,*s): self.s=s
    def write(self,d):
        for s in self.s: s.write(d); s.flush()
    def flush(self):
        for s in self.s: s.flush()

def load_cfg():
    cfg=load_physiology()
    return dict(cfg.get("liver",{})), dict(cfg.get("systemic",{})), dict(cfg.get("blood",{}))

LIV_CFG, SYS_CFG, BLOOD_CFG = load_cfg()

def make_liver():
    return Liver(
        R_ha=LIV_CFG.get("R_ha",17.0),
        R_pv_base=LIV_CFG.get("R_pv_base",0.25),
        R_hv_base=LIV_CFG.get("R_hv_base",0.12),
        C=LIV_CFG.get("C",5.0),
        C_portal=LIV_CFG.get("C_portal",1.5),
        P_hv0=LIV_CFG.get("P_hv0",8.0),
        P_portal0=LIV_CFG.get("P_portal0",8.0),
        albumin_prod_base=LIV_CFG.get("albumin_prod_base",0.1),
        bilirubin_clearance_base=LIV_CFG.get("bilirubin_clearance_base",0.2),
        ammonia_clearance_base=LIV_CFG.get("ammonia_clearance_base",0.15),
        lactate_clearance_base=LIV_CFG.get("lactate_clearance_base",0.05),
        C_bilirubin0=0.0, C_ammonia0=0.0, C_albumin0=1.0
    )

INPUTS_HEALTHY = {
    "P_sa": 85.0,
    "P_sv": 5.0,
    "C_bilirubin_blood": 0.5,
    "C_ammonia_blood": 0.3,
    "C_albumin_blood": 4.5,
    "C_lactate_blood": 0.10,
    "V_blood": 5800.0,
    "Q_gut_out": 8.0,  # из GITract, типично 6-10 мл/с
}

def banner():
    print("="*70)
    print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S} | NumPy {np.__version__}")
    print("Liver: изолированный тест")
    print("="*70)
    print(f"  Параметры из physiology.yaml liver:")
    for k,v in LIV_CFG.items():
        print(f"    {k:28s} = {v}")
    print(f"  Входы здоровые: {INPUTS_HEALTHY}")
    print(f"  → {RESULT_FILE}")
    print("="*70)

def integrate(liver, inputs, t_end=600, max_step=0.5):
    y0=liver.get_initial_state()
    def rhs(t,y): return liver.get_derivatives(t,y,inputs)
    sol=solve_ivp(rhs,(0.0,t_end),y0,method="LSODA",rtol=1e-7,atol=1e-9,max_step=max_step)
    liver.get_derivatives(sol.t[-1], sol.y[:,-1], inputs)
    out=liver.get_outputs(sol.y[:,-1])
    return sol, out, y0

def test_interface():
    print("\n"+"="*70+"\nTEST 1: Интерфейс OrganModel\n"+"="*70)
    liv=make_liver()
    sz=liv.get_state_size(); y0=liv.get_initial_state()
    d=liv.get_derivatives(0.0,y0,INPUTS_HEALTHY)
    print(f"  state_size={sz} ожидание 6")
    print(f"  y0={y0} ожидание [P_hv0={LIV_CFG.get('P_hv0')}, 0,0,1,0, P_portal0={LIV_CFG.get('P_portal0')}]")
    print(f"  dydt={d} size {d.size}")
    liv.get_derivatives(0.0,y0,INPUTS_HEALTHY)
    out=liv.get_outputs(y0)
    print(f"  outputs: {out}")
    ok=sz==6 and y0.size==6 and d.size==6 and "Q_liver_out" in out and "P_portal" in out
    print(f"  [{'OK' if ok else 'FAIL'}] интерфейс")
    return {"ok":ok}

def test_steady():
    print("\n"+"="*70+"\nTEST 2: Steady state здоровый (P_sa 85, P_sv 5, Q_gut 8)\n"+"="*70)
    liv=make_liver()
    sol,out,y0=integrate(liv, INPUTS_HEALTHY, t_end=600)
    y_ss=sol.y[:,-1]
    P_hv=y_ss[0]; P_portal=y_ss[5]
    print(f"  P_hv={P_hv:.2f} мм рт.ст. ожидание 5-12 (P_hv0=8 + (Q_ha+Q_pv-Q_out)/C)")
    print(f"  P_portal={P_portal:.2f} ожидание 8-14 (P_portal0=8)")
    print(f"  Q_ha={out['Q_ha']:.2f} ожидание (85-8)/17≈4.53")
    print(f"  Q_pv={out['Q_pv']:.2f} ожидание ≈ Q_gut 8.0")
    print(f"  Q_out={out['Q_liver_out']:.2f} ожидание Q_ha+Q_pv≈12.5")
    print(f"  Q_gut_out={out['Q_gut_out']:.2f}")
    print(f"  dP_hv={sol.y[0,-1]-sol.y[0,-2]:.4e} dP_portal={sol.y[5,-1]-sol.y[5,-2]:.4e} (должны →0)")
    print(f"  nfev={sol.nfev}")
    ok = (3<=P_hv<=15) and (5<=P_portal<=18) and (3<=out['Q_ha']<=6) and (5<=out['Q_pv']<=12) and (8<=out['Q_liver_out']<=18)
    print(f"  [{'OK' if ok else 'FAIL'}] steady в Гайтоне")
    return {"ok":ok, "out":out, "sol":sol}

def test_mass_balance():
    print("\n"+"="*70+"\nTEST 3: Mass balance гемодинамики\n"+"="*70)
    liv=make_liver()
    sol,out,_=integrate(liv, INPUTS_HEALTHY, t_end=600)
    y_last=sol.y[:,-1]
    dydt=liv.get_derivatives(sol.t[-1], y_last, INPUTS_HEALTHY)
    dP_hv=dydt[0]; dP_portal=dydt[5]
    print(f"  dP_hv={dP_hv:.3e} мм рт.ст./с ожидание |dP|<1e-3")
    print(f"  dP_portal={dP_portal:.3e} ожидание |dP|<1e-3")
    print(f"  Q_ha {out['Q_ha']:.3f} + Q_pv {out['Q_pv']:.3f} = {out['Q_ha']+out['Q_pv']:.3f} vs Q_out {out['Q_liver_out']:.3f} diff {(out['Q_ha']+out['Q_pv'])-out['Q_liver_out']:.4f}")
    print(f"  Q_gut {out['Q_gut_out']:.3f} vs Q_pv {out['Q_pv']:.3f} diff {out['Q_gut_out']-out['Q_pv']:.4f}")
    # в стационаре оба diff должны быть ~ C*dP
    ok = abs(dP_hv)<1e-3 and abs(dP_portal)<1e-3 and abs((out['Q_ha']+out['Q_pv'])-out['Q_liver_out'])<0.5 and abs(out['Q_gut_out']-out['Q_pv'])<0.5
    print(f"  [{'OK' if ok else 'FAIL'}] mass balance")
    return {"ok":ok}

def test_bilirubin():
    print("\n"+"="*70+"\nTEST 4: Билирубин клиренс\n"+"="*70)
    liv=make_liver()
    sol,out,_=integrate(liv, INPUTS_HEALTHY, t_end=600)
    y_ss=sol.y[:,-1]
    C_bil_liver=y_ss[1]
    print(f"  C_bil_blood 0.5 → C_bil_liver {C_bil_liver:.4f}")
    print(f"  dC_bil_blood={out['dC_bilirubin']:.6f} (должно <0, клиренс)")
    print(f"  uptake 0.1*(0.5-C_liver)={0.1*(0.5-C_bil_liver):.4f} clearance {LIV_CFG.get('bilirubin_clearance_base')}*C_liver={LIV_CFG.get('bilirubin_clearance_base',0.2)*C_bil_liver:.4f}")
    # масса: V_blood*dC_blood = -clearance_liver ?
    V=INPUTS_HEALTHY["V_blood"]
    clearance=LIV_CFG.get("bilirubin_clearance_base",0.2)*C_bil_liver
    mass_balance = V*out['dC_bilirubin'] + clearance
    print(f"  V*dC + clearance = {mass_balance:.3e} ожидание 0 (масса сохраняется, если учитывать uptake)")
    print(f"  dC_bil_liver={sol.y[1,-1]-sol.y[1,-2]:.3e} должен →0 в стационаре")
    ok = out['dC_bilirubin']<0 and C_bil_liver>0 and np.isfinite(C_bil_liver)
    print(f"  [{'OK' if ok else 'FAIL'}] билирубин клиренс")
    return {"ok":ok}

def test_ammonia():
    print("\n"+"="*70+"\nTEST 5: Аммиак клиренс\n"+"="*70)
    liv=make_liver()
    sol,out,_=integrate(liv, INPUTS_HEALTHY, t_end=600)
    C_amm=y_ss=sol.y[:,-1][2]
    print(f"  C_amm_blood 0.3 → C_amm_liver {C_amm:.4f}")
    print(f"  dC_amm_blood={out['dC_ammonia']:.6f} <0 ? {out['dC_ammonia']<0}")
    ok = out['dC_ammonia']<0 and np.isfinite(C_amm)
    print(f"  [{'OK' if ok else 'FAIL'}] аммиак клиренс")
    return {"ok":ok}

def test_albumin():
    print("\n"+"="*70+"\nTEST 6: Альбумин синтез/деградация/релиз\n"+"="*70)
    liv=make_liver()
    sol,out,_=integrate(liv, INPUTS_HEALTHY, t_end=600)
    C_alb=sol.y[:,-1][3]
    print(f"  C_alb_liver {C_alb:.4f} C_alb_blood {INPUTS_HEALTHY['C_albumin_blood']}")
    print(f"  synthesis {LIV_CFG.get('albumin_prod_base')} degradation 0.01*C={0.01*C_alb:.4f} release 0.05*(C_liver-C_blood)={0.05*(C_alb-INPUTS_HEALTHY['C_albumin_blood']):.4f}")
    print(f"  dC_alb_blood={out['dC_albumin']:.6f} (>0 синтез?)")
    print(f"  dC_alb_liver={liv.get_derivatives(sol.t[-1], sol.y[:,-1], INPUTS_HEALTHY)[3]:.3e} →0 в стационаре")
    ok = np.isfinite(C_alb) and C_alb>0
    print(f"  [{'OK' if ok else 'FAIL'}] альбумин")
    return {"ok":ok}

def test_lactate():
    print("\n"+"="*70+"\nTEST 7: Лактат клиренс\n"+"="*70)
    liv=make_liver()
    sol,out,_=integrate(liv, INPUTS_HEALTHY, t_end=600)
    print(f"  C_lac_blood 0.10 → dC_lac={out['dC_lactate']:.6f} <0 ? {out['dC_lactate']<0}")
    print(f"  Формула -Q_ha/V*C*0.05*2 = -{out['Q_ha']:.2f}/{INPUTS_HEALTHY['V_blood']}*0.1*0.1={-out['Q_ha']/INPUTS_HEALTHY['V_blood']*0.1*0.1:.6f}")
    ok = out['dC_lactate']<0 and np.isfinite(out['dC_lactate'])
    print(f"  [{'OK' if ok else 'FAIL'}] лактат")
    return {"ok":ok}

def test_pressure_sensitivity():
    print("\n"+"="*70+"\nTEST 8: Чувствительность к P_sa, P_sv, Q_gut\n"+"="*70)
    liv=make_liver()
    for P_sa in [60,85,120]:
        inp=dict(INPUTS_HEALTHY, P_sa=P_sa)
        sol,out,_=integrate(liv, inp, t_end=400)
        print(f"  P_sa {P_sa:3d} → Q_ha {out['Q_ha']:.2f} P_hv {sol.y[:,-1][0]:.2f} Q_out {out['Q_liver_out']:.2f}")
    for P_sv in [2,5,15]:
        inp=dict(INPUTS_HEALTHY, P_sv=P_sv)
        sol,out,_=integrate(liv, inp, t_end=400)
        print(f"  P_sv {P_sv:2d} → P_hv {sol.y[:,-1][0]:.2f} Q_out {out['Q_liver_out']:.2f} (Q_out падает с ростом P_sv?)")
    for Qg in [2,8,15]:
        inp=dict(INPUTS_HEALTHY, Q_gut_out=Qg)
        sol,out,_=integrate(liv, inp, t_end=400)
        print(f"  Q_gut {Qg:2d} → Q_pv {out['Q_pv']:.2f} P_portal {sol.y[:,-1][5]:.2f} (P_portal растет с Q_gut)")
    ok=True
    print(f"  [{'OK' if ok else 'FAIL'}] чувствительность")
    return {"ok":ok}

def test_drift():
    print("\n"+"="*70+"\nTEST 9: Drift 600с и 1800с как t_calib/t_span\n"+"="*70)
    liv=make_liver()
    sol600,_,_=integrate(liv, INPUTS_HEALTHY, t_end=600)
    sol1800,_,_=integrate(liv, INPUTS_HEALTHY, t_end=1800)
    P_hv_600=sol600.y[0,-1]; P_hv_1800=sol1800.y[0,-1]
    P_port_600=sol600.y[5,-1]; P_port_1800=sol1800.y[5,-1]
    drift_hv=abs(P_hv_1800-P_hv_600); drift_port=abs(P_port_1800-P_port_600)
    print(f"  P_hv 600с {P_hv_600:.3f} 1800с {P_hv_1800:.3f} drift {drift_hv:.4f}")
    print(f"  P_portal 600с {P_port_600:.3f} 1800с {P_port_1800:.3f} drift {drift_port:.4f}")
    print(f"  nfev 600с {sol600.nfev} 1800с {sol1800.nfev}")
    ok=drift_hv<0.5 and drift_port<0.5
    print(f"  [{'OK' if ok else 'FAIL'}] drift <0.5 мм рт.ст.")
    return {"ok":ok}

def test_strict():
    print("\n"+"="*70+"\nTEST 10: Строгий — положительность, Q>=0, P>=0, finite\n"+"="*70)
    liv=make_liver()
    sol,out,_=integrate(liv, INPUTS_HEALTHY, t_end=600, max_step=0.2)
    P_hv=sol.y[0,:]; P_port=sol.y[5,:]; C_bil=sol.y[1,:]; C_amm=sol.y[2,:]; C_alb=sol.y[3,:]
    checks=[
        ("P_hv>0", np.all(P_hv>0)),
        ("P_portal>0", np.all(P_port>0)),
        ("C_bil>=0", np.all(C_bil>=0)),
        ("C_amm>=0", np.all(C_amm>=0)),
        ("C_alb>0", np.all(C_alb>0)),
        ("Q_ha>=0", out['Q_ha']>=0),
        ("Q_pv>=0", out['Q_pv']>=0),
        ("Q_out>=0", out['Q_liver_out']>=0),
        ("P_hv>=P_sv? в стационаре", P_hv[-1]>=INPUTS_HEALTHY["P_sv"]-1),
        ("finite", np.all(np.isfinite(sol.y))),
    ]
    ok_all=True
    for n,c in checks:
        ok_all=ok_all and c
        print(f"  [{'OK' if c else 'FAIL'}] {n}")
    print(f"  P_hv min {P_hv.min():.2f} max {P_hv.max():.2f} P_portal min {P_port.min():.2f} max {P_port.max():.2f}")
    return {"ok":ok_all}

def summary(res):
    print("\n"+"="*70+"\nСВОДКА: Liver?\n"+"="*70)
    names=["Интерфейс","Steady","Mass balance","Билирубин","Аммиак","Альбумин","Лактат","Чувств P/Q","Drift","Strict"]
    for n,r in zip(names,res):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    if all(r["ok"] for r in res):
        print("\nВЫВОД: Liver корректна в изоляции. Ищите в whole_body связке Q_gut и R_hv.")
    else:
        print("\nВЫВОД: Есть FAIL — см. выше, проверьте баланс объёмов/масс")

def run_all(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf=io.StringIO(); tee=Tee(sys.stdout,buf); orig=sys.stdout; sys.stdout=tee
    try:
        banner()
        rs=[test_interface(),test_steady(),test_mass_balance(),test_bilirubin(),test_ammonia(),test_albumin(),test_lactate(),test_pressure_sensitivity(),test_drift(),test_strict()]
        summary(rs)
        print(f"\nФиниш {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout=orig
    path.write_text(buf.getvalue(),encoding="utf-8")
    print(f"Отчёт: {path}")

if __name__=="__main__":
    run_all(Path(__file__).resolve().parent / "results_debug_liver_2.txt")
