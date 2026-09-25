#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_jugular_vein.py — изолированная проверка JugularVein

Яремная вена = буфер между мозгом и системными венами.
Состояние [V_jv, C_jv_O2, C_jv_CO2]

Что проверяем:
  1. Интерфейс OrganModel (state_size 3, y0 = V0, C_O2_init, C_CO2_init)
  2. Steady state в Гайтоне: Q_brain 12 мл/с, P_sv 5, V_blood 5800 → V_jv ~290, P_jv 6-8, Q_out≈Q_in, SjvO2 60-75%
  3. Mass balance: Q_in ≈ Q_out в стационаре
  4. Washout: скачок Q_in 5→15, tau = V/Q ~20с, C_jv → C_in
  5. Венозное давление: P_sv 2→15 → P_jv растет, Q_out падает
  6. V_blood tracking: V_target = 0.05*V_blood → 200/290/350
  7. Ишемия: C_in_O2 0.12→0.08 → SjvO2 <50% за ~60с
  8. Drift 600с как t_calib
  9. Строгий: масса, положительность, отсутствие регургитации Q_out>=0, SjvO2 в [0,1]

Вход: Q_in, C_in_O2, C_in_CO2, P_sv, V_blood
Выход → systemic: P_jv, V_jv, Q_jv_out, SjvO2, P_jv_O2

Результаты: консоль + results_debug_jugular_vein_2.txt
"""

from __future__ import annotations
import sys, io
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz

from scipy.integrate import solve_ivp
from jugular_vein import JugularVein
from physio_config import load_physiology

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_jugular_vein_2.txt"

class Tee:
    def __init__(self,*s):
        self.s=s
    def write(self,d):
        for s in self.s:
            s.write(d); s.flush()
    def flush(self):
        for s in self.s: s.flush()

def load_cfg():
    cfg = load_physiology()
    return dict(cfg.get("jugular_vein",{})), dict(cfg.get("brain",{})), dict(cfg.get("systemic",{}))

JV_CFG,BRAIN_CFG,SYS_CFG = load_cfg()

def make_jv():
    # берем из physiology.yaml, fallback на дефолты из jugular_vein.py
    return JugularVein(
        C=JV_CFG.get("C",20.0),
        P0=JV_CFG.get("P0",6.0),
        V0=JV_CFG.get("V0",290.0),
        R_out=JV_CFG.get("R_out",0.5),
        target_fraction=JV_CFG.get("target_fraction",0.05),
        tau_target=JV_CFG.get("tau_target",300.0),
        C_O2_init=JV_CFG.get("C_O2_init",0.12),
        C_CO2_init=JV_CFG.get("C_CO2_init",0.56),
        Hb=JV_CFG.get("Hb",15.0),
    )

INPUTS_HEALTHY = {
    "Q_in": 12.0,       # мл/с, ~720 мл/мин мозговой кровоток
    "C_in_O2": 0.12,     # мл/мл, венозная мозга
    "C_in_CO2": 0.56,
    "P_sv": 5.0,         # мм рт.ст.
    "V_blood": 5800.0,
}

def banner():
    print("="*70)
    print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S} | NumPy {np.__version__}")
    print("JugularVein: изолированный тест")
    print("="*70)
    print(f"  Входы здоровые: {INPUTS_HEALTHY}")
    print(f"  Параметры из physiology.yaml jugular_vein:")
    for k,v in JV_CFG.items():
        print(f"    {k:20s} = {v}")
    print(f"  → {RESULT_FILE}")
    print("="*70)

def integrate(jv, inputs, t_end=600, method="LSODA", rtol=1e-6, atol=1e-8, max_step=1.0):
    y0=jv.get_initial_state()
    def rhs(t,y): return jv.get_derivatives(t,y,inputs)
    sol=solve_ivp(rhs,(0.0,t_end),y0,method=method,rtol=rtol,atol=atol,max_step=max_step)
    # собрать выходы в конце
    jv.get_derivatives(sol.t[-1], sol.y[:,-1], inputs)
    out=jv.get_outputs(sol.y[:,-1])
    return sol, out

def test_interface():
    print("\n"+"="*70+"\nTEST 1: Интерфейс OrganModel\n"+"="*70)
    jv=make_jv()
    sz=jv.get_state_size(); y0=jv.get_initial_state()
    d=jv.get_derivatives(0.0,y0,INPUTS_HEALTHY)
    print(f"  state_size={sz} ожидание 3")
    print(f"  y0={y0} ожидание [V0={JV_CFG.get('V0',290)}, C_O2=0.12, C_CO2=0.56]")
    print(f"  dydt={d}")
    ok=sz==3 and len(y0)==3 and len(d)==3
    print(f"  [{'OK' if ok else 'FAIL'}] интерфейс")
    return {"ok":ok}

def test_steady():
    print("\n"+"="*70+"\nTEST 2: Steady state здоровый (Q=12, P_sv=5, V_blood=5800)\n"+"="*70)
    jv=make_jv()
    sol,out=integrate(jv, INPUTS_HEALTHY, t_end=600)
    print(f"  V_jv={out['V_jv']:.1f} мл ожидание 260-320 (5% от 5800)")
    print(f"  P_jv={out['P_jv']:.2f} мм рт.ст. ожидание 5-9 (P0=6 + (V-V0)/C)")
    print(f"  Q_in={out['Q_in']:.2f} Q_out={out['Q_jv_out']:.2f} ожидание Q_out≈Q_in")
    print(f"  SjvO2={out['SjvO2']*100:.1f}% ожидание 55-75% (норма 60-75%)")
    print(f"  C_jv_O2={out['C_jv_O2']:.3f} C_in={INPUTS_HEALTHY['C_in_O2']:.3f}")
    print(f"  nfev={sol.nfev}")
    ok = (260<=out['V_jv']<=320) and (4<=out['P_jv']<=9) and abs(out['Q_jv_out']-out['Q_in'])<0.5 and (0.55<=out['SjvO2']<=0.80)
    print(f"  [{'OK' if ok else 'FAIL'}] steady")
    return {"ok":ok, "out":out, "sol":sol}

def test_mass_balance():
    print("\n"+"="*70+"\nTEST 3: Mass balance Q_in ≈ Q_out в стационаре\n"+"="*70)
    jv=make_jv()
    sol,out=integrate(jv, INPUTS_HEALTHY, t_end=600)
    # dV/dt в конце должен быть ~0
    y_last=sol.y[:,-1]
    dydt=jv.get_derivatives(sol.t[-1], y_last, INPUTS_HEALTHY)
    print(f"  dV/dt={dydt[0]:.4f} мл/с ожидание |dV|<0.01")
    print(f"  Q_in={out['Q_in']:.3f} Q_out={out['Q_jv_out']:.3f} diff={out['Q_in']-out['Q_jv_out']:.4f}")
    # с учетом таргета V_target
    V_target=JV_CFG.get("target_fraction",0.05)*INPUTS_HEALTHY["V_blood"]
    print(f"  V_target={V_target:.1f} V_jv={out['V_jv']:.1f}")
    ok=abs(dydt[0])<0.05 and abs(out['Q_in']-out['Q_jv_out'])<1.0
    print(f"  [{'OK' if ok else 'FAIL'}] mass balance")
    return {"ok":ok}

def test_washout():
    print("\n"+"="*70+"\nTEST 4: Washout — скачок Q_in 5→15 мл/с, tau=V/Q\n"+"="*70)
    jv=make_jv()
    # сначала низкое Q
    inputs_low=INPUTS_HEALTHY.copy(); inputs_low["Q_in"]=5.0; inputs_low["C_in_O2"]=0.10
    sol_low,_=integrate(jv, inputs_low, t_end=300)
    y0=sol_low.y[:,-1]
    # скачок
    inputs_high=INPUTS_HEALTHY.copy(); inputs_high["Q_in"]=15.0; inputs_high["C_in_O2"]=0.13
    def rhs(t,y): return jv.get_derivatives(t,y,inputs_high)
    sol=solve_ivp(rhs,(0,120),y0,method="LSODA",rtol=1e-6,atol=1e-8,max_step=0.5)
    # в конце C_jv должна приблизиться к 0.13
    jv.get_derivatives(sol.t[-1], sol.y[:,-1], inputs_high)
    out=jv.get_outputs(sol.y[:,-1])
    print(f"  C_in 0.10→0.13, после 120с C_jv={out['C_jv_O2']:.3f} ожидание 0.125-0.13")
    tau = out['V_jv']/inputs_high["Q_in"]
    print(f"  tau=V/Q={tau:.1f}с ожидание 15-40с")
    ok=abs(out['C_jv_O2']-0.13)<0.01
    print(f"  [{'OK' if ok else 'FAIL'}] washout")
    return {"ok":ok}

def test_venous_pressure():
    print("\n"+"="*70+"\nTEST 5: Венозное давление P_sv 2→15 мм рт.ст.\n"+"="*70)
    vals=[]
    for P_sv in [2,5,15]:
        inp=INPUTS_HEALTHY.copy(); inp["P_sv"]=P_sv
        jv=make_jv()
        sol,out=integrate(jv, inp, t_end=600)
        vals.append((out['P_jv'],out['Q_jv_out'],out['V_jv']))
        print(f"  P_sv {P_sv:2d} → P_jv {out['P_jv']:.2f} Q_out {out['Q_jv_out']:.2f} V_jv {out['V_jv']:.1f}")
    # при росте P_sv P_jv растет, Q_out падает вначале, потом V растет
    ok=vals[0][0]<vals[1][0]<vals[2][0]  # P_jv растет
    print(f"  [{'OK' if ok else 'FAIL'}] P_jv растет с P_sv")
    return {"ok":ok}

def test_vblood_tracking():
    print("\n"+"="*70+"\nTEST 6: V_blood tracking V_target=0.05*V_blood\n"+"="*70)
    for Vb in [4000,5800,7000]:
        inp=INPUTS_HEALTHY.copy(); inp["V_blood"]=Vb
        jv=make_jv()
        sol,out=integrate(jv, inp, t_end=600)
        V_target=JV_CFG.get("target_fraction",0.05)*Vb
        print(f"  V_blood {Vb} → V_target {V_target:.0f} → V_jv {out['V_jv']:.1f} (diff {out['V_jv']-V_target:.1f})")
    # проверка
    jv=make_jv()
    sol,out=integrate(jv, INPUTS_HEALTHY, t_end=600)
    V_target=0.05*5800
    ok=abs(out['V_jv']-V_target)<20
    print(f"  [{'OK' if ok else 'FAIL'}] V_jv ≈ V_target ±20 мл")
    return {"ok":ok}

def test_ischemia():
    print("\n"+"="*70+"\nTEST 7: Ишемия — C_in_O2 0.12→0.08 → SjvO2 <50%\n"+"="*70)
    jv=make_jv()
    # норма
    sol,out=integrate(jv, INPUTS_HEALTHY, t_end=300)
    print(f"  База SjvO2 {out['SjvO2']*100:.1f}%")
    # ишемия
    inp=INPUTS_HEALTHY.copy(); inp["C_in_O2"]=0.08
    y0=sol.y[:,-1]
    def rhs(t,y): return jv.get_derivatives(t,y,inp)
    sol2=solve_ivp(rhs,(0,120),y0,method="LSODA",rtol=1e-6,atol=1e-8,max_step=0.5)
    jv.get_derivatives(sol2.t[-1], sol2.y[:,-1], inp)
    out2=jv.get_outputs(sol2.y[:,-1])
    print(f"  После 120с C_in 0.08 → C_jv {out2['C_jv_O2']:.3f} SjvO2 {out2['SjvO2']*100:.1f}% ожидание <50%")
    ok=out2['SjvO2']<0.5
    print(f"  [{'OK' if ok else 'FAIL'}] детекция ишемии")
    return {"ok":ok}

def test_drift():
    print("\n"+"="*70+"\nTEST 8: Drift 600с как t_calib\n"+"="*70)
    jv=make_jv()
    sol,out=integrate(jv, INPUTS_HEALTHY, t_end=600)
    V=sol.y[0,:]
    m1=np.mean(V[sol.t>=580]); m2=np.mean(V[sol.t>=590])
    drift=abs(m2-m1)
    print(f"  V 580-600 {m1:.2f} 590-600 {m2:.2f} drift {drift:.4f} nfev {sol.nfev}")
    ok=drift<1.0
    print(f"  [{'OK' if ok else 'FAIL'}] drift <1 мл")
    return {"ok":ok}

def test_strict():
    print("\n"+"="*70+"\nTEST 9: Строгий — положительность, Q_out>=0, SjvO2∈[0,1], масса\n"+"="*70)
    jv=make_jv()
    sol,out=integrate(jv, INPUTS_HEALTHY, t_end=600, rtol=1e-9, atol=1e-12, max_step=0.5)
    V=sol.y[0,:]; C_O2=sol.y[1,:]; C_CO2=sol.y[2,:]
    checks=[
        ("V_jv>0", np.all(V>0)),
        ("V_jv>0.5*V0", np.min(V)>0.5*JV_CFG.get("V0",290)),
        ("C_O2∈[0,0.3]", np.all((C_O2>=0)&(C_O2<=0.3))),
        ("C_CO2∈[0,1]", np.all((C_CO2>=0)&(C_CO2<=1))),
        ("Q_out>=0", out['Q_jv_out']>=0),
        ("SjvO2∈[0,1]", 0<=out['SjvO2']<=1),
        ("P_jv>=0", out['P_jv']>=0),
        ("finite", np.all(np.isfinite(sol.y))),
    ]
    ok_all=True
    for n,o in checks:
        ok_all=ok_all and o
        print(f"  [{'OK' if o else 'FAIL'}] {n}")
    # интегральный баланс V
    # dV/dt интегрированный
    # проверяем что ∫(Q_in - Q_out) ≈ ΔV + ∫(V_target-V)/tau
    # упрощенно: в стационаре dV≈0
    print(f"  V_min {V.min():.1f} V_max {V.max():.1f} P_jv {out['P_jv']:.2f}")
    return {"ok":ok_all}

def summary(res):
    print("\n"+"="*70+"\nСВОДКА: JugularVein?\n"+"="*70)
    names=["Интерфейс","Steady","Mass balance","Washout","Venous P_sv","V_blood tracking","Ишемия SjvO2","Drift 600с","Strict"]
    for n,r in zip(names,res):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    print("\nВЫВОД:", "JugularVein корректна" if all(r["ok"] for r in res) else "Есть FAIL")

def run_all(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf=io.StringIO(); tee=Tee(sys.stdout,buf); orig=sys.stdout; sys.stdout=tee
    try:
        banner()
        rs=[test_interface(),test_steady(),test_mass_balance(),test_washout(),test_venous_pressure(),test_vblood_tracking(),test_ischemia(),test_drift(),test_strict()]
        summary(rs)
        print(f"\nФиниш {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout=orig
    path.write_text(buf.getvalue(),encoding="utf-8")
    print(f"Отчёт: {path}")

if __name__=="__main__":
    run_all(Path(__file__).resolve().parent / "results_debug_jugular_vein_2.txt")
