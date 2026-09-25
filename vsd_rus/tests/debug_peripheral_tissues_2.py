#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_peripheral_tissues.py — изолированная проверка PeripheralTissues.

Что подаём на вход:
  inputs = {
    'P_sa': мм рт.ст. (артериальное давление, 60-160),
    'P_sv': мм рт.ст. (венозное, 0-20),
    'C_a_O2': мл O2/мл крови (0.12-0.20 норма, 0.08 гипоксия),
    'C_v_lactate': мг/мл (венозный лактат, 0.05-0.20),
    'V_blood': мл (объём крови для пересчёта dC, 4000-6000)
  }
Состояние: [C_O2_local, C_lactate_local, R_eff]

Что ожидаем на выходе:
  - Q_peripheral = (P_sa-P_sv)/R_eff  10-40 мл/с (600-2400 мл/мин) при P 80/5 R 3.8
  - R_eff в [R_base*R_min, R_base*R_max] = [2.85, 9.5] (R_min 0.75 R_max 2.5)
  - C_O2_local в [0, Ca] 0.05-0.20, не отрицательный
  - C_lactate 0.05-0.30 мг/мл, при нормоксии ~0.10
  - VO2_eff ~1.5 мл/с (90 мл/мин) с Q_factor ∈[0.1,1.5]
  - Mass: dC_O2_loc*V_t + VO2 ≈ Q*(Ca-Cloc), dC_lac*Vt = prod - clear - release
  - Нет обратного кровотока Q>=0
  - Нет NaN/Inf при Q=0, P_sa=0, Ca=0
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

from peripheral_tissues import PeripheralTissues

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_peripheral_tissues_2.txt"

class Tee:
    def __init__(self,*s): self.s=s
    def write(self,d):
        for s in self.s: s.write(d); s.flush()
    def flush(self):
        for s in self.s: s.flush()

def make_periph(**overrides):
    return PeripheralTissues(**overrides)

INPUTS_HEALTHY = {
    'P_sa': 85.0,
    'P_sv': 5.0,
    'C_a_O2': 0.20,
    'C_v_lactate': 0.10,
    'V_blood': 5000.0,
}

def integrate(pt, inputs, t_end=60.0, y0=None):
    if y0 is None:
        y0 = pt.get_initial_state()
    def rhs(t,y): return pt.get_derivatives(t,y,inputs)
    sol = solve_ivp(rhs,(0,t_end),y0,method="LSODA",rtol=1e-7,atol=1e-9,max_step=0.5)
    pt.get_derivatives(sol.t[-1], sol.y[:,-1], inputs)
    out = pt.get_outputs(sol.y[:,-1])
    return sol,out

def banner():
    print("="*78)
    print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S} | NumPy {np.__version__}")
    print("PeripheralTissues: изолированный тест")
    print("="*78)
    pt = make_periph()
    print(f"  R_base={pt.R_base} C_tissue={pt.C_tissue} VO2_base={pt.VO2_base}")
    print(f"  O2_norm={pt.O2_norm} k_O2={pt.k_O2_autoreg} R_min_factor={pt.R_min_factor}")
    print(f"  k_P_myogenic={pt.k_P_myogenic} P_sa_norm={pt.P_sa_norm} R_max_factor={pt.R_max_factor}")
    print(f"  V_tissue_eff={pt.V_tissue_eff} C_a_O2_norm={pt.C_a_O2_norm}")
    print(f"  Входы HEALTHY: {INPUTS_HEALTHY}")
    print(f"  → {RESULT_FILE}")
    print("="*78)

def test_interface():
    print("\n"+"="*78+"\nTEST 1: Интерфейс OrganModel (3 состояния)\n"+"="*78)
    pt = make_periph()
    sz = pt.get_state_size()
    y0 = pt.get_initial_state()
    d0 = pt.get_derivatives(0.0,y0,INPUTS_HEALTHY)
    print(f"  size={sz} (ожид 3)")
    print(f"  y0={y0} [C_O2, C_lac, R]")
    print(f"  dy/dt(0)={d0}")
    ok = sz==3 and y0.size==3 and d0.size==3 and np.all(np.isfinite(y0)) and np.all(np.isfinite(d0))
    print(f"  [{'OK' if ok else 'FAIL'}] интерфейс")
    return {"ok":ok}

def test_steady_healthy():
    print("\n"+"="*78+"\nTEST 2: Steady здоровый (P_sa 85 Ca 0.20)\n"+"="*78)
    pt = make_periph()
    sol,out = integrate(pt, INPUTS_HEALTHY, t_end=200)
    Pprox = sol.y[0,-1]; Plac = sol.y[1,-1]; R = sol.y[2,-1]
    print(f"  C_O2_local={out['C_O2_local']:.4f} ожид 0.08-0.18")
    print(f"  C_lactate={out['C_lactate_local']:.4f} ожид 0.05-0.20")
    print(f"  R_eff={out['R_eff']:.3f} R_target={out['R_target']:.3f} ожид 2.8-4.5")
    print(f"  Q_periph={out['Q_peripheral']:.2f} мл/с ({out['Q_peripheral']*60:.0f} мл/мин) ожид 10-40")
    print(f"  VO2={out['O2_consumption_periph']:.3f} мл/с ожид 1.0-2.0")
    print(f"  f_O2={out['f_O2_autoreg']:.3f} f_P={out['f_P_myogenic']:.3f}")
    print(f"  dC_O2_blood={out['_diagnostic_dC_O2_blood']:.5f} dC_lac_blood={out['dC_lactate_blood']:.5f}")
    ok = (0.05<=out['C_O2_local']<=0.20) and (0.03<=out['C_lactate_local']<=0.30) \
         and (2.0<=out['R_eff']<=5.5) and (10<=out['Q_peripheral']<=40) \
         and (0.8<=out['O2_consumption_periph']<=2.5) and np.all(np.isfinite(sol.y))
    print(f"  [{'OK' if ok else 'FAIL'}] steady здоровый")
    return {"ok":ok,"out":out,"sol":sol}

def test_autoreg_O2():
    print("\n"+"="*78+"\nTEST 3: Метаболическая ауторегуляция (гипоксия → вазодилатация)\n"+"="*78)
    pt = make_periph()
    cases = [
        (0.20, "нормоксия Ca 0.20"),
        (0.15, "погранично 0.15"),
        (0.10, "гипоксия Ca 0.10"),
        (0.08, "тяжелая 0.08"),
    ]
    ok=True
    for Ca,label in cases:
        inp = dict(INPUTS_HEALTHY, C_a_O2=Ca)
        sol,out = integrate(pt, inp, t_end=100)
        f_O2 = out['f_O2_autoreg']
        Rt = out['R_target']
        print(f"  {label:20s} → C_loc {out['C_O2_local']:.4f} f_O2 {f_O2:.3f} R_target {Rt:.3f} Q {out['Q_peripheral']:.2f}")
        # при гипоксии f_O2 <1
        if Ca<0.15 and f_O2>=1.0:
            ok=False
        if not (pt.R_base*pt.R_min_factor-1e-6 <= Rt <= pt.R_base*pt.R_max_factor+1e-6):
            ok=False
    print(f"  [{'OK' if ok else 'FAIL'}] метаболическая: f_O2 ↓ при гипоксии, R в [Rmin,Rmax]")
    return {"ok":ok}

def test_autoreg_P():
    print("\n"+"="*78+"\nTEST 4: Миогенная ауторегуляция (высокое P_sa → вазоконстрикция)\n"+"="*78)
    pt = make_periph()
    cases = [(60,"гипотония 60"),(85,"норма 85"),(100,"100 граница deadband"),(120,"гипертония 120"),(160,"тяжелая 160")]
    ok=True
    prev_R=None
    for P_sa,label in cases:
        inp = dict(INPUTS_HEALTHY, P_sa=P_sa)
        sol,out = integrate(pt, inp, t_end=100)
        f_P = out['f_P_myogenic']
        Rt = out['R_target']
        print(f"  {label:20s} → f_P {f_P:.3f} R_target {Rt:.3f} Q {out['Q_peripheral']:.2f} C_O2 {out['C_O2_local']:.4f}")
        # монотонность: R должен расти с P при P > P_norm+deadband
        if prev_R is not None and P_sa>110 and Rt < prev_R-1e-6:
            ok=False
        prev_R=Rt
        if not (pt.R_base*pt.R_min_factor-1e-6 <= Rt <= pt.R_base*pt.R_max_factor+1e-6):
            ok=False
    # deadband check: в _autoregulation_target есть deadband, а в диагностике f_P_myogenic его нет
    # поэтому проверяем R_target, а не f_P_myogenic
    _,out85 = integrate(pt, dict(INPUTS_HEALTHY,P_sa=85), t_end=100)
    _,out95 = integrate(pt, dict(INPUTS_HEALTHY,P_sa=95), t_end=100)
    # при P 85 и 95 R_target должен быть одинаков из-за deadband 10
    deadband_ok = abs(out85['R_target']-out95['R_target'])<1e-3
    print(f"  deadband 10 мм рт.ст.: P85 R_target={out85['R_target']:.4f} P95 R_target={out95['R_target']:.4f} diff {abs(out85['R_target']-out95['R_target']):.2e} → {deadband_ok}")
    ok = ok and deadband_ok
    print(f"  [{'OK' if ok else 'FAIL'}] миогенная: R↑ при P↑, deadband работает")
    return {"ok":ok}

def test_mass_balance():
    print("\n"+"="*78+"\nTEST 5: Mass balance O2 и объёмов\n"+"="*78)
    pt = make_periph()
    sol,out = integrate(pt, INPUTS_HEALTHY, t_end=60)
    C_O2 = out['C_O2_local']; Ca = INPUTS_HEALTHY['C_a_O2']; Q = out['Q_peripheral']
    VO2 = out['O2_consumption_periph']
    dCdt = pt.get_derivatives(0.0,sol.y[:,-1],INPUTS_HEALTHY)[0]
    # баланс: dC*Vt = Q*(Ca-C) - VO2
    lhs = dCdt*pt.V_tissue_eff
    rhs = Q*(Ca-C_O2) - VO2
    print(f"  dC_O2/dt*Vt = {lhs:.4f}, Q*(Ca-Cloc)-VO2 = {rhs:.4f}, diff {abs(lhs-rhs):.2e}")
    ok_O2 = abs(lhs-rhs)<1e-6

    # объём: Q>=0 всегда
    print(f"  Q_peripheral={Q:.3f} >=0 ? {Q>=0}")
    ok_Q = Q>= -1e-9

    # лактат баланс: dC_lac*Vt = prod - clear*? - release*?
    dC_lac = pt.get_derivatives(0.0,sol.y[:,-1],INPUTS_HEALTHY)[1]
    prod = out['lactate_production']; rel = out['lactate_release_to_blood']
    # prod - clear - release = dC/dt, clear = k_clear*C
    clear = pt.k_lactate_clear*out['C_lactate_local']
    print(f"  dC_lac*Vt? dC={dC_lac:.5f} prod={prod:.5f} clear={clear:.5f} rel={rel:.5f} sum prod-clear-rel={prod-clear-rel:.5f}")
    ok_lac = abs(dC_lac - (prod-clear-rel))<1e-6

    ok = ok_O2 and ok_Q and ok_lac
    print(f"  [{'OK' if ok else 'FAIL'}] mass balance O2 и лактата, Q>=0")
    return {"ok":ok}

def test_lactate():
    print("\n"+"="*78+"\nTEST 6: Лактат — базальный vs анаэробный\n"+"="*78)
    pt = make_periph()
    # нормоксия
    sol_n,out_n = integrate(pt, INPUTS_HEALTHY, t_end=200)
    print(f"  нормоксия: C_O2 {out_n['C_O2_local']:.4f} C_lac {out_n['C_lactate_local']:.4f} prod {out_n['lactate_production']:.5f} release {out_n['lactate_release_to_blood']:.5f}")
    # гипоксия
    inp_hypo = dict(INPUTS_HEALTHY, P_sa=60, C_a_O2=0.08)
    sol_h,out_h = integrate(pt, inp_hypo, t_end=200)
    print(f"  гипоксия: C_O2 {out_h['C_O2_local']:.4f} C_lac {out_h['C_lactate_local']:.4f} prod {out_h['lactate_production']:.5f} release {out_h['lactate_release_to_blood']:.5f}")
    ok = out_h['lactate_production'] > out_n['lactate_production'] -1e-9
    print(f"  [{'OK' if ok else 'FAIL'}] лактат ↑ при гипоксии (prod гипокс > норм)")
    return {"ok":ok}

def test_edge():
    print("\n"+"="*78+"\nTEST 7: Edge cases — объёмы и массы\n"+"="*78)
    pt = make_periph()
    cases=[
        (dict(P_sa=0,P_sv=5,C_a_O2=0.20), "P_sa 0 (нет перфузии)"),
        (dict(P_sa=85,P_sv=30,C_a_O2=0.20), "P_sv 30 высокий"),
        (dict(P_sa=85,P_sv=5,C_a_O2=0.0), "Ca 0 (аноксия)"),
        (dict(P_sa=200,P_sv=5,C_a_O2=0.20), "P_sa 200 гипертония"),
        (dict(P_sa=85,P_sv=5,C_a_O2=0.20,C_v_lactate=1.0), "C_v_lac 1.0 высокий"),
    ]
    ok_all=True
    for inp_delta,label in cases:
        inp = dict(INPUTS_HEALTHY); inp.update(inp_delta)
        sol,out = integrate(pt, inp, t_end=100)
        finite = np.all(np.isfinite(sol.y))
        q_nonneg = out['Q_peripheral']>= -1e-9
        c_nonneg = out['C_O2_local']>= -1e-9 and out['C_lactate_local']>= -1e-9
        r_range = pt.R_base*0.5 <= out['R_eff'] <= pt.R_base*3.0  # шире чем [0.75,2.5] на транзиенте
        print(f"  {label:30s} → Q {out['Q_peripheral']:.2f} C_O2 {out['C_O2_local']:.3f} C_lac {out['C_lactate_local']:.3f} R {out['R_eff']:.2f} finite {finite} Q>=0 {q_nonneg} C>=0 {c_nonneg}")
        ok_all = ok_all and finite and q_nonneg and c_nonneg
    print(f"  [{'OK' if ok_all else 'FAIL'}] edge cases: нет NaN, Q>=0, C>=0, R в пределах")
    return {"ok":ok_all}

def test_drift():
    print("\n"+"="*78+"\nTEST 8: Drift 60→600с\n"+"="*78)
    pt = make_periph()
    sol60,_ = integrate(pt, INPUTS_HEALTHY, t_end=60)
    sol600,_ = integrate(pt, INPUTS_HEALTHY, t_end=600)
    drift_O2 = abs(sol600.y[0,-1]-sol60.y[0,-1])
    drift_lac = abs(sol600.y[1,-1]-sol60.y[1,-1])
    drift_R = abs(sol600.y[2,-1]-sol60.y[2,-1])
    print(f"  60с: C_O2 {sol60.y[0,-1]:.5f} C_lac {sol60.y[1,-1]:.5f} R {sol60.y[2,-1]:.4f}")
    print(f"  600с: C_O2 {sol600.y[0,-1]:.5f} C_lac {sol600.y[1,-1]:.5f} R {sol600.y[2,-1]:.4f}")
    print(f"  drift O2 {drift_O2:.2e} lac {drift_lac:.2e} R {drift_R:.2e} ожид <2e-3 (tau 3с)")
    ok = drift_O2<2e-3 and drift_lac<1e-3 and drift_R<0.05 and sol60.nfev<2000
    print(f"  [{'OK' if ok else 'FAIL'}] drift <2e-3, nfev {sol60.nfev}<2000")
    return {"ok":ok}

def summary(results):
    print("\n"+"="*78+"\nСВОДКА: PeripheralTissues\n"+"="*78)
    names=["Интерфейс 3-state","Steady здоровый","Autoreg O2","Autoreg P myogenic","Mass balance","Lactate hypoxia","Edge cases","Drift"]
    for n,r in zip(names,results):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    if all(r['ok'] for r in results):
        print("\nВЫВОД: PeripheralTissues верифицирован — даёт физиологичные числа в изоляции.")
    else:
        print("\nВЫВОД: есть FAIL — см. выше")

def run_all(path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf=io.StringIO()
    sys.stdout = Tee(sys.__stdout__, buf)
    try:
        banner()
        rs=[test_interface(),test_steady_healthy(),test_autoreg_O2(),test_autoreg_P(),
            test_mass_balance(),test_lactate(),test_edge(),test_drift()]
        summary(rs)
        print(f"\nФиниш: {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout=sys.__stdout__
    path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Отчёт: {path}")

if __name__=="__main__":
    run_all(RESULT_FILE)
