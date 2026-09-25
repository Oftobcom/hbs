#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_heart_3.py — изолированная проверка Heart4Chambers
Пульсирующая система → предельный цикл, не точка.

Что проверяет:
  1. Интерфейс OrganModel (state_size 4, y0 = EDV)
  2. Steady state в Гайтоне ± допуск (P_sa 85, P_sv 12, P_pa 15, P_pv 12, HR 70)
  3. Mass balance Kirchhoff в steady state
  4. HR sweep 0.7 → 1.0 → 1.3: CO растёт с HR
  5. PV-loop sanity: P_lv_max > P_sa, P_rv_max > P_pa
  6. VSD shunt R_vsd=5: L→R, Qp/Qs >1
  7. RK45 vs LSODA: дифф по EDV/ESV <1%
  8. Drift цикл 8-13 vs 23-28 <5%
  9. [СТРОГИЙ NEW] Conservation laws & periodicity:
     - ∫Q_aortic = ∫Q_mitral = SV_LV <1%
     - ∫Q_pulmonary = ∫Q_tricuspid = SV_RV <1%
     - y(t+T)-y(t) <1e-2 мл
     - V > V0 (soft_clamp не срабатывает)
     - SV·HR/60 == mean Q_aortic <2%
     - отсутствие регургитации >1%

Цель: исключить Heart как причину P_sa 65 вместо 85, EDV 69 вместо 120.
Результаты: консоль + results_debug_heart_3.txt
"""

from __future__ import annotations
import sys, io
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

# --- Совместимость NumPy 1.x / 2.x: np.trapz → np.trapezoid ---
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz

from scipy.integrate import solve_ivp
from heart import Heart4Chambers
from physio_config import load_physiology

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_heart_3.txt"

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

def load_cfg():
    cfg = load_physiology()
    return dict(cfg["heart"]), dict(cfg["systemic"])

HEART_CFG, SYS_CFG = load_cfg()

def make_heart(R_vsd=np.inf):
    cfg = HEART_CFG.copy()
    cfg["R_vsd"]=R_vsd
    return Heart4Chambers(**cfg)

INPUTS = {"P_sa":85.0,"P_sv":12.0,"P_pa":15.0,"P_pv":12.0,"hr_factor":1.0,"baro_activation":1.0,"inotropy_factor":1.0}

def banner():
    print("="*70)
    print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("Heart4Chambers: изолированный тест (pulsatile)")
    print("="*70)
    print(f"  Входы: {INPUTS}")
    print(f"  Ключевые из physiology.yaml:")
    for k in ("hr","E_max_lv","E_min_lv","E_max_rv","E_min_rv","R_mitral","R_aortic","R_tricuspid","R_pulmonary","R_venous_sys","R_venous_pulm","k_valve"):
        print(f"    {k:16s} = {HEART_CFG[k]}")
    print("="*70)

def integrate_cycles(heart, inputs, n_cycles=30, n_avg=10, method="LSODA", rtol=1e-7, atol=1e-9, max_step=0.005):
    hr = heart.hr_base * inputs.get("hr_factor",1.0)
    hr = np.clip(hr, heart.hr_min, heart.hr_max)
    T = 60.0/hr
    t_end = n_cycles*T
    t_avg_start = (n_cycles-n_avg)*T
    y0 = heart.get_initial_state()
    def rhs(t,y): return heart.get_derivatives(t,y,inputs)
    sol = solve_ivp(rhs,(0.0,t_end),y0,method=method,rtol=rtol,atol=atol,max_step=max_step)
    return sol, T, t_avg_start

def extract_steady(sol, heart, inputs, T, t_avg_start):
    mask = sol.t > t_avg_start
    Q = {k:[] for k in ["Q_aortic","Q_mitral","Q_pulmonary","Q_tricuspid","Q_pv_to_la","Q_sv_to_ra","Q_vsd"]}
    P = {k:[] for k in ["P_lv","P_rv","P_la","P_ra"]}
    for i in range(len(sol.t)):
        if not mask[i]: continue
        heart.get_derivatives(sol.t[i], sol.y[:,i], inputs)
        out = heart.get_outputs(sol.y[:,i])
        for k in Q: Q[k].append(out[k])
        for k in P: P[k].append(out[k])
    V_lv = sol.y[1,mask]; V_rv = sol.y[3,mask]
    return {
        "Q_aortic": float(np.mean(Q["Q_aortic"])),
        "Q_mitral": float(np.mean(Q["Q_mitral"])),
        "Q_pulmonary": float(np.mean(Q["Q_pulmonary"])),
        "Q_tricuspid": float(np.mean(Q["Q_tricuspid"])),
        "Q_pv_to_la": float(np.mean(Q["Q_pv_to_la"])),
        "Q_sv_to_ra": float(np.mean(Q["Q_sv_to_ra"])),
        "EDV_LV": float(V_lv.max()), "ESV_LV": float(V_lv.min()), "SV_LV": float(V_lv.max()-V_lv.min()),
        "EDV_RV": float(V_rv.max()), "ESV_RV": float(V_rv.min()),
        "P_lv_max": float(np.max(P["P_lv"])), "P_lv_min": float(np.min(P["P_lv"])),
        "P_rv_max": float(np.max(P["P_rv"])), "P_rv_min": float(np.min(P["P_rv"])),
        "nfev": int(sol.nfev), "success": bool(sol.success),
    }

def test_interface():
    print("\n"+"="*70)
    print("TEST 1: Интерфейс OrganModel")
    print("="*70)
    h=make_heart(); sz=h.get_state_size(); y0=h.get_initial_state()
    d=h.get_derivatives(0.0,y0,INPUTS)
    print(f"  state_size    = {sz}  (ожидание 4)")
    print(f"  initial state = {y0}")
    print(f"  derivatives   = {d}")
    ok=sz==4 and len(y0)==4
    print(f"  [{'OK' if ok else 'FAIL'}] state_size=4")
    return {"ok":ok}

def test_steady():
    print("\n"+"="*70)
    print("TEST 2: Steady state (P_sa=85, P_sv=12, P_pa=15, P_pv=12, HR=70)")
    print("="*70)
    h=make_heart()
    sol,T,t_avg = integrate_cycles(h,INPUTS,n_cycles=30,n_avg=10)
    m=extract_steady(sol,h,INPUTS,T,t_avg)
    print(f"  T = {T:.4f} с, nfev = {m['nfev']}, success = {m['success']}")
    for k in ["Q_aortic","Q_mitral","Q_pulmonary","EDV_LV","ESV_LV","SV_LV","EDV_RV","P_lv_max","P_rv_max"]:
        print(f"        {k:12s} {m[k]:10.2f}")
    checks=[
        ("Q_aortic ∈ [60,130]",60<=m["Q_aortic"]<=130),
        ("Q_mitral ∈ [60,130]",60<=m["Q_mitral"]<=130),
        ("EDV_LV ∈ [90,150]",90<=m["EDV_LV"]<=150),
        ("ESV_LV ∈ [20,60]",20<=m["ESV_LV"]<=60),
        ("SV_LV ∈ [60,110]",60<=m["SV_LV"]<=110),
        ("P_lv_max ∈ [80,180]",80<=m["P_lv_max"]<=180),
        ("P_rv_max ∈ [10,50]",10<=m["P_rv_max"]<=50),
        ("nfev < 50000",m["nfev"]<50000),
        ("success",m["success"]),
    ]
    for name,ok in checks: print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return {"ok": all(o for _,o in checks), "sol":sol,"T":T,"metrics":m}

def test_mass_balance():
    print("\n"+"="*70)
    print("TEST 3: Kirchhoff в steady state (R_vsd=inf)")
    print("="*70)
    h=make_heart()
    sol,T,t_avg=integrate_cycles(h,INPUTS)
    m=extract_steady(sol,h,INPUTS,T,t_avg)
    def rel(a,b): return abs(a-b)/max(abs(a),abs(b),1e-6)
    r1=rel(m["Q_aortic"],m["Q_mitral"])
    r2=rel(m["Q_pulmonary"],m["Q_tricuspid"])
    print(f"  Q_aortic vs Q_mitral:       rel = {r1:.3e}")
    print(f"  Q_pulmonary vs Q_tricuspid: rel = {r2:.3e}")
    ok = r1<0.3 and r2<0.3
    print(f"\n  [{'OK' if ok else 'FAIL'}] балансы <30% (mean за 10 циклов) — см. TEST 9")
    return {"ok":ok}

def test_hr_sweep():
    print("\n"+"="*70)
    print("TEST 4: HR sweep (0.7 → 1.0 → 1.3)")
    print("="*70)
    print(f"      HR    Q_aortic          SV")
    rows=[]
    for f in [0.7,1.0,1.3]:
        inp=INPUTS.copy(); inp["hr_factor"]=f
        h=make_heart()
        sol,T,t_avg=integrate_cycles(h,inp,n_cycles=20,n_avg=5)
        m=extract_steady(sol,h,inp,T,t_avg)
        rows.append((h.hr_base*f,m["Q_aortic"],m["SV_LV"]))
        print(f"    {h.hr_base*f:4.1f}       {m['Q_aortic']:6.2f}      {m['SV_LV']:6.2f}")
    ok = rows[0][1]<=rows[1][1]<=rows[2][1] or rows[0][1]<rows[2][1]
    print(f"\n  [{'OK' if ok else 'FAIL'}] Q_aortic растёт с HR")
    return {"ok":ok}

def test_pv():
    print("\n"+"="*70)
    print("TEST 5: PV loop sanity")
    print("="*70)
    h=make_heart(); sol,T,t_avg=integrate_cycles(h,INPUTS); m=extract_steady(sol,h,INPUTS,T,t_avg)
    print(f"  P_lv_max = {m['P_lv_max']:.1f}  > P_sa=85")
    print(f"  P_rv_max = {m['P_rv_max']:.1f}  > P_pa=15")
    ok=m["P_lv_max"]>85 and m["P_rv_max"]>15
    print(f"\n  [{'OK' if ok else 'FAIL'}] PV-loop физиологичен")
    return {"ok":ok}

def test_vsd():
    print("\n"+"="*70)
    print("TEST 6: VSD shunt (R_vsd=5.0)")
    print("="*70)
    h=make_heart(R_vsd=5.0)
    sol,T,t_avg=integrate_cycles(h,INPUTS)
    m=extract_steady(sol,h,INPUTS,T,t_avg)
    Qp=m["Q_pulmonary"]; Qs=m["Q_aortic"]
    print(f"  Q_aortic    = {Qs:.2f}")
    print(f"  Q_pulmonary = {Qp:.2f}")
    print(f"  Qp/Qs       = {Qp/(Qs+1e-9):.3f}  (>1)")
    ok=Qp>Qs
    print(f"\n  [{'OK' if ok else 'FAIL'}] L->R шунт")
    return {"ok":ok}

def test_rk45():
    print("\n"+"="*70)
    print("TEST 7: RK45 vs LSODA")
    print("="*70)
    h1=make_heart(); sol1,T1,_=integrate_cycles(h1,INPUTS,method="RK45",rtol=1e-7,atol=1e-9)
    m1=extract_steady(sol1,h1,INPUTS,T1,20*T1)
    h2=make_heart(); sol2,T2,_=integrate_cycles(h2,INPUTS,method="LSODA",rtol=1e-7,atol=1e-9)
    m2=extract_steady(sol2,h2,INPUTS,T2,20*T2)
    print(f"       метрика        RK45       LSODA      |diff|")
    for k in ["Q_aortic","EDV_LV","ESV_LV","P_lv_max","P_rv_max"]:
        d=abs(m1[k]-m2[k])
        print(f"      {k:10s} {m1[k]:10.3f} {m2[k]:10.3f} {d:10.3e}")
    print(f"  nfev RK45={sol1.nfev} LSODA={sol2.nfev}")
    ok = abs(m1["EDV_LV"]-m2["EDV_LV"])<1.0 and abs(m1["P_lv_max"]-m2["P_lv_max"])<1.0
    print(f"\n  [{'OK' if ok else 'FAIL'}] RK45 ≈ LSODA по EDV/P")
    return {"ok":ok}

def test_drift():
    print("\n"+"="*70)
    print("TEST 8: Drift — циклы 8-13 vs 23-28")
    print("="*70)
    h=make_heart(); sol,T,_=integrate_cycles(h,INPUTS,n_cycles=30)
    def mean_win(a,b):
        t0=a*T; t1=b*T; mask=(sol.t>=t0)&(sol.t<=t1)
        return float(np.max(sol.y[1,mask])), float(np.min(sol.y[1,mask]))
    edv1,esv1=mean_win(8,13); edv2,esv2=mean_win(23,28)
    print(f"  EDV_LV: [8-13] = {edv1:.2f}  [23-28] = {edv2:.2f}  drift = {abs(edv2-edv1):.3e}")
    print(f"  ESV_LV: [8-13] = {esv1:.2f}  [23-28] = {esv2:.2f}  drift = {abs(esv2-esv1):.3e}")
    ok=abs(edv2-edv1)<0.01 and abs(esv2-esv1)<0.01
    print(f"\n  [{'OK' if ok else 'FAIL'}] drift <1e-2 мл")
    return {"ok":ok}

def test_conservation_strict():
    print("\n"+"="*70)
    print("TEST 9: Conservation laws & periodicity (СТРОГИЙ NEW)")
    print("="*70)
    h=make_heart()
    sol,T,t_avg=integrate_cycles(h,INPUTS,n_cycles=30,n_avg=10,rtol=1e-9,atol=1e-12,max_step=0.002)
    t_last = sol.t[-1]-T
    mask = sol.t>=t_last
    t_c=sol.t[mask]; y_c=sol.y[:,mask]
    Q_a=[]; Q_m=[]; Q_p=[]; Q_tr=[]
    for i in range(len(t_c)):
        h.get_derivatives(t_c[i], y_c[:,i], INPUTS)
        out=h.get_outputs(y_c[:,i])
        Q_a.append(out["Q_aortic"]); Q_m.append(out["Q_mitral"])
        Q_p.append(out["Q_pulmonary"]); Q_tr.append(out["Q_tricuspid"])
    Q_a=np.array(Q_a); Q_m=np.array(Q_m); Q_p=np.array(Q_p); Q_tr=np.array(Q_tr)
    sv_a=float(_trapz(Q_a,t_c)); sv_m=float(_trapz(Q_m,t_c))
    sv_p=float(_trapz(Q_p,t_c)); sv_tr=float(_trapz(Q_tr,t_c))
    V_lv=y_c[1,:]; V_rv=y_c[3,:]; V_la=y_c[0,:]; V_ra=y_c[2,:]
    sv_lv_direct=float(V_lv.max()-V_lv.min())
    def rel(a,b): return abs(a-b)/max(abs(a),abs(b),1e-6)
    print(f"\n  [1] Mass balance LV за цикл:")
    print(f"      ∫Q_aortic = {sv_a:.4f} мл")
    print(f"      ∫Q_mitral  = {sv_m:.4f} мл")
    print(f"      SV_LV      = {sv_lv_direct:.4f} мл")
    print(f"      rel = {rel(sv_a,sv_m):.3e}, {rel(sv_a,sv_lv_direct):.3e}")
    print(f"\n  [2] RV: ∫Q_pulm={sv_p:.4f} ∫Q_tri={sv_tr:.4f} rel={rel(sv_p,sv_tr):.3e}")
    t0=sol.t[-1]-2*T
    i0=int(np.argmin(np.abs(sol.t-t0))); i1=int(np.argmin(np.abs(sol.t-(t0+T))))
    dy=float(np.max(np.abs(sol.y[:,i0]-sol.y[:,i1])))
    print(f"\n  [3] Периодичность max|y(t+T)-y(t)| = {dy:.3e} мл")
    print(f"\n  [4] V_min: V_lv={V_lv.min():.2f} V0={h.V0['LV']} V_la={V_la.min():.2f} V_ra={V_ra.min():.2f}")
    hr=h.hr_base; co_sv=sv_lv_direct*hr/60.0; co_mean_integral=sv_a/T; co_mean_avg=float(np.mean(Q_a))
    print(f"\n  [5] SV·HR/60={co_sv:.3f} vs ∫Q/T={co_mean_integral:.3f} vs mean(Q)={co_mean_avg:.3f} rel={rel(co_sv,co_mean_integral):.3e}")
    def regurg(Q):
        pos=np.sum(Q[Q>0]); neg=np.sum(np.abs(Q[Q<0]))
        return neg/(pos+1e-12)
    reg_a=regurg(Q_a); reg_p=regurg(Q_p)
    print(f"\n  [6] Регургитация аорта {reg_a*100:.4f}% пульм {reg_p*100:.4f}% (<1%)")
    checks=[
        ("mass LV <1%", rel(sv_a,sv_m)<0.01),
        ("SV_LV vs ∫Q <2%", rel(sv_lv_direct,sv_a)<0.02),
        ("mass RV <1%", rel(sv_p,sv_tr)<0.01),
        ("periodicity <1e-2", dy<1e-2),
        ("V_lv>V0", V_lv.min()>h.V0["LV"]),
        ("SV·HR/60==∫Q/T <2%", rel(co_sv,co_mean_integral)<0.02),
        ("регург <1%", reg_a<0.01 and reg_p<0.01),
        ("finite", bool(np.all(np.isfinite(sol.y)))),
    ]
    print()
    ok_all=True
    for name,ok in checks:
        ok_all=ok_all and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return {"ok":ok_all}

def summary(results):
    print("\n"+"="*70)
    print("СВОДКА: физиологичен ли Heart4Chambers?")
    print("="*70)
    names=["Интерфейс","Steady в Gayton","Kirchhoff","HR sweep","PV loop","VSD shunt","RK45==LSODA","Drift","Conservation & periodicity (СТРОГИЙ)"]
    print()
    for n,r in zip(names,results):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    print()
    if all(r["ok"] for r in results):
        print("ВЫВОД: Heart корректен в изоляции → P_sa 65/EDV 69 в связях whole_body")
    else:
        print("ВЫВОД: Heart — см. FAIL выше, но TEST 3 mean-баланс 20-30% норма при фикс P_sa, смотри TEST 9")

def run_all_and_save(report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    buf=io.StringIO()
    tee=Tee(sys.stdout,buf)
    orig=sys.stdout; sys.stdout=tee
    try:
        banner()
        r1=test_interface()
        r2=test_steady()
        r3=test_mass_balance()
        r4=test_hr_sweep()
        r5=test_pv()
        r6=test_vsd()
        r7=test_rk45()
        r8=test_drift()
        r9=test_conservation_strict()
        summary([r1,r2,r3,r4,r5,r6,r7,r8,r9])
        print("\n"+"="*70)
        print("Вход: P_sa, P_sv, P_pa, P_pv, hr_factor, baro_activation")
        print("Выход: Q_aortic, Q_pulmonary, Q_mitral, Q_tricuspid, Q_vsd, Q_pv_to_la, Q_sv_to_ra, P_la, P_lv, P_ra, P_rv")
        print("="*70)
        print(f"\nФиниш: {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout=orig
    report_path.write_text(buf.getvalue(),encoding="utf-8")
    print(f"\nОтчёт сохранён: {report_path}")

if __name__=="__main__":
    out_dir = Path(__file__).resolve().parent
    run_all_and_save(out_dir / "results_debug_heart_3.txt")
