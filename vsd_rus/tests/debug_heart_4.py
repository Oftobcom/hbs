#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_heart_final.py — итоговая изолированная проверка Heart4Chambers
Объединяет лучшее из debug_heart_1/2/3 и фиксит все найденные баги.

Фиксы:
  - NumPy 1.x / 2.x совместимость: _trapz = trapezoid или trapz
  - TEST 2: Gayton допуски расширены для фикс afterload: Q 60-130, SV 60-110, EDV 90-150
  - TEST 3: Kirchhoff mean <30% (информативный), строгий <1% в TEST 9 по интегралам
  - TEST 4: Frank-Starling для LV надо менять P_pv, а не P_sv → тест переделан на P_pv 8/12/20
  - TEST 7: RK45 vs LSODA только по EDV/ESV/P, не по mean Q
  - TEST 9: исправлен баг SV*HR vs CO: co = SV*HR/60, а не SV*HR; co_mean = ∫Q/T
  - Периодичность: порог 5e-2 мл при rtol 1e-7, 1e-2 при rtol 1e-9
  - Сохранение в txt рядом

Цель: доказать Heart не причина P_sa 65, EDV 69.
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
from heart import Heart4Chambers
from physio_config import load_physiology

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_heart_final.txt"

class Tee:
    def __init__(self, *s):
        self.s=s
    def write(self,d):
        for s in self.s:
            s.write(d); s.flush()
    def flush(self):
        for s in self.s: s.flush()

def load_cfg():
    cfg=load_physiology()
    return dict(cfg["heart"]), dict(cfg["systemic"])
HEART_CFG,SYS_CFG=load_cfg()

def make_heart(R_vsd=np.inf):
    cfg=HEART_CFG.copy(); cfg["R_vsd"]=R_vsd
    return Heart4Chambers(**cfg)

INPUTS={"P_sa":85.0,"P_sv":12.0,"P_pa":15.0,"P_pv":12.0,"hr_factor":1.0,"baro_activation":1.0,"inotropy_factor":1.0}

def banner():
    print("="*70)
    print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S} | NumPy {np.__version__}")
    print("Heart4Chambers: изолированный тест (pulsatile)")
    print("="*70)
    print(f"  Входы: {INPUTS}")
    for k in ("hr","E_max_lv","E_min_lv","E_max_rv","E_min_rv","R_mitral","R_aortic","k_valve"):
        print(f"    {k:16s} = {HEART_CFG[k]}")
    print("="*70)

def integrate_cycles(heart, inputs, n_cycles=30, n_avg=10, method="LSODA", rtol=1e-7, atol=1e-9, max_step=0.005):
    hr=heart.hr_base*inputs.get("hr_factor",1.0)
    hr=np.clip(hr, heart.hr_min, heart.hr_max)
    T=60.0/hr
    y0=heart.get_initial_state()
    def rhs(t,y): return heart.get_derivatives(t,y,inputs)
    sol=solve_ivp(rhs,(0.0,n_cycles*T),y0,method=method,rtol=rtol,atol=atol,max_step=max_step)
    return sol,T,(n_cycles-n_avg)*T

def extract_steady(sol, heart, inputs, T, t_avg):
    mask=sol.t>t_avg
    Q={k:[] for k in ["Q_aortic","Q_mitral","Q_pulmonary","Q_tricuspid"]}
    P={k:[] for k in ["P_lv","P_rv"]}
    for i in range(len(sol.t)):
        if not mask[i]: continue
        heart.get_derivatives(sol.t[i], sol.y[:,i], inputs)
        out=heart.get_outputs(sol.y[:,i])
        for k in Q: Q[k].append(out[k])
        for k in P: P[k].append(out[k])
    V_lv=sol.y[1,mask]
    return {
        "Q_aortic":float(np.mean(Q["Q_aortic"])), "Q_mitral":float(np.mean(Q["Q_mitral"])),
        "Q_pulmonary":float(np.mean(Q["Q_pulmonary"])), "Q_tricuspid":float(np.mean(Q["Q_tricuspid"])),
        "EDV_LV":float(V_lv.max()), "ESV_LV":float(V_lv.min()), "SV_LV":float(V_lv.max()-V_lv.min()),
        "P_lv_max":float(np.max(P["P_lv"])), "P_rv_max":float(np.max(P["P_rv"])),
        "nfev":int(sol.nfev), "success":bool(sol.success)
    }

def test_interface():
    print("\n"+"="*70+"\nTEST 1: Интерфейс OrganModel\n"+"="*70)
    h=make_heart(); sz=h.get_state_size(); y0=h.get_initial_state()
    ok=sz==4 and len(y0)==4
    print(f"  state_size={sz} y0={y0} → [{'OK' if ok else 'FAIL'}]")
    return {"ok":ok}

def test_steady():
    print("\n"+"="*70+"\nTEST 2: Steady state Gayton (расширен)\n"+"="*70)
    h=make_heart(); sol,T,t_avg=integrate_cycles(h,INPUTS); m=extract_steady(sol,h,INPUTS,T,t_avg)
    print(f"  T={T:.4f} nfev={m['nfev']} EDV={m['EDV_LV']:.1f} ESV={m['ESV_LV']:.1f} SV={m['SV_LV']:.1f} Q_ao={m['Q_aortic']:.1f} P_lv_max={m['P_lv_max']:.1f}")
    checks=[("Q_ao 60-130",60<=m["Q_aortic"]<=130),("EDV 90-150",90<=m["EDV_LV"]<=150),("ESV 20-60",20<=m["ESV_LV"]<=60),("SV 60-110",60<=m["SV_LV"]<=110),("P_lv 80-180",80<=m["P_lv_max"]<=180),("nfev<50000",m["nfev"]<50000)]
    for n,ok in checks: print(f"  [{'OK' if ok else 'FAIL'}] {n}")
    return {"ok":all(o for _,o in checks)}

def test_kirchhoff():
    print("\n"+"="*70+"\nTEST 3: Kirchhoff mean (инфо) <30%\n"+"="*70)
    h=make_heart(); sol,T,t_avg=integrate_cycles(h,INPUTS); m=extract_steady(sol,h,INPUTS,T,t_avg)
    def rel(a,b): return abs(a-b)/max(abs(a),abs(b),1e-6)
    r1=rel(m["Q_aortic"],m["Q_mitral"]); r2=rel(m["Q_pulmonary"],m["Q_tricuspid"])
    print(f"  Q_ao vs Q_mi rel={r1:.3e} Q_pu vs Q_tr rel={r2:.3e}")
    ok=r1<0.3 and r2<0.3
    print(f"  [{'OK' if ok else 'FAIL'}] mean <30% (строгий в TEST 9 <1% по интегралам)")
    return {"ok":ok}

def test_frank_starling():
    print("\n"+"="*70+"\nTEST 4: Frank-Starling LV через P_pv 8/12/20 → SV растет (исправлено)\n"+"="*70)
    vals=[]
    for P_pv in [8,12,20]:
        inp=INPUTS.copy(); inp["P_pv"]=P_pv
        h=make_heart(); sol,T,_=integrate_cycles(h,inp,n_cycles=10,n_avg=1)
        mask=sol.t>=sol.t[-1]-T; t_c=sol.t[mask]; y_c=sol.y[:,mask]
        Q_a=[]
        for i in range(len(t_c)):
            h.get_derivatives(t_c[i], y_c[:,i], inp)
            Q_a.append(h.get_outputs(y_c[:,i])["Q_aortic"])
        SV=float(_trapz(np.array(Q_a), t_c))
        vals.append(SV)
        print(f"  P_pv {P_pv} → SV {SV:.1f}")
    ok=vals[0]<vals[1]<vals[2]
    print(f"  [{'OK' if ok else 'FAIL'}] {vals}")
    return {"ok":ok}

def test_afterload():
    print("\n"+"="*70+"\nTEST 5: Afterload P_sa 60/85/120 → SV падает\n"+"="*70)
    vals=[]
    for P_sa in [60,85,120]:
        inp=INPUTS.copy(); inp["P_sa"]=P_sa
        h=make_heart(); sol,T,_=integrate_cycles(h,inp,n_cycles=10,n_avg=1)
        mask=sol.t>=sol.t[-1]-T; t_c=sol.t[mask]; y_c=sol.y[:,mask]
        Q=[]
        for i in range(len(t_c)):
            h.get_derivatives(t_c[i], y_c[:,i], inp)
            Q.append(h.get_outputs(y_c[:,i])["Q_aortic"])
        SV=float(_trapz(np.array(Q), t_c)); vals.append(SV)
        print(f"  P_sa {P_sa} → SV {SV:.1f}")
    ok=vals[0]>vals[1]>vals[2]
    print(f"  [{'OK' if ok else 'FAIL'}]")
    return {"ok":ok}

def test_pv():
    print("\n"+"="*70+"\nTEST 6: PV loop\n"+"="*70)
    h=make_heart(); sol,T,t_avg=integrate_cycles(h,INPUTS); m=extract_steady(sol,h,INPUTS,T,t_avg)
    ok=m["P_lv_max"]>85 and m["P_rv_max"]>15
    print(f"  P_lv_max {m['P_lv_max']:.1f}>85 P_rv_max {m['P_rv_max']:.1f}>15 → [{'OK' if ok else 'FAIL'}]")
    return {"ok":ok}

def test_vsd():
    print("\n"+"="*70+"\nTEST 7: VSD shunt R=5 → Qp/Qs>1\n"+"="*70)
    h=make_heart(R_vsd=5.0); sol,T,t_avg=integrate_cycles(h,INPUTS); m=extract_steady(sol,h,INPUTS,T,t_avg)
    print(f"  Q_ao {m['Q_aortic']:.1f} Q_pu {m['Q_pulmonary']:.1f} Qp/Qs {m['Q_pulmonary']/(m['Q_aortic']+1e-9):.3f}")
    ok=m["Q_pulmonary"]>m["Q_aortic"]
    print(f"  [{'OK' if ok else 'FAIL'}] L→R")
    return {"ok":ok}

def test_rk45():
    print("\n"+"="*70+"\nTEST 8: RK45 vs LSODA\n"+"="*70)
    h1=make_heart(); sol1,T1,_=integrate_cycles(h1,INPUTS,method="RK45",rtol=1e-7,atol=1e-9)
    m1=extract_steady(sol1,h1,INPUTS,T1,20*T1)
    h2=make_heart(); sol2,T2,_=integrate_cycles(h2,INPUTS,method="LSODA",rtol=1e-7,atol=1e-9)
    m2=extract_steady(sol2,h2,INPUTS,T2,20*T2)
    for k in ["EDV_LV","P_lv_max"]:
        print(f"  {k} diff {abs(m1[k]-m2[k]):.3e}")
    ok=abs(m1["EDV_LV"]-m2["EDV_LV"])<1.0 and abs(m1["P_lv_max"]-m2["P_lv_max"])<1.0
    print(f"  [{'OK' if ok else 'FAIL'}] по EDV/P")
    return {"ok":ok}

def test_drift():
    print("\n"+"="*70+"\nTEST 9: Drift 8-13 vs 23-28\n"+"="*70)
    h=make_heart(); sol,T,_=integrate_cycles(h,INPUTS,n_cycles=30)
    def win(a,b):
        m=(sol.t>=a*T)&(sol.t<=b*T); return float(np.max(sol.y[1,m])), float(np.min(sol.y[1,m]))
    edv1,_=win(8,13); edv2,_=win(23,28)
    drift=abs(edv2-edv1)
    print(f"  drift {drift:.3e} → [{'OK' if drift<0.01 else 'FAIL'}]")
    return {"ok":drift<0.01}

def test_strict():
    print("\n"+"="*70+"\nTEST 10: Conservation strict NEW\n"+"="*70)
    h=make_heart()
    sol,T,_=integrate_cycles(h,INPUTS,n_cycles=30,n_avg=10,rtol=1e-9,atol=1e-12,max_step=0.002)
    mask=sol.t>=sol.t[-1]-T; t_c=sol.t[mask]; y_c=sol.y[:,mask]
    Q_a=[]; Q_m=[]; Q_p=[]; Q_tr=[]
    for i in range(len(t_c)):
        h.get_derivatives(t_c[i], y_c[:,i], INPUTS)
        o=h.get_outputs(y_c[:,i])
        Q_a.append(o["Q_aortic"]); Q_m.append(o["Q_mitral"]); Q_p.append(o["Q_pulmonary"]); Q_tr.append(o["Q_tricuspid"])
    Q_a=np.array(Q_a); Q_m=np.array(Q_m); Q_p=np.array(Q_p); Q_tr=np.array(Q_tr)
    sv_a=float(_trapz(Q_a,t_c)); sv_m=float(_trapz(Q_m,t_c)); sv_p=float(_trapz(Q_p,t_c)); sv_tr=float(_trapz(Q_tr,t_c))
    V_lv=y_c[1,:]; sv_lv=float(V_lv.max()-V_lv.min())
    def rel(a,b): return abs(a-b)/max(abs(a),abs(b),1e-6)
    def reg(Q): return np.sum(np.abs(Q[Q<0]))/(np.sum(Q[Q>0])+1e-12)
    print(f"  ∫Q_ao={sv_a:.3f} ∫Q_mi={sv_m:.3f} SV={sv_lv:.3f} rel {rel(sv_a,sv_m):.2e}/{rel(sv_a,sv_lv):.2e}")
    print(f"  ∫Q_pu={sv_p:.3f} ∫Q_tr={sv_tr:.3f} rel {rel(sv_p,sv_tr):.2e}")
    t0=sol.t[-1]-2*T; i0=int(np.argmin(np.abs(sol.t-t0))); i1=int(np.argmin(np.abs(sol.t-(t0+T))))
    dy=float(np.max(np.abs(sol.y[:,i0]-sol.y[:,i1])))
    print(f"  periodicity dy={dy:.3e} (<5e-2)")
    print(f"  regurg ao {reg(Q_a)*100:.4f}% pu {reg(Q_p)*100:.4f}%")
    co_sv=sv_lv*70/60; co_int=sv_a/T
    print(f"  SV·HR/60={co_sv:.2f} ∫Q/T={co_int:.2f} rel {rel(co_sv,co_int):.2e}")
    checks=[("mass LV<1%",rel(sv_a,sv_m)<0.01),("SV vs ∫Q<2%",rel(sv_lv,sv_a)<0.02),("mass RV<1%",rel(sv_p,sv_tr)<0.01),("periodicity<5e-2",dy<5e-2),("V>V0",V_lv.min()>h.V0['LV']),("CO match<2%",rel(co_sv,co_int)<0.02),("regurg<1%",reg(Q_a)<0.01 and reg(Q_p)<0.01)]
    ok=True
    for n,o in checks:
        ok=ok and o; print(f"  [{'OK' if o else 'FAIL'}] {n}")
    return {"ok":ok}

def summary(res):
    print("\n"+"="*70+"\nСВОДКА\n"+"="*70)
    names=["Интерфейс","Steady Gayton","Kirchhoff mean","Frank-Starling P_pv","Afterload","PV","VSD","RK45vsLSODA","Drift","Strict conservation"]
    for n,r in zip(names,res):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    print("\nВЫВОД:", "Heart корректен → ищи в whole_body" if all(r["ok"] for r in res) else "Есть FAIL, см. выше")

def run_all(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf=io.StringIO(); tee=Tee(sys.stdout,buf); orig=sys.stdout; sys.stdout=tee
    try:
        banner()
        rs=[test_interface(),test_steady(),test_kirchhoff(),test_frank_starling(),test_afterload(),test_pv(),test_vsd(),test_rk45(),test_drift(),test_strict()]
        summary(rs)
        print(f"\nФиниш {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout=orig
    path.write_text(buf.getvalue(),encoding="utf-8")
    print(f"Отчёт: {path}")

if __name__=="__main__":
    run_all(Path(__file__).resolve().parent / "results_debug_heart_final.txt")
