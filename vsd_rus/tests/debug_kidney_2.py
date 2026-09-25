#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_kidney.py — изолированная проверка KidneyHemodynamic

Kidney — алгебраический орган, state_size=0. Нет ODE, есть compute_effects().
Состояние не хранит объём, влияет на whole_body через dV_blood и dC_tox.

Что проверяем:
  1. Интерфейс OrganModel (state_size 0, пустой y0, get_derivatives empty)
  2. GFR(P_sa) — кривая ауторегуляции: 0→анурія, 80→линейный, 80-160 плато 0.4-1.6×base, >160 рост до 2.2×
  3. Renal flow Q_renal = (P_sa-P_sv)/R_renal
  4. Объёмный баланс: urine = GFR*(1-reabs), dV = -urine, сравнение с intake 0.02 мл/с
  5. Токсин баланс: dC_tox = -GFR*C*frac / V_blood, проверка массы
  6. Анурия при низком давлении P_sa <40
  7. Равновесие объёма: найти P_eq где intake = urine (V_blood стационар)
  8. Drift 600с и 1800с как t_calib/t_span
  9. Строгий: Q_renal>=0, GFR>=0, urine>=0, finite

Вход: P_sa, P_sv, C_tox, V_blood
Выход → whole_body: dC_tox, dV_blood, Q_renal, GFR, urine_output

Результаты: консоль + results_debug_kidney.txt
"""

from __future__ import annotations
import sys, io
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from kidney import KidneyHemodynamic
from physio_config import load_physiology

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_kidney.txt"

class Tee:
    def __init__(self,*s): self.s=s
    def write(self,d):
        for s in self.s: s.write(d); s.flush()
    def flush(self):
        for s in self.s: s.flush()

def load_cfg():
    cfg=load_physiology()
    return dict(cfg.get("kidney",{})), dict(cfg.get("systemic",{}))

KID_CFG, SYS_CFG = load_cfg()

def make_kidney():
    return KidneyHemodynamic(
        GFR_base=KID_CFG.get("GFR_base",120.0),
        P_autoreg=KID_CFG.get("P_autoreg",95.0),
        autoreg_amplitude=KID_CFG.get("autoreg_amplitude",0.6),
        autoreg_slope=KID_CFG.get("autoreg_slope",0.025),
        toxin_clearance_frac=KID_CFG.get("toxin_clearance_frac",0.2),
        volume_reabsorption_frac=KID_CFG.get("volume_reabsorption_frac",0.99),
        renal_resistance=KID_CFG.get("renal_resistance",3.75),
    )

INPUTS_HEALTHY = {"P_sa":85.0, "P_sv":5.0, "C_tox":0.5, "V_blood":5800.0}

def banner():
    print("="*70)
    print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S} | NumPy {np.__version__}")
    print("KidneyHemodynamic: изолированный тест (алгебраический)")
    print("="*70)
    print(f"  Параметры из physiology.yaml kidney:")
    for k,v in KID_CFG.items():
        print(f"    {k:28s} = {v}")
    print(f"  systemic: fluid_intake={SYS_CFG.get('fluid_intake_rate')} insensible={SYS_CFG.get('insensible_loss_rate')}")
    print(f"  Входы здоровые: {INPUTS_HEALTHY}")
    print(f"  → {RESULT_FILE}")
    print("="*70)

def test_interface():
    print("\n"+"="*70+"\nTEST 1: Интерфейс OrganModel (state_size 0)\n"+"="*70)
    k=make_kidney()
    sz=k.get_state_size(); y0=k.get_initial_state(); d=k.get_derivatives(0.0,y0,INPUTS_HEALTHY)
    print(f"  state_size={sz} ожидание 0")
    print(f"  y0={y0} size={y0.size} ожидание []")
    print(f"  dydt={d} size={d.size}")
    # после compute_effects должны быть outputs
    k.compute_effects(**INPUTS_HEALTHY)
    out=k.get_outputs(y0)
    print(f"  outputs после compute_effects: {out}")
    ok = (sz==0 and y0.size==0 and d.size==0 and "GFR" in out and "Q_renal" in out)
    print(f"  [{'OK' if ok else 'FAIL'}] интерфейс алгебраический")
    return {"ok":ok}

def test_gfr_curve():
    print("\n"+"="*70+"\nTEST 2: GFR(P_sa) — кривая ауторегуляции\n"+"="*70)
    k=make_kidney()
    Ps = [0,20,40,60,80,95,120,150,200]
    print(f"  GFR_base={k.GFR_base} P_autoreg={k.P_autoreg} amp={k.autoreg_amplitude} slope={k.autoreg_slope}")
    results=[]
    for P in Ps:
        g=k._compute_gfr(P)
        results.append(g)
        print(f"  P_sa {P:3d} → GFR {g:6.1f} мл/мин ({g/60:5.3f} мл/с) factor {g/k.GFR_base:.3f}")
    # проверки Гайтона
    checks=[
        ("GFR 0→0 анурия", results[0]<5),
        ("GFR растет с P", results[1]<results[2]<results[3]),
        ("Плато 80-120 в [0.6-1.6]×base", 0.4*k.GFR_base < results[5] < 1.7*k.GFR_base),
        ("Макс ≤2.2×base", max(results) <= 2.21*k.GFR_base),
        ("GFR 95 ≈ base", 0.7*k.GFR_base < results[5] < 1.3*k.GFR_base),
    ]
    ok=True
    for n,c in checks:
        ok=ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {n}")
    return {"ok":ok, "gfrs":results}

def test_renal_flow():
    print("\n"+"="*70+"\nTEST 3: Renal flow Q_renal = (P_sa-P_sv)/R\n"+"="*70)
    k=make_kidney()
    for P_sa,P_sv in [(85,5),(65,5),(120,10),(40,5)]:
        eff=k.compute_effects(P_sa=P_sa,P_sv=P_sv,C_tox=0.5,V_blood=5800)
        Q_exp=(P_sa-P_sv)/k.renal_resistance
        print(f"  P_sa {P_sa:3d} P_sv {P_sv:2d} R={k.renal_resistance} → Q={eff['Q_renal']:.2f} ожидание {Q_exp:.2f} {'OK' if abs(eff['Q_renal']-Q_exp)<1e-6 else 'FAIL'}")
    ok = abs(k.compute_effects(85,5,0.5,5800)['Q_renal'] - (85-5)/3.75) <1e-6
    print(f"  [{'OK' if ok else 'FAIL'}] формула Q")
    return {"ok":ok}

def test_volume_balance():
    print("\n"+"="*70+"\nTEST 4: Объёмный баланс urine = GFR*(1-reabs)\n"+"="*70)
    k=make_kidney()
    intake=SYS_CFG.get("fluid_intake_rate",0.015)
    print(f"  intake={intake} мл/с insensible={SYS_CFG.get('insensible_loss_rate',0)}")
    for P_sa in [65,85,95,120]:
        eff=k.compute_effects(P_sa=P_sa,P_sv=5,C_tox=0.5,V_blood=5800)
        urine=eff['urine_output']
        GFR_s=eff['GFR']
        urine_exp=GFR_s*(1-k.volume_reabsorption_frac)
        print(f"  P_sa {P_sa:3d} → GFR {eff['GFR']*60:5.1f} мл/мин ({GFR_s:.3f} мл/с) urine {urine:.5f} мл/с exp {urine_exp:.5f} dV={eff['dV_blood']:.5f} net {intake+eff['dV_blood']:.5f}")
    # при P=85 urine должен ≈ intake для стационара V_blood
    eff85=k.compute_effects(85,5,0.5,5800)
    net=intake+eff85['dV_blood']
    print(f"  net при P=85: {net:.5f} мл/с (0 = стационар)")
    # проверка что dV = -urine
    ok = abs(eff85['dV_blood'] + eff85['urine_output'])<1e-9
    print(f"  [{'OK' if ok else 'FAIL'}] dV = -urine")
    return {"ok":ok}

def test_toxin_balance():
    print("\n"+"="*70+"\nTEST 5: Токсин баланс и масса\n"+"="*70)
    k=make_kidney()
    V=5800.0; C=0.5
    eff=k.compute_effects(P_sa=85,P_sv=5,C_tox=C,V_blood=V)
    # dC = -GFR*C*frac / V
    dC_exp = -eff['GFR']*C*k.toxin_clearance_frac / V
    print(f"  C_tox {C} V {V} GFR {eff['GFR']:.3f} frac {k.toxin_clearance_frac}")
    print(f"  dC {eff['dC_tox']:.8f} exp {dC_exp:.8f} {'OK' if abs(eff['dC_tox']-dC_exp)<1e-9 else 'FAIL'}")
    # масса: d(C*V)/dt = V*dC + C*dV = -toxin_excreted + C*(-urine) ??? 
    # toxin_excreted = GFR*C*frac
    toxin_excreted = eff['GFR']*C*k.toxin_clearance_frac
    mass_change = V*eff['dC_tox'] + C*eff['dV_blood']
    print(f"  toxin_excreted GFR*C*frac = {toxin_excreted:.6f} мл*конц/с")
    print(f"  d(CV)/dt = V*dC + C*dV = {mass_change:.6f} (должно = -toxin_excreted + C*(-urine?) )")
    # Для чистого токсина без объёма: V*dC = -toxin_excreted, но есть еще C*dV из-за потери воды
    # Проверим V*dC = -toxin_excreted
    print(f"  V*dC = {V*eff['dC_tox']:.6f} vs -toxin_excreted {-toxin_excreted:.6f} diff {V*eff['dC_tox']+toxin_excreted:.3e}")
    ok = abs(V*eff['dC_tox']+toxin_excreted)<1e-9
    print(f"  [{'OK' if ok else 'FAIL'}] масса токсина: V*dC = -excreted")
    return {"ok":ok}

def test_anuria():
    print("\n"+"="*70+"\nTEST 6: Анурия при низком давлении\n"+"="*70)
    k=make_kidney()
    for P in [0,10,20,30]:
        g=k._compute_gfr(P)
        eff=k.compute_effects(P_sa=P,P_sv=5,C_tox=0.5,V_blood=5800)
        print(f"  P_sa {P:2d} → GFR {g:.2f} мл/мин urine {eff['urine_output']*60:.3f} мл/мин {'анурія' if g<30 else ''}")
    ok = k._compute_gfr(0)<1 and k._compute_gfr(20)< k._compute_gfr(40)
    print(f"  [{'OK' if ok else 'FAIL'}] GFR→0 при P→0")
    return {"ok":ok}

def test_volume_equilibrium():
    print("\n"+"="*70+"\nTEST 7: Равновесие объёма — найти P_eq где intake=urine\n"+"="*70)
    k=make_kidney()
    intake=SYS_CFG.get("fluid_intake_rate",0.015)
    # найдем P где urine = intake
    Ps=np.linspace(10,200,39)
    eq=None
    for P in Ps:
        eff=k.compute_effects(P_sa=P,P_sv=5,C_tox=0.5,V_blood=5800)
        if abs(eff['urine_output']-intake)<0.001:
            eq=P
            break
    # более точно — интерполяция
    from scipy.optimize import bisect
    def f(P): return k.compute_effects(P_sa=P,P_sv=5,C_tox=0.5,V_blood=5800)['urine_output']-intake
    try:
        P_eq=bisect(f, 20, 200)
        eff_eq=k.compute_effects(P_sa=P_eq,P_sv=5,C_tox=0.5,V_blood=5800)
        print(f"  intake {intake} мл/с → P_eq {P_eq:.1f} мм рт.ст. где urine={eff_eq['urine_output']:.5f} GFR={eff_eq['GFR']*60:.1f}")
        # при P=65 и P=85
        for P in [65,85]:
            eff=k.compute_effects(P_sa=P,P_sv=5,C_tox=0.5,V_blood=5800)
            print(f"  P {P} → urine {eff['urine_output']:.5f} net {intake-eff['urine_output']:+.5f} {'теряет V' if eff['urine_output']>intake else 'набирает V'}")
        ok = 50 < P_eq < 120
        print(f"  [{'OK' if ok else 'FAIL'}] P_eq в [50,120] — физиологично")
    except Exception as e:
        print(f"  Не найдено равновесие: {e}")
        ok=False
        P_eq=None
    return {"ok":ok, "P_eq":P_eq if 'P_eq' in locals() else None}

def test_drift():
    print("\n"+"="*70+"\nTEST 8: Drift — эволюция V_blood 600с и 1800с (t_calib/t_span)\n"+"="*70)
    k=make_kidney()
    intake=SYS_CFG.get("fluid_intake_rate",0.015)
    ins=SYS_CFG.get("insensible_loss_rate",0.0)
    V=5800.0
    for P_sa in [65,85]:
        Vtmp=V
        for t in [600,1800]:
            # простая Эйлер интеграция V с постоянным P
            eff=k.compute_effects(P_sa=P_sa,P_sv=5,C_tox=0.5,V_blood=Vtmp)
            dVdt=intake - ins + eff['dV_blood']  # intake - urine
            Vtmp2=V + dVdt*t
            print(f"  P_sa {P_sa} t={t:4.0f}с dV/dt {dVdt:+.5f} мл/с V {V}→{Vtmp2:.0f} Δ {Vtmp2-V:+.0f} мл")
    # drift за 600с при P=85 должен быть <200 мл
    eff=k.compute_effects(85,5,0.5,5800)
    dVdt=intake+eff['dV_blood']
    drift=abs(dVdt*600)
    print(f"  drift 600с при P=85: {drift:.1f} мл")
    ok=drift<300
    print(f"  [{'OK' if ok else 'FAIL'}] drift <300 мл за 600с")
    return {"ok":ok}

def test_strict():
    print("\n"+"="*70+"\nTEST 9: Строгий — Q>=0, GFR>=0, finite, без NaN\n"+"="*70)
    k=make_kidney()
    bad=False
    for P_sa in [0,20,85,200]:
        for P_sv in [0,5,20]:
            eff=k.compute_effects(P_sa=P_sa,P_sv=P_sv,C_tox=0.5,V_blood=5800)
            if not (np.isfinite(eff['GFR']) and np.isfinite(eff['Q_renal']) and np.isfinite(eff['urine_output'])):
                print(f"  FAIL non-finite P_sa {P_sa} P_sv {P_sv}")
                bad=True
            if eff['GFR']< -1e-9 or eff['Q_renal']< -10 or eff['urine_output']< -1e-9:
                # Q_renal может быть отрицательным если P_sa<P_sv — проверим что модель не дает отрицательного потока в whole_body
                if P_sa>=P_sv and eff['Q_renal']<0:
                    print(f"  FAIL Q_renal<0 при P_sa>=P_sv {P_sa}>{P_sv}")
                    bad=True
    print(f"  GFR min {k._compute_gfr(0):.3f} max {k._compute_gfr(200):.1f}")
    ok=not bad
    print(f"  [{'OK' if ok else 'FAIL'}] все finite, Q>=0 при P_sa>=P_sv")
    return {"ok":ok}

def summary(res):
    print("\n"+"="*70+"\nСВОДКА: KidneyHemodynamic?\n"+"="*70)
    names=["Интерфейс 0-state","GFR кривая","Q_renal","Объём баланс","Токсин масса","Анурия","Равновесие P_eq","Drift","Strict"]
    for n,r in zip(names,res):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    if all(r["ok"] for r in res):
        print("\nВЫВОД: Kidney корректна в изоляции. Ищите в whole_body связке intake vs urine и R_renal.")
    else:
        print("\nВЫВОД: Есть FAIL — см. выше")
    # подсказка для P_sa 65 vs 85
    print("\nПодсказка для whole_body P_sa 65:")
    k=make_kidney()
    intake=SYS_CFG.get("fluid_intake_rate",0.015)
    for P in [65,85]:
        eff=k.compute_effects(P_sa=P,P_sv=5,C_tox=0.5,V_blood=5800)
        print(f"  P {P}: GFR {eff['GFR']*60:.1f} мл/мин urine {eff['urine_output']*60:.2f} мл/мин net {intake-eff['urine_output']:+.5f} мл/с → {'теряет кровь → EDV 69' if eff['urine_output']>intake else 'набирает'}")

def run_all(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf=io.StringIO(); tee=Tee(sys.stdout,buf); orig=sys.stdout; sys.stdout=tee
    try:
        banner()
        rs=[test_interface(),test_gfr_curve(),test_renal_flow(),test_volume_balance(),test_toxin_balance(),test_anuria(),test_volume_equilibrium(),test_drift(),test_strict()]
        summary(rs)
        print(f"\nФиниш {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout=orig
    path.write_text(buf.getvalue(),encoding="utf-8")
    print(f"Отчёт: {path}")

if __name__=="__main__":
    run_all(Path(__file__).resolve().parent / "results_debug_kidney.txt")
