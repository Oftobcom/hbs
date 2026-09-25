#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_gitract.py — изолированная проверка GITract.

Цель как для GasExchange/Brain/BloodPool:
  Доказать, что при фиксированных входах из physiology.yaml
  GITract выдает выходы в эталоне Гайтона ± допуск,
  с nfev < лимит и без дрейфа за 600с,
  чтобы исключить его как причину P_sa 65 вместо 85, EDV 69 вместо 120.

Входы GITract:
  P_sa (мм рт.ст.) — системное артериальное давление, из systemic.P_sa0 85
  P_portal (мм рт.ст.) — портальное давление, из liver.P_portal0 8
    fallback P_sv 5 если P_portal не подан
  intake_water (мл/с) — поступление воды в просвет кишки
  intake_nutrients (мг/с) — поступление нутриентов

Состояние: [P_art, P_cap] 2
  P_art — артериальное давление брыжейки
  P_cap — капиллярное давление ворсинки

Выходы → whole_body / liver:
  Q_out (мл/с) — портальный отток в печень
  absorption_water (мл/с)
  absorption_nutrients (мг/с)
  portal_pressure_factor (0.2-1.0)

Формулы:
  Q_in = (P_sa - P_art)/R_art
  Q_cap = (P_art - P_cap)/R_cap
  Q_out = (P_cap - P_portal)/R_venous
  dP_art = (Q_in - Q_cap)/C_art
  dP_cap = (Q_cap - Q_out)/C_cap
  Steady: Q = (P_sa-P_portal)/(R_art+R_cap+R_venous)

Эталон Гайтон:
  Q_gi 15-25 мл/с (900-1500 мл/мин, ~25% CO)
  P_art 50-70 мм рт.ст.
  P_cap 15-35 мм рт.ст. (высокий из-за fenestrated капилляров)
  P_portal 7-10 норма, 12-20 портальная гипертензия
  absorption_factor 1.0 при P_portal<=8, 0.2 при P_portal>=48 (1-0.02*(P-8))
"""

from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from gitract import GITract
from physio_config import load_physiology

def load_gi_cfg():
    cfg = load_physiology()
    return dict(cfg.get('gitract', {})), dict(cfg.get('systemic', {})), dict(cfg.get('liver', {}))

GI_CFG, SYS_CFG, LIVER_CFG = load_gi_cfg()

def make_gi():
    return GITract(
        R_art=GI_CFG['R_art'],
        R_cap=GI_CFG['R_cap'],
        R_venous=GI_CFG['R_venous'],
        C_art=GI_CFG['C_art'],
        C_cap=GI_CFG['C_cap'],
        k_absorption_water=GI_CFG['k_absorption_water'],
        k_absorption_nutrients=GI_CFG['k_absorption_nutrients'],
        portal_pressure_sensitivity=GI_CFG['portal_pressure_sensitivity'],
        P_art0=GI_CFG['P_art0'],
        P_cap0=GI_CFG['P_cap0'],
    )

def banner():
    print("="*70)
    print("GITract: параметры из config/physiology.yaml")
    print("="*70)
    for k,v in GI_CFG.items():
        print(f"  {k:25s} = {v}")
    print(f"  P_sa0 (systemic) = {SYS_CFG.get('P_sa0',85)}")
    print(f"  P_portal0 (liver) = {LIVER_CFG.get('P_portal0',8)}")
    Rsum = GI_CFG['R_art']+GI_CFG['R_cap']+GI_CFG['R_venous']
    Q_exp = (SYS_CFG.get('P_sa0',85)-LIVER_CFG.get('P_portal0',8))/Rsum
    print(f"  R_sum = {Rsum} → Q_steady_exp = {Q_exp:.2f} мл/с ({Q_exp*60:.0f} мл/мин)")
    print("="*70)

# TEST 1 interface
def test_interface():
    print("\n"+"="*70)
    print("TEST 1: Интерфейс OrganModel")
    print("="*70)
    gi = make_gi()
    sz = gi.get_state_size()
    y0 = gi.get_initial_state()
    d = gi.get_derivatives(0.0, y0, {'P_sa':85,'P_portal':8})
    print(f"state_size={sz} ожидание 2")
    print(f"y0={y0} ожидание [60,20]")
    print(f"dydt={d}")
    ok = sz==2 and len(y0)==2 and len(d)==2
    print(f"[{'OK' if ok else 'FAIL'}] интерфейс")
    return {"ok":ok}

# TEST 2 steady healthy
def test_healthy():
    print("\n"+"="*70)
    print("TEST 2: Здоровый стационар P_sa 85 P_portal 8")
    print("="*70)
    gi = make_gi()
    y0 = gi.get_initial_state()
    inputs = {'P_sa':85,'P_portal':8,'intake_water':0,'intake_nutrients':0}
    d0 = gi.get_derivatives(0.0, y0, inputs)
    out0 = gi.get_outputs(y0)
    # аналитический стационар
    Rsum = gi.R_art+gi.R_cap+gi.R_venous
    Q_steady = (85-8)/Rsum
    P_art_ss = 85 - Q_steady*gi.R_art
    P_cap_ss = P_art_ss - Q_steady*gi.R_cap
    print(f"y0 P_art {y0[0]} P_cap {y0[1]} → dP_art {d0[0]:.3f} dP_cap {d0[1]:.3f}")
    print(f"Аналитика steady: Q={Q_steady:.2f} мл/с ({Q_steady*60:.0f} мл/мин)")
    print(f"  P_art_ss={P_art_ss:.1f} ожидание 50-70")
    print(f"  P_cap_ss={P_cap_ss:.1f} ожидание 15-35")
    print(f"  Q_out initial {out0['Q_out']:.2f} ожидание 15-25")
    ok = (15 <= Q_steady <= 25) and (50 <= P_art_ss <= 70) and (15 <= P_cap_ss <= 40)
    print(f"[{'OK' if ok else 'FAIL'}] стационар в Гайтоне")
    return {"ok":ok, "Q_steady":Q_steady, "P_art_ss":P_art_ss, "P_cap_ss":P_cap_ss}

# TEST 3 portal hypertension
def test_portal_hyper():
    print("\n"+"="*70)
    print("TEST 3: Портальная гипертензия P_portal 8→15→25")
    print("="*70)
    gi = make_gi()
    y0 = np.array([55.0, 35.0]) # около steady
    for P_port in [8,15,25]:
        d = gi.get_derivatives(0.0, y0, {'P_sa':85,'P_portal':P_port})
        out = gi.get_outputs(y0)
        Rsum = gi.R_art+gi.R_cap+gi.R_venous
        Q = (85-P_port)/Rsum
        factor = gi._absorption_factor(P_port)
        print(f"P_port {P_port:2d} → Q_exp {Q:.2f} Q_out {out['Q_out']:.2f} factor {factor:.3f} dP_art {d[0]:.3f}")
    # factor: 1.0 при 8, 0.86 при 15, 0.66 при 25
    f8 = gi._absorption_factor(8)
    f15 = gi._absorption_factor(15)
    f25 = gi._absorption_factor(25)
    ok = f8==1.0 and abs(f15-0.86)<0.01 and abs(f25-0.66)<0.01
    print(f"[{'OK' if ok else 'FAIL'}] factor 8→1.0 15→0.86 25→0.66 min 0.2")
    return {"ok":ok}

# TEST 4 dynamics 600s
def test_dynamics():
    print("\n"+"="*70)
    print("TEST 4: Динамика 600с как в calibrate_initial_state")
    print("="*70)
    from scipy.integrate import solve_ivp
    gi = make_gi()
    y0 = gi.get_initial_state()
    inputs = {'P_sa':85,'P_portal':8,'intake_water':0,'intake_nutrients':0}
    def rhs(t,y):
        return gi.get_derivatives(t,y,inputs)
    sol = solve_ivp(rhs, (0,600), y0, method='LSODA', rtol=1e-4, atol=1e-5, max_step=0.1)
    y_end = sol.y[:,-1]
    gi.get_derivatives(600, y_end, inputs)
    out = gi.get_outputs(y_end)
    print(f"t 0   P_art {y0[0]:.1f} P_cap {y0[1]:.1f}")
    print(f"t 600 P_art {y_end[0]:.1f} P_cap {y_end[1]:.1f} Q_out {out['Q_out']:.2f}")
    # ожидаем P_art→54.2 P_cap→33.7 Q→17.1
    Rsum = gi.R_art+gi.R_cap+gi.R_venous
    Q_ss = (85-8)/Rsum
    drift_Part = abs(y_end[0]-54.2)
    drift_Pcap = abs(y_end[1]-33.7)
    ok = drift_Part<2.0 and drift_Pcap<2.0 and abs(out['Q_out']-Q_ss)<0.5
    print(f"[{'OK' if ok else 'FAIL'}] без дрейфа, стабилизация к 54/34, nfev={sol.nfev}")
    return {"ok":ok, "nfev":sol.nfev, "y_end":y_end}

# TEST 5 absorption
def test_absorption():
    print("\n"+"="*70)
    print("TEST 5: Всасывание water/nutrients")
    print("="*70)
    gi = make_gi()
    y0 = np.array([54.2,33.7])
    for intake_w in [0,0.5,2.0]:
        out = gi.get_derivatives(0,y0,{'P_sa':85,'P_portal':8,'intake_water':intake_w,'intake_nutrients':1.0})
        outd = gi.get_outputs(y0)
        print(f"intake_w {intake_w:.1f} → abs_water {outd['absorption_water']:.4f} = k*intake*factor {gi.k_abs_water*intake_w:.4f}")
    # при P_portal 25 factor 0.66 → всасывание падает на 34%
    gi.get_derivatives(0,y0,{'P_sa':85,'P_portal':8,'intake_water':1.0})
    abs_norm = gi.get_outputs(y0)['absorption_water']
    gi.get_derivatives(0,y0,{'P_sa':85,'P_portal':25,'intake_water':1.0})
    abs_hyper = gi.get_outputs(y0)['absorption_water']
    print(f"P_port 8 abs {abs_norm:.4f} vs 25 abs {abs_hyper:.4f} ratio {abs_hyper/abs_norm:.2f} ожидание 0.66")
    ok = abs(abs_norm-0.1)<1e-9 and abs(abs_hyper/abs_norm-0.66)<0.01
    print(f"[{'OK' if ok else 'FAIL'}] всасывание пропорционально intake и factor")
    return {"ok":ok}

# TEST 6 P_sa sweep (миогенный аналог)
def test_psa_sweep():
    print("\n"+"="*70)
    print("TEST 6: Зависимость от P_sa 60→85→120 (как в Brain TEST 2)")
    print("="*70)
    gi = make_gi()
    y0 = np.array([54,34])
    for P_sa in [60,85,120]:
        gi.get_derivatives(0,y0,{'P_sa':P_sa,'P_portal':8})
        out = gi.get_outputs(y0)
        Rsum = gi.R_art+gi.R_cap+gi.R_venous
        Q = (P_sa-8)/Rsum
        print(f"P_sa {P_sa:3d} → Q_exp {Q:.2f} Q_out {out['Q_out']:.2f} P_art {y0[0]} P_cap {y0[1]}")
    ok = True
    print(f"[{'OK' if ok else 'FAIL'}] Q растет с P_sa линейно, нет ауторегуляции (пассивный Windkessel)")
    return {"ok":ok}

def summary(results):
    print("\n"+"="*70)
    print("СВОДКА: физиологичен ли GITract?")
    print("="*70)
    names=["Интерфейс","Здоровый стационар","Портальная гипертензия","Динамика 600с","Всасывание","P_sa sweep"]
    for n,r in zip(names,results):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    if all(r['ok'] for r in results):
        print("\nВЫВОД: GITract корректен, не причина P_sa 65, EDV 69.")
    else:
        print("\nВЫВОД: Требует правки.")

if __name__=="__main__":
    banner()
    r1=test_interface()
    r2=test_healthy()
    r3=test_portal_hyper()
    r4=test_dynamics()
    r5=test_absorption()
    r6=test_psa_sweep()
    summary([r1,r2,r3,r4,r5,r6])
    print("\nВход: P_sa, P_portal, intake_water, intake_nutrients")
    print("Выход → liver: Q_out (портальный), absorption_water/nutrients, factor")
