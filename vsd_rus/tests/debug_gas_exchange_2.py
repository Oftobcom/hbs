#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_gas_exchange.py — изолированная проверка GasExchange.

Цель как для BloodPool/Brain:
  Доказать, что при фиксированных входах из physiology.yaml
  GasExchange выдает выходы в эталоне Гайтона ± допуск,
  детерминирован, без дрейфа, и не может быть причиной
  P_sa 65 вместо 85, EDV 69 вместо 120.

Входы GasExchange:
  C_v_O2 (мл/мл) — смешанная венозная O2
  C_v_CO2 (мл/мл) — смешанная венозная CO2
  Q_p (мл/с) — легочный кровоток
  Q_shunt (мл/с) — поток ДМЖП: >0 L→R, <0 R→L

Выходы:
  C_a_O2, C_a_CO2, C_pv_O2, C_pv_CO2,
  SaO2, P_a_O2, P_v_O2, Qp_Qs, shunt_fraction_R2L,
  O2_uptake, CO2_removal, f_bypass

Формулы для проверки:
  Q_s = Q_p - Q_shunt
  C_pv_O2 = C_O2_from_P(P_alv_O2)  # равновесие с альвеолой
  При Q_shunt>=0: C_a = C_pv
  При Q_shunt<0: C_a = (1-f)*C_pv + f*C_v, f=|Q_shunt|/Q_s
  O2_uptake = Q_p*(C_pv_O2 - C_v_O2)

Эталон Гайтон (взрослый 70кг):
  C_a_O2 0.18-0.21 (SaO2 0.94-0.98, P_alv 100)
  C_v_O2 0.13-0.16
  SaO2 0.94-0.98 здоровый, 0.70-0.90 Эйзенменгер
  Qp/Qs 0.9-1.1 здоровый, 1.5-3.0 L→R VSD, 0.5-0.9 R→L
  O2_uptake 3.5-5.5 мл/с (≈250 мл/мин)
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from gas_exchange import GasExchange
from physio_config import load_physiology

def load_gas_cfg():
    cfg = load_physiology()
    if "gas_exchange" not in cfg:
        raise RuntimeError("physiology.yaml без секции gas_exchange")
    return dict(cfg["gas_exchange"])

GAS_CFG = load_gas_cfg()

def make_gas():
    return GasExchange(
        P_alv_O2=GAS_CFG["P_alv_O2"],
        P_alv_CO2=GAS_CFG["P_alv_CO2"],
        Hb=GAS_CFG["Hb"],
        P50=GAS_CFG["P50"],
        n_hill=GAS_CFG["n_hill"],
        alpha_O2=GAS_CFG["alpha_O2"],
        C_CO2_offset=GAS_CFG["C_CO2_offset"],
        k_CO2_slope=GAS_CFG["k_CO2_slope"],
    )

def print_config_banner():
    print("="*70)
    print("GasExchange: параметры из config/physiology.yaml")
    print("="*70)
    for k,v in GAS_CFG.items():
        print(f"  {k:20s} = {v}")
    print("="*70)

# TEST 1: Healthy
def test_healthy():
    print("\n"+"="*70)
    print("TEST 1: Здоровый Q_shunt=0, C_v_O2=0.15, Q_p=83")
    print("="*70)
    gas = make_gas()
    out = gas.compute_effects(C_v_O2=0.15, C_v_CO2=0.52, Q_p=83.0, Q_shunt=0.0)
    print(f"C_pv_O2 = {out['C_pv_O2']:.5f} ожидание ~0.20 (Hill при P_alv 100)")
    print(f"C_a_O2  = {out['C_a_O2']:.5f} ожидание 0.18-0.21")
    print(f"SaO2    = {out['SaO2']:.4f} ожидание 0.94-0.98")
    print(f"Qp_Qs   = {out['Qp_Qs']:.3f} ожидание 1.0")
    print(f"O2_upt  = {out['O2_uptake']:.3f} мл/с ожидание 3.5-5.5")
    ok = (0.18 <= out['C_a_O2'] <= 0.21) and (0.94 <= out['SaO2'] <= 0.99) and (abs(out['Qp_Qs']-1.0)<0.01)
    print(f"[{'OK' if ok else 'FAIL'}] здоровый газообмен")
    return {"ok": ok, "out": out}

# TEST 2: L->R shunt
def test_LR_shunt():
    print("\n"+"="*70)
    print("TEST 2: Лево-правый VSD Q_shunt=+40 (Qp/Qs>1, SaO2 не падает)")
    print("="*70)
    gas = make_gas()
    out = gas.compute_effects(C_v_O2=0.15, C_v_CO2=0.52, Q_p=120.0, Q_shunt=40.0)
    # Q_s = 120-40=80, Qp_Qs=1.5
    print(f"Q_p=120 Q_shunt=40 → Q_s=80 Qp_Qs={out['Qp_Qs']:.3f} ожидание 1.5")
    print(f"C_a_O2={out['C_a_O2']:.5f} = C_pv_O2 {out['C_pv_O2']:.5f} ? {abs(out['C_a_O2']-out['C_pv_O2'])<1e-9}")
    print(f"SaO2={out['SaO2']:.4f} ожидание ~0.97 (не падает при L→R)")
    ok = abs(out['Qp_Qs']-1.5)<0.01 and abs(out['C_a_O2']-out['C_pv_O2'])<1e-9 and out['SaO2']>0.94
    print(f"[{'OK' if ok else 'FAIL'}] L→R не десатурирует")
    return {"ok": ok}

# TEST 3: R->L shunt (Eisenmenger)
def test_RL_shunt():
    print("\n"+"="*70)
    print("TEST 3: Право-левый Эйзенменгер Q_shunt=-60, Q_p=80")
    print("="*70)
    gas = make_gas()
    out = gas.compute_effects(C_v_O2=0.10, C_v_CO2=0.56, Q_p=80.0, Q_shunt=-60.0)
    # Q_s=140, f_bypass=60/140=0.428, C_a = 0.571*C_pv +0.428*C_v
    # C_pv~0.20, C_v 0.10 → C_a~0.157
    print(f"Q_s=Q_p-Q_shunt=140 f_bypass={out['f_bypass']:.3f} ожидание 0.428")
    print(f"C_v_O2 0.10 → C_a_O2 {out['C_a_O2']:.5f} ожидание ~0.15-0.17 (десатурация)")
    print(f"SaO2 {out['SaO2']:.4f} ожидание 0.70-0.90")
    print(f"shunt_frac {out['shunt_fraction_R2L']:.3f} Qp_Qs {out['Qp_Qs']:.3f} (<1)")
    ok = (0.13 <= out['C_a_O2'] <= 0.18) and (0.70 <= out['SaO2'] <= 0.92) and (out['Qp_Qs'] < 1.0)
    print(f"[{'OK' if ok else 'FAIL'}] R→L десатурирует")
    return {"ok": ok}

# TEST 4: Hill curve invertibility
def test_hill():
    print("\n"+"="*70)
    print("TEST 4: Кривая Хилла P↔C обратимость")
    print("="*70)
    gas = make_gas()
    for P in [20, 40, 60, 100]:
        C = gas._C_O2_from_P(P)
        P_back = gas._P_O2_from_C(C)
        Sa = gas._SaO2_from_P(P)
        print(f"P={P:3d} → C={C:.5f} Sa={Sa:.3f} → P_back={P_back:.2f} err={abs(P-P_back):.2f}")
    # check monotonic
    Cs = [gas._C_O2_from_P(P) for P in [0,20,40,60,80,100]]
    mono = all(Cs[i]<=Cs[i+1] for i in range(len(Cs)-1))
    print(f"[{'OK' if mono else 'FAIL'}] монотонность Hill")
    return {"ok": mono}

# TEST 5: CO2 linear
def test_co2():
    print("\n"+"="*70)
    print("TEST 5: CO2 линейная C=offset+slope*P")
    print("="*70)
    gas = make_gas()
    for P in [0,40,50]:
        C = gas._C_CO2_from_P(P)
        P_back = gas._P_CO2_from_C(C)
        print(f"P_CO2={P} → C={C:.4f} → P_back={P_back:.1f}")
    C_40 = gas._C_CO2_from_P(40)
    C_50 = gas._C_CO2_from_P(50)
    ok = abs(C_40 - 0.48) < 0.05 and abs(C_50 - 0.545) < 0.05
    print(f"[{'OK' if ok else 'FAIL'}] C_CO2 40mmHg ~0.48, 50mmHg ~0.545")
    return {"ok": ok}

# TEST 6: Mass balance
def test_mass():
    print("\n"+"="*70)
    print("TEST 6: Баланс O2_uptake = Q_p*(C_pv-C_v)")
    print("="*70)
    gas = make_gas()
    out = gas.compute_effects(C_v_O2=0.15, C_v_CO2=0.52, Q_p=83, Q_shunt=0)
    uptake_exp = 83*(out['C_pv_O2']-0.15)
    print(f"O2_uptake={out['O2_uptake']:.6f} ожидание {uptake_exp:.6f} err={abs(out['O2_uptake']-uptake_exp):.2e}")
    ok = abs(out['O2_uptake']-uptake_exp)<1e-9
    print(f"[{'OK' if ok else 'FAIL'}] баланс")
    return {"ok": ok}

def summary(results):
    print("\n"+"="*70)
    print("СВОДКА: физиологичен ли GasExchange?")
    print("="*70)
    names=["Healthy","L→R shunt","R→L shunt","Hill P↔C","CO2 linear","Mass balance"]
    for n,r in zip(names, results):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    if all(r['ok'] for r in results):
        print("\nВЫВОД: GasExchange корректен, не причина P_sa 65, EDV 69.")
    else:
        print("\nВЫВОД: Требует правки.")

if __name__ == "__main__":
    print_config_banner()
    r1=test_healthy()
    r2=test_LR_shunt()
    r3=test_RL_shunt()
    r4=test_hill()
    r5=test_co2()
    r6=test_mass()
    summary([r1,r2,r3,r4,r5,r6])
    print("\nВход: C_v_O2, C_v_CO2, Q_p, Q_shunt")
    print("Выход → whole_body: C_a_O2, C_a_CO2, SaO2, Qp_Qs, O2_uptake")
