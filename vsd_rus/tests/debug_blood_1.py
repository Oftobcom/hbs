#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_blood.py — изолированная проверка BloodPool.

Что подаём:
  state = [V_blood, C_0..C_7]
  inputs = {'dV': float, 'dC': np.ndarray(num_substances)}

Что ожидаем:
  - Zero-input: V, C инвариантны (steady state whole_body)
  - Dilution law: M_i = V · C_i сохраняется при dC_i = 0
  - Source term: C_i(t) = C_i0 + k_i · t при dV = 0
  - Volume clamp: dV < 0 гасится при V < V_min
  - Physio init: V0=5800, C_O2=0.15, C_CO2=0.52 из physiology.yaml

Формулы (что должно выполняться математически):
  dV/dt = dV_input
  dC_i/dt = dC_i_input − C_i · dV/V
  dM_i/dt = V · dC_i_input     (проверка mass balance)

Физиологический reference (Gayton, взрослый 70 кг):
  V_blood    ≈ 5000–6000 мл
  C_O2       ≈ 0.15–0.20 мл/мл
  C_CO2      ≈ 0.48–0.52 мл/мл
  C_lactate  ≈ 0.05–0.15 мг/мл

Запуск:
    python tests/debug_blood.py
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import solve_ivp

from blood import BloodPool
from physio_config import load_physiology


# =====================================================================
# 0. Загрузка конфигурации из physiology.yaml
# =====================================================================

def load_blood_cfg() -> dict:
    cfg = load_physiology()
    if "blood" not in cfg:
        raise RuntimeError(
            f"physiology.yaml найден, но без секции 'blood'.\n"
            f"Проверь: {ROOT / 'config' / 'physiology.yaml'}"
        )
    blood_cfg = dict(cfg["blood"])
    for k in ("V0", "initial_concentrations"):
        if k not in blood_cfg:
            raise RuntimeError(f"blood: отсутствует ключ {k}")
    return blood_cfg


BLOOD_CFG    = load_blood_cfg()
V0           = float(BLOOD_CFG["V0"])
INIT_CONC    = dict(BLOOD_CFG["initial_concentrations"])
SUBSTANCES   = list(INIT_CONC.keys())
N_SUBST      = len(SUBSTANCES)
V_MIN        = 2000.0     # дефолт в BloodPool.__init__

IDX          = {name: i for i, name in enumerate(SUBSTANCES)}
IDX_O2       = IDX["oxygen"]
IDX_CO2      = IDX["co2"]
IDX_LACTATE  = IDX["lactate"]


def make_blood() -> BloodPool:
    return BloodPool(
        substance_names=SUBSTANCES,
        V0=V0,
        V_min=V_MIN,
        initial_concentrations=INIT_CONC,
    )


def print_config_banner():
    print("=" * 70)
    print("BloodPool: параметры из config/physiology.yaml")
    print("=" * 70)
    print(f"  V0        = {V0:.1f} мл")
    print(f"  V_min     = {V_MIN:.1f} мл (дефолт)")
    print(f"  substances = {SUBSTANCES}")
    print(f"  initial_concentrations:")
    for name, c0 in INIT_CONC.items():
        print(f"    {name:>10s} = {c0:.4f}")
    print("=" * 70)


# =====================================================================
# TEST 1: Zero-input invariance (steady state)
# =====================================================================

def test_zero_input():
    print("\n" + "=" * 70)
    print("TEST 1: Zero-input invariance (dV=0, dC=0)")
    print("=" * 70)

    blood = make_blood()
    y0 = blood.get_initial_state()
    dC_zero = np.zeros(N_SUBST)

    def rhs(t, y):
        return blood.get_derivatives(t, y, {"dV": 0.0, "dC": dC_zero})

    sol = solve_ivp(rhs, (0.0, 1000.0), y0,
                    method="LSODA", rtol=1e-10, atol=1e-12)

    V_end = float(sol.y[0, -1])
    C_end = sol.y[1:, -1]

    drift_V = abs(V_end - V0) / V0
    drift_C = np.max(np.abs(C_end - y0[1:]))

    print(f"V(t=1000)         = {V_end:.9f}   (target {V0})")
    print(f"drift_V           = {drift_V:.2e}")
    print(f"max |ΔC_i|        = {drift_C:.2e}")
    print(f"nfev              = {sol.nfev}")

    ok_V = drift_V < 1e-9
    ok_C = drift_C < 1e-9
    flag = "OK" if (ok_V and ok_C) else "FAIL"
    print(f"[{flag}] V, C инвариантны при нулевых входах")

    return {"drift_V": drift_V, "drift_C": drift_C, "nfev": sol.nfev,
            "ok": ok_V and ok_C}


# =====================================================================
# TEST 2: Dilution law — mass conservation
# =====================================================================

def test_dilution_mass():
    print("\n" + "=" * 70)
    print("TEST 2: Dilution law (dV=+0.02, dC=0) — сохранение массы")
    print("=" * 70)

    blood = make_blood()
    y0 = blood.get_initial_state()
    dC_zero = np.zeros(N_SUBST)
    dV = 0.02      # мл/с — «пьём воду, не писаем»

    def rhs(t, y):
        return blood.get_derivatives(t, y, {"dV": dV, "dC": dC_zero})

    sol = solve_ivp(rhs, (0.0, 1000.0), y0,
                    method="LSODA", rtol=1e-10, atol=1e-12)

    V_end = float(sol.y[0, -1])
    V_expected = V0 + dV * 1000.0

    print(f"V(t=1000) = {V_end:.6f}   ожидание {V_expected:.6f}   "
          f"err = {abs(V_end - V_expected):.2e}")

    # Проверка mass conservation по всем веществам
    print(f"\n  {'substance':>10}  {'M(0)':>12}  {'M(1000)':>12}  {'rel_err':>10}")
    max_mass_err = 0.0
    for name in SUBSTANCES:
        i = IDX[name]
        c0 = y0[1 + i]
        M_0 = V0 * c0
        M_end = V_end * sol.y[1 + i, -1]
        if abs(M_0) > 1e-12:
            rel = abs(M_end - M_0) / abs(M_0)
        else:
            rel = abs(M_end - M_0)
        max_mass_err = max(max_mass_err, rel)
        print(f"  {name:>10}  {M_0:>12.6f}  {M_end:>12.6f}  {rel:>10.2e}")

    # Проверка expected dilution для O2
    c_o2_expected = INIT_CONC["oxygen"] * V0 / V_end
    c_o2_end = float(sol.y[1 + IDX_O2, -1])
    print(f"\nC_O2(t=1000) = {c_o2_end:.8f}   ожидание {c_o2_expected:.8f}   "
          f"err = {abs(c_o2_end - c_o2_expected):.2e}")

    ok_mass = max_mass_err < 1e-8
    ok_dilution = abs(c_o2_end - c_o2_expected) < 1e-8
    flag = "OK" if (ok_mass and ok_dilution) else "FAIL"
    print(f"\n[{flag}] масса сохраняется, разбавление соответствует V0/V(t)")

    return {"V_end": V_end, "max_mass_err": max_mass_err,
            "nfev": sol.nfev, "ok": ok_mass and ok_dilution}


# =====================================================================
# TEST 3: Source accumulation (dV=0, dC≠0)
# =====================================================================

def test_source_accumulation():
    print("\n" + "=" * 70)
    print("TEST 3: Source accumulation (dV=0, dC_O2=+1e-5)")
    print("=" * 70)

    blood = make_blood()
    y0 = blood.get_initial_state()
    dC = np.zeros(N_SUBST)
    dC[IDX_O2] = 1e-5      # (мл/мл)/с

    def rhs(t, y):
        return blood.get_derivatives(t, y, {"dV": 0.0, "dC": dC})

    sol = solve_ivp(rhs, (0.0, 1000.0), y0,
                    method="LSODA", rtol=1e-10, atol=1e-12)

    V_end = float(sol.y[0, -1])
    c_o2_end = float(sol.y[1 + IDX_O2, -1])
    c_o2_expected = INIT_CONC["oxygen"] + 1e-5 * 1000.0  # = 0.16

    print(f"V(t=1000)    = {V_end:.6f}   (dV=0 → V не меняется)")
    print(f"C_O2(t=1000) = {c_o2_end:.8f}   ожидание {c_o2_expected:.8f}   "
          f"err = {abs(c_o2_end - c_o2_expected):.2e}")

    # Другие вещества не изменились
    other_drift = max(
        abs(float(sol.y[1 + i, -1]) - y0[1 + i])
        for i in range(N_SUBST) if i != IDX_O2
    )
    print(f"max |ΔC_i| для остальных = {other_drift:.2e}")

    ok = (abs(c_o2_end - c_o2_expected) < 1e-8) and (other_drift < 1e-9)
    flag = "OK" if ok else "FAIL"
    print(f"[{flag}] линейный рост источника, остальные C не меняются")

    return {"c_o2_end": c_o2_end, "other_drift": other_drift,
            "nfev": sol.nfev, "ok": ok}


# =====================================================================
# TEST 4: Volume soft clamp (V < V_min)
# =====================================================================

def test_volume_clamp():
    print("\n" + "=" * 70)
    print("TEST 4: Volume soft clamp (V < V_min = 2000)")
    print("=" * 70)

    blood = make_blood()
    dC_zero = np.zeros(N_SUBST)

    # 4.1: V = V_min − 1, dV < 0 → dV должен быть обнулён
    y = blood.get_initial_state()
    y[0] = V_MIN - 1.0
    dy = blood.get_derivatives(0.0, y, {"dV": -0.01, "dC": dC_zero})
    dV_out_neg = float(dy[0])
    flag_neg = "OK" if dV_out_neg == 0.0 else "FAIL"
    print(f"V = {V_MIN-1:.1f}, dV_in = -0.01 → dV_out = {dV_out_neg:.6f}  [{flag_neg}]")

    # 4.2: V = V_min − 1, dV > 0 → dV сохраняется
    dy = blood.get_derivatives(0.0, y, {"dV": +0.01, "dC": dC_zero})
    dV_out_pos = float(dy[0])
    flag_pos = "OK" if dV_out_pos == 0.01 else "FAIL"
    print(f"V = {V_MIN-1:.1f}, dV_in = +0.01 → dV_out = {dV_out_pos:.6f}  [{flag_pos}]")

    # 4.3: V = 5000 (далеко от V_min), dV < 0 → dV сохраняется
    y[0] = 5000.0
    dy = blood.get_derivatives(0.0, y, {"dV": -0.01, "dC": dC_zero})
    dV_out_norm = float(dy[0])
    flag_norm = "OK" if abs(dV_out_norm - (-0.01)) < 1e-12 else "FAIL"
    print(f"V = 5000.0, dV_in = -0.01 → dV_out = {dV_out_norm:.6f}  [{flag_norm}]")

    # 4.4: симуляция с dV < 0 при V < V_min: V не должно уходить в минус
    y0 = blood.get_initial_state()
    y0[0] = V_MIN + 5.0
    def rhs(t, y):
        return blood.get_derivatives(t, y, {"dV": -1.0, "dC": dC_zero})
    sol = solve_ivp(rhs, (0.0, 100.0), y0,
                    method="LSODA", rtol=1e-10, atol=1e-12)
    V_min_reached = float(sol.y[0, :].min())
    V_end = float(sol.y[0, -1])
    print(f"V(0) = {V_MIN+5:.1f}, dV = -1.0 → V_end = {V_end:.4f}, "
          f"V_min_during = {V_min_reached:.4f}")

    ok = (dV_out_neg == 0.0) and (dV_out_pos == 0.01) \
         and (abs(dV_out_norm + 0.01) < 1e-12) \
         and (V_min_reached >= V_MIN - 0.001)
    flag = "OK" if ok else "FAIL"
    print(f"[{flag}] soft clamp работает корректно")

    return {"ok": ok, "V_end": V_end, "V_min_during": V_min_reached}


# =====================================================================
# TEST 5: Physiological transient
# =====================================================================

def test_physio_transient():
    print("\n" + "=" * 70)
    print("TEST 5: Физиологический транзиент (dV=0, dC_O2=-1e-5, dC_CO2=+5e-6)")
    print("=" * 70)

    blood = make_blood()
    y0 = blood.get_initial_state()
    dC = np.zeros(N_SUBST)
    dC[IDX_O2]  = -1e-5
    dC[IDX_CO2] = +5e-6

    def rhs(t, y):
        return blood.get_derivatives(t, y, {"dV": 0.0, "dC": dC})

    sol = solve_ivp(rhs, (0.0, 1000.0), y0,
                    method="LSODA", rtol=1e-10, atol=1e-12)

    c_o2_end  = float(sol.y[1 + IDX_O2,  -1])
    c_co2_end = float(sol.y[1 + IDX_CO2, -1])

    c_o2_expected  = INIT_CONC["oxygen"] - 1e-5 * 1000.0
    c_co2_expected = INIT_CONC["co2"]    + 5e-6 * 1000.0

    print(f"C_O2(t=1000)  = {c_o2_end:.8f}   ожидание {c_o2_expected:.8f}")
    print(f"C_CO2(t=1000) = {c_co2_end:.8f}   ожидание {c_co2_expected:.8f}")

    # Проверка физиологического диапазона
    EPS = 1e-9
    in_range_O2  = (0.14 - EPS) <= c_o2_end  <= (0.20 + EPS)
    in_range_CO2 = (0.48 - EPS) <= c_co2_end <= (0.55 + EPS)
    flag = "OK" if (in_range_O2 and in_range_CO2) else "FAIL"
    print(f"[{flag}] выходы в физиологическом диапазоне Gayton")

    return {"c_o2_end": c_o2_end, "c_co2_end": c_co2_end,
            "ok": in_range_O2 and in_range_CO2, "nfev": sol.nfev}


# =====================================================================
# TEST 6: Initial state sanity + RK45 vs LSODA
# =====================================================================

def test_init_state():
    print("\n" + "=" * 70)
    print("TEST 6: Initial state sanity (совпадение с physiology.yaml)")
    print("=" * 70)

    blood = make_blood()
    y0 = blood.get_initial_state()

    print(f"V_blood(0) = {y0[0]:.4f}   ожидание {V0:.4f}")
    ok_V = abs(y0[0] - V0) < 1e-12

    all_ok = ok_V
    for name in SUBSTANCES:
        i = IDX[name]
        c0_got = float(y0[1 + i])
        c0_exp = INIT_CONC[name]
        ok_i = abs(c0_got - c0_exp) < 1e-12
        all_ok = all_ok and ok_i
        flag_i = "OK" if ok_i else "FAIL"
        print(f"  C_{name:>10s}(0) = {c0_got:.6f}   ожидание {c0_exp:.6f}  [{flag_i}]")

    print(f"\n[{'OK' if all_ok else 'FAIL'}] начальное состояние полностью "
          f"совпадает с physiology.yaml")

    # RK45 vs LSODA на 100 с
    dC_zero = np.zeros(N_SUBST)
    def rhs(t, y):
        return blood.get_derivatives(t, y, {"dV": 0.01, "dC": dC_zero})

    sol_R = solve_ivp(rhs, (0.0, 100.0), y0, method="RK45", rtol=1e-10, atol=1e-12)
    sol_L = solve_ivp(rhs, (0.0, 100.0), y0, method="LSODA", rtol=1e-10, atol=1e-12)
    diff_V = abs(sol_R.y[0, -1] - sol_L.y[0, -1])
    print(f"\nRK45 vs LSODA на 100 с (dV=0.01):")
    print(f"  V_RK45  = {sol_R.y[0, -1]:.9f}  nfev = {sol_R.nfev}")
    print(f"  V_LSODA = {sol_L.y[0, -1]:.9f}  nfev = {sol_L.nfev}")
    print(f"  |diff|  = {diff_V:.2e}  [{'OK' if diff_V < 1e-8 else 'FAIL'}]")

    return {"ok": all_ok, "diff_RK45_LSODA": diff_V}


# =====================================================================
# Сводка
# =====================================================================

def summary(results: list):
    print("\n" + "=" * 70)
    print("СВОДКА: физиологичен ли BloodPool?")
    print("=" * 70)

    checks = [
        ("Zero-input invariance", results[0]["ok"],
         f"drift_V = {results[0]['drift_V']:.2e}"),
        ("Dilution mass conservation", results[1]["ok"],
         f"max_err = {results[1]['max_mass_err']:.2e}"),
        ("Source accumulation", results[2]["ok"],
         f"c_O2 err = {abs(results[2]['c_o2_end'] - 0.16):.2e}"),
        ("Volume soft clamp", results[3]["ok"],
         f"V_end = {results[3]['V_end']:.2f}"),
        ("Physio transient range", results[4]["ok"],
         f"C_O2={results[4]['c_o2_end']:.4f}, C_CO2={results[4]['c_co2_end']:.4f}"),
        ("Initial state matches yaml", results[5]["ok"],
         f"|RK45-LSODA| = {results[5]['diff_RK45_LSODA']:.2e}"),
    ]

    print()
    for name, ok, detail in checks:
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}] {name:<32} : {detail}")
    print()

    if all(ok for _, ok, _ in checks):
        print("ВЫВОД: BloodPool корректен и готов к интеграции с whole_body.")
    else:
        print("ВЫВОД: BloodPool требует правки — см. выше.")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    print_config_banner()

    r1 = test_zero_input()
    r2 = test_dilution_mass()
    r3 = test_source_accumulation()
    r4 = test_volume_clamp()
    r5 = test_physio_transient()
    r6 = test_init_state()

    summary([r1, r2, r3, r4, r5, r6])

    print("\n" + "=" * 70)
    print("Вход BloodPool: dV (float), dC (np.ndarray длины 8).")
    print("Выход BloodPool → whole_body: V_blood, C_<substance>.")
    print("=" * 70)