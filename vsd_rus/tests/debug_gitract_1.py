#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_gitract.py — изолированная проверка GITract.

Модель: 2 состояния [P_art, P_cap], входы P_sa, P_portal,
intake_water, intake_nutrients.
Выходы: Q_out, absorption_water, absorption_nutrients,
portal_pressure_factor.

Цель: доказать, что GITract при физиологических входах из
physiology.yaml даёт稳态 в Gayton-диапазоне, без дрейфа,
nfev < лимит, и не может быть причиной P_sa 65, EDV 69.

Steady state (аналитика):
    Q_ss   = (P_sa - P_portal)/(R_art + R_cap + R_venous)
    P_art  = P_sa - Q_ss · R_art
    P_cap  = P_art - Q_ss · R_cap

Gayton (взрослый 70 кг):
    Splanchnic flow: 15-20% CO = 12-17 ml/s
    P_art (мезентериальная): 40-80 мм рт. ст.
    P_cap: 20-45 мм рт. ст.
    P_portal: 5-12 мм рт. ст.

Запуск:
    python tests/debug_gitract.py
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import solve_ivp

from gitract import GITract
from physio_config import load_physiology


# =====================================================================
# 0. Загрузка из physiology.yaml
# =====================================================================

def load_git_cfg() -> dict:
    cfg = load_physiology()
    if "gitract" not in cfg:
        raise RuntimeError("physiology.yaml: нет секции 'gitract'")
    return dict(cfg["gitract"])


def load_inputs() -> dict:
    cfg = load_physiology()
    sys_cfg = cfg.get("systemic", {})
    liver_cfg = cfg.get("liver", {})
    return {
        "P_sa":     float(sys_cfg.get("P_sa0", 85.0)),
        "P_portal": float(liver_cfg.get("P_portal0", 8.0)),
    }


GIT_CFG = load_git_cfg()
INPUTS_NORM = load_inputs()


def make_git() -> GITract:
    return GITract(**GIT_CFG)


def print_banner():
    print("=" * 70)
    print("GITract: параметры из config/physiology.yaml")
    print("=" * 70)
    for k, v in GIT_CFG.items():
        print(f"  {k:30s} = {v}")
    print(f"\n  Нормальные входы:")
    print(f"    P_sa     = {INPUTS_NORM['P_sa']}")
    print(f"    P_portal = {INPUTS_NORM['P_portal']}")
    print("=" * 70)


# =====================================================================
# Интеграция до稳态
# =====================================================================

def simulate_steady(git: GITract, inputs: dict, t_end=300.0,
                    method="LSODA", rtol=1e-8, atol=1e-10):
    y0 = git.get_initial_state()

    def rhs(t, y):
        return git.get_derivatives(t, y, inputs)

    return solve_ivp(rhs, (0.0, t_end), y0, method=method,
                     rtol=rtol, atol=atol, max_step=0.5)


def extract_steady(sol, git, inputs):
    t_end = sol.t[-1]
    win = sol.t > t_end * 0.75
    y_ss = sol.y[:, win].mean(axis=1)
    git.get_derivatives(t_end, y_ss, inputs)
    out = git.get_outputs(y_ss)
    out["P_art"] = float(y_ss[0])
    out["P_cap"] = float(y_ss[1])
    out["nfev"] = int(sol.nfev)
    return out


# =====================================================================
# TEST 1: Интерфейс
# =====================================================================

def test_interface():
    print("\n" + "=" * 70)
    print("TEST 1: Интерфейс OrganModel")
    print("=" * 70)
    git = make_git()
    n  = git.get_state_size()
    y0 = git.get_initial_state()
    print(f"  state_size    = {n}")
    print(f"  initial state = {y0}")
    print(f"  ожидание      = [{GIT_CFG['P_art0']}, {GIT_CFG['P_cap0']}]")
    ok = (n == 2
          and abs(y0[0] - GIT_CFG["P_art0"]) < 1e-9
          and abs(y0[1] - GIT_CFG["P_cap0"]) < 1e-9)
    print(f"  [{'OK' if ok else 'FAIL'}] state_size=2, init=[P_art0, P_cap0]")
    return {"ok": ok}


# =====================================================================
# TEST 2: Steady state vs аналитика
# =====================================================================

def test_steady_state():
    print("\n" + "=" * 70)
    print("TEST 2: Steady state vs аналитика")
    print("=" * 70)
    git = make_git()
    inputs = dict(INPUTS_NORM, intake_water=0.0, intake_nutrients=0.0)

    sol = simulate_steady(git, inputs)
    out = extract_steady(sol, git, inputs)

    R_total = GIT_CFG["R_art"] + GIT_CFG["R_cap"] + GIT_CFG["R_venous"]
    Q_ss     = (inputs["P_sa"] - inputs["P_portal"]) / R_total
    P_art_ex = inputs["P_sa"] - Q_ss * GIT_CFG["R_art"]
    P_cap_ex = P_art_ex - Q_ss * GIT_CFG["R_cap"]

    print(f"  {'':>8}  {'аналитика':>12}  {'модель':>12}")
    print(f"  {'Q_ss':>8}  {Q_ss:>12.4f}  {out['Q_out']:>12.4f}")
    print(f"  {'P_art':>8}  {P_art_ex:>12.4f}  {out['P_art']:>12.4f}")
    print(f"  {'P_cap':>8}  {P_cap_ex:>12.4f}  {out['P_cap']:>12.4f}")
    print(f"  nfev = {sol.nfev}")

    ok_Q = abs(out["Q_out"] - Q_ss) < 0.01
    ok_Pa = abs(out["P_art"] - P_art_ex) < 0.1
    ok_Pc = abs(out["P_cap"] - P_cap_ex) < 0.1
    print(f"\n  [{'OK' if ok_Q else 'FAIL'}] Q_out = аналитика")
    print(f"  [{'OK' if ok_Pa else 'FAIL'}] P_art = аналитика")
    print(f"  [{'OK' if ok_Pc else 'FAIL'}] P_cap = аналитика")
    return {"ok": ok_Q and ok_Pa and ok_Pc}


# =====================================================================
# TEST 3: Kirchhoff
# =====================================================================

def test_kirchhoff():
    print("\n" + "=" * 70)
    print("TEST 3: Kirchhoff Q_in = Q_cap = Q_out в稳态")
    print("=" * 70)
    git = make_git()
    inputs = dict(INPUTS_NORM, intake_water=0.0, intake_nutrients=0.0)
    sol = simulate_steady(git, inputs)

    t_end = sol.t[-1]
    win = sol.t > t_end * 0.75
    P_art = float(sol.y[0, win].mean())
    P_cap = float(sol.y[1, win].mean())

    Q_in  = (inputs["P_sa"]     - P_art) / GIT_CFG["R_art"]
    Q_cap = (P_art              - P_cap) / GIT_CFG["R_cap"]
    Q_out = (P_cap - inputs["P_portal"]) / GIT_CFG["R_venous"]

    print(f"  Q_in  = {Q_in:.6f}")
    print(f"  Q_cap = {Q_cap:.6f}")
    print(f"  Q_out = {Q_out:.6f}")
    print(f"  |Q_in − Q_cap|  = {abs(Q_in - Q_cap):.2e}")
    print(f"  |Q_cap − Q_out| = {abs(Q_cap - Q_out):.2e}")

    ok = abs(Q_in - Q_cap) < 1e-3 and abs(Q_cap - Q_out) < 1e-3
    print(f"  [{'OK' if ok else 'FAIL'}] Kirchhoff выполняется")
    return {"ok": ok}


# =====================================================================
# TEST 4: Drift
# =====================================================================

def test_drift():
    print("\n" + "=" * 70)
    print("TEST 4: Drift — два окна по 20 с в конце симуляции")
    print("=" * 70)
    git = make_git()
    inputs = dict(INPUTS_NORM, intake_water=0.0, intake_nutrients=0.0)
    sol = simulate_steady(git, inputs, t_end=300.0)

    w1 = (sol.t > 200.0) & (sol.t <= 220.0)
    w2 = (sol.t > 280.0) & (sol.t <= 300.0)
    Pa1, Pa2 = sol.y[0, w1].mean(), sol.y[0, w2].mean()
    Pc1, Pc2 = sol.y[1, w1].mean(), sol.y[1, w2].mean()
    dPa = abs(Pa2 - Pa1) / max(abs(Pa2), 1.0)
    dPc = abs(Pc2 - Pc1) / max(abs(Pc2), 1.0)

    print(f"  P_art: [200-220]={Pa1:.6f}  [280-300]={Pa2:.6f}  drift={dPa:.2e}")
    print(f"  P_cap: [200-220]={Pc1:.6f}  [280-300]={Pc2:.6f}  drift={dPc:.2e}")

    ok = dPa < 1e-3 and dPc < 1e-3
    print(f"  [{'OK' if ok else 'FAIL'}] drift < 1e-3")
    return {"ok": ok}


# =====================================================================
# TEST 5: portal_pressure_factor
# =====================================================================

def test_absorption_factor():
    print("\n" + "=" * 70)
    print("TEST 5: portal_pressure_factor (P≤8 → 1.0, P>8 → падает)")
    print("=" * 70)
    git = make_git()
    print(f"  {'P_portal':>10}  {'factor':>10}  {'ожидание':>10}  флаг")
    all_ok = True
    for P_portal in [5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
        f = git._absorption_factor(P_portal)
        exp = 1.0 if P_portal <= 8.0 else max(0.2, 1.0 - 0.02*(P_portal - 8.0))
        ok = abs(f - exp) < 1e-9
        all_ok = all_ok and ok
        print(f"  {P_portal:>10.1f}  {f:>10.4f}  {exp:>10.4f}  "
              f"[{'OK' if ok else 'FAIL'}]")
    print(f"  [{'OK' if all_ok else 'FAIL'}] portal_pressure_factor корректен")
    return {"ok": all_ok}


# =====================================================================
# TEST 6: Абсорбция
# =====================================================================

def test_absorption():
    print("\n" + "=" * 70)
    print("TEST 6: Абсорбция воды и нутриентов")
    print("=" * 70)
    git = make_git()
    inputs = dict(INPUTS_NORM, intake_water=0.5, intake_nutrients=1.0)

    y0 = git.get_initial_state()
    git.get_derivatives(0.0, y0, inputs)
    out = git.get_outputs(y0)

    factor = git._absorption_factor(inputs["P_portal"])
    exp_w = GIT_CFG["k_absorption_water"]     * inputs["intake_water"]     * factor
    exp_n = GIT_CFG["k_absorption_nutrients"] * inputs["intake_nutrients"] * factor

    print(f"  intake_water       = {inputs['intake_water']}")
    print(f"  intake_nutrients   = {inputs['intake_nutrients']}")
    print(f"  abs_factor         = {factor:.4f}")
    print(f"  absorption_water   = {out['absorption_water']:.6f}  ожидание {exp_w:.6f}")
    print(f"  absorption_nutr    = {out['absorption_nutrients']:.6f}  ожидание {exp_n:.6f}")

    ok_w = abs(out["absorption_water"] - exp_w) < 1e-9
    ok_n = abs(out["absorption_nutrients"] - exp_n) < 1e-9
    print(f"  [{'OK' if ok_w else 'FAIL'}] water = k_abs · intake · factor")
    print(f"  [{'OK' if ok_n else 'FAIL'}] nutrients = k_abs · intake · factor")
    return {"ok": ok_w and ok_n}


# =====================================================================
# TEST 7: Физиологические диапазоны
# =====================================================================

def test_physio_ranges():
    print("\n" + "=" * 70)
    print("TEST 7: Физиологические диапазоны (Gayton)")
    print("=" * 70)
    git = make_git()
    inputs = dict(INPUTS_NORM, intake_water=0.0, intake_nutrients=0.0)
    sol = simulate_steady(git, inputs)
    out = extract_steady(sol, git, inputs)

    checks = [
        ("Q_out    (мл/с)", out["Q_out"],           10.0, 25.0),
        ("P_art    (mmHg)", out["P_art"],           40.0, 80.0),
        ("P_cap    (mmHg)", out["P_cap"],           20.0, 45.0),
        ("P_portal (mmHg)", inputs["P_portal"],      5.0, 12.0),
    ]
    all_ok = True
    for name, val, lo, hi in checks:
        ok = lo <= val <= hi
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name:<20s} = {val:>8.3f}  "
              f"Gayton [{lo}, {hi}]")
    return {"ok": all_ok}


# =====================================================================
# TEST 8: RK45 vs LSODA + nfev
# =====================================================================

def test_integration_accuracy():
    print("\n" + "=" * 70)
    print("TEST 8: RK45 vs LSODA + nfev")
    print("=" * 70)
    inputs = dict(INPUTS_NORM, intake_water=0.0, intake_nutrients=0.0)

    git_R = make_git()
    sol_R = simulate_steady(git_R, inputs, method="RK45")
    out_R = extract_steady(sol_R, git_R, inputs)

    git_L = make_git()
    sol_L = simulate_steady(git_L, inputs, method="LSODA")
    out_L = extract_steady(sol_L, git_L, inputs)

    dPa = abs(out_R["P_art"] - out_L["P_art"])
    dPc = abs(out_R["P_cap"] - out_L["P_cap"])
    dQ  = abs(out_R["Q_out"] - out_L["Q_out"])

    print(f"  RK45:  P_art={out_R['P_art']:.6f}  P_cap={out_R['P_cap']:.6f}  "
          f"Q_out={out_R['Q_out']:.6f}  nfev={sol_R.nfev}")
    print(f"  LSODA: P_art={out_L['P_art']:.6f}  P_cap={out_L['P_cap']:.6f}  "
          f"Q_out={out_L['Q_out']:.6f}  nfev={sol_L.nfev}")
    print(f"  |ΔP_art|={dPa:.2e}  |ΔP_cap|={dPc:.2e}  |ΔQ|={dQ:.2e}")

    ok = dPa < 1e-3 and dPc < 1e-3 and dQ < 1e-3
    nfev_ok = sol_L.nfev < 5000
    print(f"  [{'OK' if ok else 'FAIL'}] RK45 ≈ LSODA")
    print(f"  [{'OK' if nfev_ok else 'FAIL'}] nfev(LSODA)={sol_L.nfev} < 5000")
    return {"ok": ok and nfev_ok, "nfev": int(sol_L.nfev)}


# =====================================================================
# Сводка
# =====================================================================

def summary(results):
    print("\n" + "=" * 70)
    print("СВОДКА: физиологичен ли GITract?")
    print("=" * 70)
    names = [
        "Интерфейс OrganModel",
        "Steady state vs аналитика",
        "Kirchhoff Q_in=Q_cap=Q_out",
        "Drift < 1e-3",
        "portal_pressure_factor",
        "Абсорбция воды/нутриентов",
        "Физиологические диапазоны",
        "RK45 vs LSODA + nfev",
    ]
    print()
    for name, r in zip(names, results):
        flag = "OK" if r["ok"] else "FAIL"
        print(f"  [{flag}] {name}")
    print()
    if all(r["ok"] for r in results):
        print("ВЫВОД: GITract корректен, не причина P_sa 65, EDV 69.")
    else:
        print("ВЫВОД: GITract требует правки — см. выше.")


if __name__ == "__main__":
    print_banner()
    r1 = test_interface()
    r2 = test_steady_state()
    r3 = test_kirchhoff()
    r4 = test_drift()
    r5 = test_absorption_factor()
    r6 = test_absorption()
    r7 = test_physio_ranges()
    r8 = test_integration_accuracy()
    summary([r1, r2, r3, r4, r5, r6, r7, r8])
    print("\n" + "=" * 70)
    print("Вход GITract: P_sa, P_portal, intake_water, intake_nutrients")
    print("Выход → whole_body: Q_out, absorption_water, absorption_nutrients")
    print("=" * 70)