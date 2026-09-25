#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_brain.py — изолированная проверка Brain.

Состояние: [P_br, C_O2_tis, C_CO2_tis, C_lac_tis, C_amm_tis]
Входы: P_sa, P_sv, C_a_O2, C_a_CO2, C_lactate_blood, C_ammonia,
       V_blood, occlusion_factor.
Выходы: Q_br, VO2_brain, C_v_O2_brain, C_v_CO2_brain, R_eff, факторы ауторегуляции.

Запуск:
    python tests/debug_brain.py
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import solve_ivp

from brain import Brain
from physio_config import load_physiology


# =====================================================================
# 0. Конфигурация из physiology.yaml
# =====================================================================

def load_brain_cfg() -> dict:
    cfg = load_physiology()
    if "brain" not in cfg:
        raise RuntimeError("physiology.yaml: нет секции 'brain'")
    return dict(cfg["brain"])


BRAIN_CFG = load_brain_cfg()


def make_brain() -> Brain:
    return Brain(**BRAIN_CFG)


def print_banner():
    print("=" * 70)
    print("Brain: параметры из config/physiology.yaml")
    print("=" * 70)
    keys = ["R_base", "C", "P_autoreg", "CMRO2_target", "C_v_min",
            "max_extraction", "V_tissue", "RQ", "C_O2_critical"]
    for k in keys:
        if k in BRAIN_CFG:
            print(f"  {k:<16s} = {BRAIN_CFG[k]}")
    print("=" * 70)


# =====================================================================
# Стандартный вход
# =====================================================================

INPUTS_BASE = {
    "P_sa": 85.0,
    "P_sv": 6.0,
    "C_a_O2": 0.20,
    "C_a_CO2": 0.50,
    "C_lactate_blood": 0.10,
    "C_ammonia": 0.30,
    "V_blood": 5800.0,
    "occlusion_factor": 1.0,
}


def simulate_steady(brain: Brain, inputs: dict,
                    t_end: float = 300.0, rtol: float = 1e-8,
                    atol: float = 1e-10, method: str = "LSODA"):
    """Интегрирует изолированный Brain до稳态."""
    y0 = brain.get_initial_state()

    def rhs(t, y):
        return brain.get_derivatives(t, y, inputs)

    sol = solve_ivp(rhs, (0.0, t_end), y0, method=method,
                    rtol=rtol, atol=atol, max_step=0.5)
    return sol


def extract_steady(sol, brain: Brain, inputs: dict, win_frac: float = 0.25):
    """Извлекает稳态-значения из последних win_frac·t_end."""
    t_end = sol.t[-1]
    win = sol.t > t_end * (1.0 - win_frac)
    y_steady = sol.y[:, win].mean(axis=1)
    brain.get_derivatives(t_end, y_steady, inputs)
    out = brain.get_outputs(y_steady)
    out["P_br"] = float(y_steady[0])
    out["C_O2_tis"] = float(y_steady[1])
    out["C_CO2_tis"] = float(y_steady[2])
    out["C_lac_tis"] = float(y_steady[3])
    out["C_amm_tis"] = float(y_steady[4])
    out["nfev"] = sol.nfev
    return out


# =====================================================================
# TEST 1: Initial state sanity
# =====================================================================

def test_init_state():
    print("\n" + "=" * 70)
    print("TEST 1: Initial state sanity")
    print("=" * 70)
    brain = make_brain()
    y0 = brain.get_initial_state()
    expected = {
        "P_br": BRAIN_CFG["P0"],
        "C_O2_tis": BRAIN_CFG["C_a_O2_norm"],
        "C_CO2_tis": BRAIN_CFG["C_a_CO2_norm"],
        "C_lac_tis": BRAIN_CFG["C_lac_norm"],
        "C_amm_tis": BRAIN_CFG["C_amm_norm"],
    }
    names = ["P_br", "C_O2_tis", "C_CO2_tis", "C_lac_tis", "C_amm_tis"]
    all_ok = True
    for i, (name, exp_val) in enumerate(zip(names, expected.values())):
        val = float(y0[i])
        ok = abs(val - exp_val) < 1e-9
        all_ok = all_ok and ok
        flag = "OK" if ok else "FAIL"
        print(f"  y0[{i}] {name:>10s} = {val:.4f}   ожидание {exp_val:.4f}  [{flag}]")
    print(f"\n[{'OK' if all_ok else 'FAIL'}] начальное состояние соответствует yaml")
    return {"ok": all_ok}


# =====================================================================
# TEST 2: Steady-state hemodynamics
# =====================================================================

def test_steady_hemodynamics():
    print("\n" + "=" * 70)
    print("TEST 2: Steady-state hemodynamics (P_sa=85, P_sv=6)")
    print("=" * 70)
    brain = make_brain()
    sol = simulate_steady(brain, INPUTS_BASE, t_end=300.0)
    out = extract_steady(sol, brain, INPUTS_BASE)

    P_br_expected = (INPUTS_BASE["P_sa"] + INPUTS_BASE["P_sv"]) / 2.0
    Q_br_expected = (INPUTS_BASE["P_sa"] - P_br_expected) / out["R_eff"]

    print(f"  P_br      = {out['P_br']:.4f}   ожидание {P_br_expected:.4f}")
    print(f"  Q_br      = {out['Q_br']:.4f} мл/с   ожидание {Q_br_expected:.4f}")
    print(f"  Q_out     = {out['Q_out']:.4f} мл/с")
    print(f"  R_eff     = {out['R_eff']:.4f}")

    ok_P = abs(out["P_br"] - P_br_expected) < 0.05
    ok_Q = abs(out["Q_br"] - Q_br_expected) < 0.05
    ok_bal = abs(out["Q_br"] - out["Q_out"]) < 0.01

    print(f"\n  |P_br − (P_sa+P_sv)/2| = {abs(out['P_br']-P_br_expected):.4f}  [{'OK' if ok_P else 'FAIL'}]")
    print(f"  |Q_br − аналитика|     = {abs(out['Q_br']-Q_br_expected):.4f}  [{'OK' if ok_Q else 'FAIL'}]")
    print(f"  |Q_br − Q_out|         = {abs(out['Q_br']-out['Q_out']):.4f}  [{'OK' if ok_bal else 'FAIL'}]")

    return {"ok": ok_P and ok_Q and ok_bal, "Q_br": out["Q_br"],
            "P_br": out["P_br"], "R_eff": out["R_eff"]}


# =====================================================================
# TEST 3: Oxygen metabolism
# =====================================================================

def test_oxygen_metabolism():
    print("\n" + "=" * 70)
    print("TEST 3: O₂ metabolism")
    print("=" * 70)
    brain = make_brain()
    sol = simulate_steady(brain, INPUTS_BASE, t_end=300.0)
    out = extract_steady(sol, brain, INPUTS_BASE)

    VO2_target = BRAIN_CFG["CMRO2_target"]
    C_v_O2_expected = INPUTS_BASE["C_a_O2"] - VO2_target / out["Q_br"]
    CO2_prod_expected = VO2_target * BRAIN_CFG["RQ"]
    C_v_CO2_expected = INPUTS_BASE["C_a_CO2"] + CO2_prod_expected / out["Q_br"]

    print(f"  VO2_brain      = {out['VO2_brain']:.6f}   ожидание {VO2_target:.6f}")
    print(f"  extraction     = {out['extraction_used']:.6f}")
    print(f"  C_v_O2_brain   = {out['C_v_O2_brain']:.6f}   ожидание {C_v_O2_expected:.6f}")
    print(f"  C_v_CO2_brain  = {out['C_v_CO2_brain']:.6f}   ожидание {C_v_CO2_expected:.6f}")

    ok_VO2 = abs(out["VO2_brain"] - VO2_target) < 1e-4
    ok_CvO2 = abs(out["C_v_O2_brain"] - C_v_O2_expected) < 1e-3
    ok_CvCO2 = abs(out["C_v_CO2_brain"] - C_v_CO2_expected) < 1e-3

    print(f"\n  [{'OK' if ok_VO2 else 'FAIL'}] VO2_brain = CMRO2_target (по конструкции)")
    print(f"  [{'OK' if ok_CvO2 else 'FAIL'}] C_v_O2 = C_a_O2 − VO2/Q_br")
    print(f"  [{'OK' if ok_CvCO2 else 'FAIL'}] C_v_CO2 = C_a_CO2 + VO2·RQ/Q_br")

    return {"ok": ok_VO2 and ok_CvO2 and ok_CvCO2,
            "C_v_O2": out["C_v_O2_brain"], "C_v_CO2": out["C_v_CO2_brain"]}


# =====================================================================
# TEST 4: Autoregulation response to P_sa
# =====================================================================

def test_autoregulation():
    print("\n" + "=" * 70)
    print("TEST 4: Autoregulation (P_sa: 60 → 85 → 120)")
    print("=" * 70)
    rows = []
    for P_sa in [60.0, 85.0, 120.0]:
        inputs = dict(INPUTS_BASE)
        inputs["P_sa"] = P_sa
        brain = make_brain()
        sol = simulate_steady(brain, inputs, t_end=200.0)
        out = extract_steady(sol, brain, inputs)
        rows.append({
            "P_sa": P_sa, "R_eff": out["R_eff"],
            "Q_br": out["Q_br"], "f_P": out["f_P_myogenic"],
            "P_br": out["P_br"],
        })

    print(f"  {'P_sa':>6}  {'R_eff':>8}  {'Q_br':>8}  {'f_P':>8}  {'P_br':>8}")
    for r in rows:
        print(f"  {r['P_sa']:>6.0f}  {r['R_eff']:>8.3f}  {r['Q_br']:>8.3f}  "
              f"{r['f_P']:>8.4f}  {r['P_br']:>8.3f}")

    R_lo, R_hi = rows[0]["R_eff"], rows[2]["R_eff"]
    Q_lo, Q_hi = rows[0]["Q_br"], rows[2]["Q_br"]
    ok_R = R_hi > R_lo
    ok_Q = (Q_hi / Q_lo) < 1.5  # автoreg держит Q_br в пределах 1.5x
    print(f"\n  R_eff(P_sa=120)/R_eff(P_sa=60) = {R_hi/R_lo:.3f}  [{'OK' if ok_R else 'FAIL'}]")
    print(f"  Q_br(P_sa=120)/Q_br(P_sa=60)   = {Q_hi/Q_lo:.3f}  "
          f"[{'OK' if ok_Q else 'FAIL'}]  (частичная ауторегуляция)")

    return {"ok": ok_R and ok_Q}


# =====================================================================
# TEST 5: Hypoxia → vasodilation
# =====================================================================

def test_hypoxia():
    print("\n" + "=" * 70)
    print("TEST 5: Гипоксия (C_a_O2: 0.20 → 0.12) → вазодилатация")
    print("=" * 70)
    brain_norm = make_brain()
    sol_norm = simulate_steady(brain_norm, INPUTS_BASE, t_end=200.0)
    out_norm = extract_steady(sol_norm, brain_norm, INPUTS_BASE)

    inputs_hyp = dict(INPUTS_BASE)
    inputs_hyp["C_a_O2"] = 0.12
    brain_hyp = make_brain()
    sol_hyp = simulate_steady(brain_hyp, inputs_hyp, t_end=200.0)
    out_hyp = extract_steady(sol_hyp, brain_hyp, inputs_hyp)

    print(f"  Norm:  f_O2 = {out_norm['f_O2_autoreg']:.4f}  "
          f"R_eff = {out_norm['R_eff']:.3f}  Q_br = {out_norm['Q_br']:.3f}")
    print(f"  Hyp:   f_O2 = {out_hyp['f_O2_autoreg']:.4f}  "
          f"R_eff = {out_hyp['R_eff']:.3f}  Q_br = {out_hyp['Q_br']:.3f}")

    ok_f = out_hyp["f_O2_autoreg"] < out_norm["f_O2_autoreg"]
    ok_R = out_hyp["R_eff"] < out_norm["R_eff"]
    ok_Q = out_hyp["Q_br"] > out_norm["Q_br"]
    print(f"\n  [{'OK' if ok_f else 'FAIL'}] f_O2 снизился (вазодилатация)")
    print(f"  [{'OK' if ok_R else 'FAIL'}] R_eff снизился")
    print(f"  [{'OK' if ok_Q else 'FAIL'}] Q_br вырос")

    return {"ok": ok_f and ok_R and ok_Q}


# =====================================================================
# TEST 6: Hypercapnia → vasodilation
# =====================================================================

def test_hypercapnia():
    print("\n" + "=" * 70)
    print("TEST 6: Гиперкапния (C_a_CO2: 0.50 → 0.55) → вазодилатация")
    print("=" * 70)
    brain_norm = make_brain()
    sol_norm = simulate_steady(brain_norm, INPUTS_BASE, t_end=200.0)
    out_norm = extract_steady(sol_norm, brain_norm, INPUTS_BASE)

    inputs_hyp = dict(INPUTS_BASE)
    inputs_hyp["C_a_CO2"] = 0.55
    brain_hyp = make_brain()
    sol_hyp = simulate_steady(brain_hyp, inputs_hyp, t_end=200.0)
    out_hyp = extract_steady(sol_hyp, brain_hyp, inputs_hyp)

    print(f"  Norm:  f_CO2 = {out_norm['f_CO2_autoreg']:.4f}  "
          f"R_eff = {out_norm['R_eff']:.3f}  Q_br = {out_norm['Q_br']:.3f}")
    print(f"  HC:    f_CO2 = {out_hyp['f_CO2_autoreg']:.4f}  "
          f"R_eff = {out_hyp['R_eff']:.3f}  Q_br = {out_hyp['Q_br']:.3f}")

    ok = (out_hyp["f_CO2_autoreg"] < out_norm["f_CO2_autoreg"]
          and out_hyp["R_eff"] < out_norm["R_eff"]
          and out_hyp["Q_br"] > out_norm["Q_br"])
    print(f"\n  [{'OK' if ok else 'FAIL'}] гиперкапния → вазодилатация → рост Q_br")
    return {"ok": ok}


# =====================================================================
# TEST 7: Occlusion
# =====================================================================

def test_occlusion():
    print("\n" + "=" * 70)
    print("TEST 7: Occlusion factor (1.0 / 0.5 / 0.0)")
    print("=" * 70)
    rows = []
    for occl in [1.0, 0.5, 0.0]:
        inputs = dict(INPUTS_BASE)
        inputs["occlusion_factor"] = occl
        brain = make_brain()
        sol = simulate_steady(brain, inputs, t_end=200.0)
        out = extract_steady(sol, brain, inputs)
        rows.append({"occl": occl, "Q_br": out["Q_br"],
                     "C_O2_tis": out["C_O2_tis"],
                     "inhib_O2": out["inhib_O2"]})

    print(f"  {'occl':>6}  {'Q_br':>8}  {'C_O2_tis':>10}  {'inhib_O2':>10}")
    for r in rows:
        print(f"  {r['occl']:>6.1f}  {r['Q_br']:>8.3f}  "
              f"{r['C_O2_tis']:>10.4f}  {r['inhib_O2']:>10.4f}")

    ok_1 = abs(rows[0]["Q_br"] / max(rows[1]["Q_br"], 1e-6) - 2.0) < 0.1
    ok_2 = rows[2]["Q_br"] < 0.01
    print(f"\n  [{'OK' if ok_1 else 'FAIL'}] occl=0.5 → Q_br ~ половина от occl=1.0")
    print(f"  [{'OK' if ok_2 else 'FAIL'}] occl=0.0 → Q_br ≈ 0")

    return {"ok": ok_1 and ok_2}


# =====================================================================
# TEST 8: Physiological ranges (Gayton-scaled)
# =====================================================================

def test_physio_ranges():
    print("\n" + "=" * 70)
    print("TEST 8: Физиологические диапазоны (Gayton-scaled)")
    print("=" * 70)
    brain = make_brain()
    sol = simulate_steady(brain, INPUTS_BASE, t_end=300.0)
    out = extract_steady(sol, brain, INPUTS_BASE)

    # Расчёт P_vO2 из C_v_O2 через Hill
    P50, n_hill = 26.8, 2.7
    C_max = 1.34 * 15.0 / 100.0
    Sa = np.clip(out["C_v_O2_brain"] / C_max, 1e-6, 0.999)
    P_vO2 = P50 * (Sa / (1 - Sa)) ** (1 / n_hill)

    checks = [
        ("Q_br",         out["Q_br"],         4.0,  7.0),
        ("P_br",         out["P_br"],        40.0, 55.0),
        ("VO2_brain",    out["VO2_brain"],    0.45, 0.65),
        ("C_v_O2_brain", out["C_v_O2_brain"], 0.08, 0.15),
        ("C_v_CO2_brain",out["C_v_CO2_brain"],0.52, 0.62),
        ("C_O2_tis",     out["C_O2_tis"],     0.06, 0.14),
        ("C_lac_tis",    out["C_lac_tis"],    0.5,  1.2),
        ("C_amm_tis",    out["C_amm_tis"],    0.2,  0.4),
    ]
    all_ok = True
    for name, val, lo, hi in checks:
        ok = lo <= val <= hi
        all_ok = all_ok and ok
        flag = "OK" if ok else "FAIL"
        print(f"  {name:<16s} = {val:>8.4f}   Gayton [{lo}, {hi}]  [{flag}]")

    print(f"\n  P_vO2 (derived) = {P_vO2:.1f} mmHg   Gayton [25, 45]")
    return {"ok": all_ok, "P_vO2": float(P_vO2)}


# =====================================================================
# TEST 9: RK45 vs LSODA + nfev
# =====================================================================

def test_integration_accuracy():
    print("\n" + "=" * 70)
    print("TEST 9: RK45 vs LSODA + nfev")
    print("=" * 70)
    brain_R = make_brain()
    sol_R = simulate_steady(brain_R, INPUTS_BASE, t_end=200.0,
                            method="RK45", rtol=1e-8, atol=1e-10)
    out_R = extract_steady(sol_R, brain_R, INPUTS_BASE)

    brain_L = make_brain()
    sol_L = simulate_steady(brain_L, INPUTS_BASE, t_end=200.0,
                            method="LSODA", rtol=1e-8, atol=1e-10)
    out_L = extract_steady(sol_L, brain_L, INPUTS_BASE)

    diff_Q = abs(out_R["Q_br"] - out_L["Q_br"])
    diff_P = abs(out_R["P_br"] - out_L["P_br"])

    print(f"  RK45:  Q_br = {out_R['Q_br']:.8f}  P_br = {out_R['P_br']:.8f}  nfev = {sol_R.nfev}")
    print(f"  LSODA: Q_br = {out_L['Q_br']:.8f}  P_br = {out_L['P_br']:.8f}  nfev = {sol_L.nfev}")
    print(f"  |ΔQ_br| = {diff_Q:.2e}  |ΔP_br| = {diff_P:.2e}")

    ok = diff_Q < 1e-3 and diff_P < 1e-3
    nfev_ok = sol_L.nfev < 20000
    print(f"\n  [{'OK' if ok else 'FAIL'}] RK45 ≈ LSODA")
    print(f"  [{'OK' if nfev_ok else 'FAIL'}] nfev(LSODA) = {sol_L.nfev} < 20000")
    return {"ok": ok and nfev_ok, "nfev": int(sol_L.nfev)}


# =====================================================================
# Сводка
# =====================================================================

def summary(results: list):
    print("\n" + "=" * 70)
    print("СВОДКА: физиологичен ли Brain?")
    print("=" * 70)
    names = [
        "Initial state sanity",
        "Steady-state hemodynamics",
        "O₂ metabolism",
        "Autoregulation (P_sa)",
        "Hypoxia response",
        "Hypercapnia response",
        "Occlusion",
        "Physiological ranges",
        "RK45 vs LSODA + nfev",
    ]
    print()
    for name, r in zip(names, results):
        flag = "OK" if r["ok"] else "FAIL"
        print(f"  [{flag}] {name}")

    print()
    if all(r["ok"] for r in results):
        print("ВЫВОД: Brain корректен и готов к интеграции с whole_body.")
    else:
        print("ВЫВОД: Brain требует правки — см. выше.")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    print_banner()
    r1 = test_init_state()
    r2 = test_steady_hemodynamics()
    r3 = test_oxygen_metabolism()
    r4 = test_autoregulation()
    r5 = test_hypoxia()
    r6 = test_hypercapnia()
    r7 = test_occlusion()
    r8 = test_physio_ranges()
    r9 = test_integration_accuracy()
    summary([r1, r2, r3, r4, r5, r6, r7, r8, r9])
    print("\n" + "=" * 70)
    print("Вход Brain: P_sa, P_sv, C_a_O2, C_a_CO2, C_lactate_blood,")
    print("            C_ammonia, V_blood, occlusion_factor.")
    print("Выход Brain → whole_body: Q_br, VO2_brain, C_v_O2_brain,")
    print("                          C_v_CO2_brain, C_v_lactate_brain.")
    print("=" * 70)