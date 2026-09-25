#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_gas_exchange.py — изолированная проверка GasExchange.

Ключевое отличие от Baroreflex/Brain/BloodPool:
  GasExchange — АЛГЕБРАИЧЕСКИЙ орган. Нет состояний, нет ODE.
  → Нет nfev (нечего интегрировать)
  → Нет drift (нет динамики)
  → Критерии адаптированы:
      * round-trip P ↔ C для кривых Хилла и CO2
      * устойчивость к граничным входам (нет NaN/Inf)
      * физиологичность выходов при входах из physiology.yaml

Запуск:
    python tests/debug_gas_exchange.py
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


# =====================================================================
# 0. Загрузка конфигурации из physiology.yaml
# =====================================================================

def load_ge_cfg() -> dict:
    cfg = load_physiology()
    if "gas_exchange" not in cfg:
        raise RuntimeError("physiology.yaml: нет секции 'gas_exchange'")
    return dict(cfg["gas_exchange"])


def load_blood_defaults() -> dict:
    cfg = load_physiology()
    init = dict(cfg.get("blood", {}).get("initial_concentrations", {}))
    return {
        "C_v_O2":  float(init.get("oxygen", 0.15)),
        "C_v_CO2": float(init.get("co2",    0.52)),
    }


def load_target_co() -> float:
    cfg = load_physiology()
    return float(cfg.get("systemic", {}).get("target_CO", 83.0))


GE_CFG = load_ge_cfg()
BLOOD  = load_blood_defaults()
Q_P_NORM = load_target_co()

# Стандартный «здоровый» вход
INPUTS_HEALTHY = {
    "C_v_O2":  BLOOD["C_v_O2"],   # 0.15
    "C_v_CO2": BLOOD["C_v_CO2"],  # 0.52
    "Q_p":     Q_P_NORM,          # 83
    "Q_shunt": 0.0,
}


def make_ge() -> GasExchange:
    return GasExchange(**GE_CFG)


def print_banner():
    print("=" * 70)
    print("GasExchange: параметры из config/physiology.yaml")
    print("=" * 70)
    for k in ("P_alv_O2", "P_alv_CO2", "Hb", "P50", "n_hill",
              "alpha_O2", "C_CO2_offset", "k_CO2_slope"):
        print(f"  {k:<14s} = {GE_CFG[k]}")
    print()
    print("  Нормальные входы из blood.initial_concentrations:")
    print(f"    C_v_O2  = {BLOOD['C_v_O2']}")
    print(f"    C_v_CO2 = {BLOOD['C_v_CO2']}")
    print(f"    Q_p     = {Q_P_NORM} (target_CO)")
    print("=" * 70)


# =====================================================================
# TEST 1: Интерфейс OrganModel
# =====================================================================

def test_interface():
    print("\n" + "=" * 70)
    print("TEST 1: Интерфейс OrganModel (алгебраический орган)")
    print("=" * 70)
    ge = make_ge()

    n  = ge.get_state_size()
    y0 = ge.get_initial_state()
    dy = ge.get_derivatives(0.0, np.array([]), {})

    print(f"  get_state_size()      = {n}")
    print(f"  get_initial_state()   = {y0} (size {y0.size})")
    print(f"  get_derivatives(...)  = {dy} (size {dy.size})")

    ok = (n == 0 and y0.size == 0 and dy.size == 0)
    print(f"\n  [{'OK' if ok else 'FAIL'}] state_size=0, init=[], derivatives=[]")
    return {"ok": ok}


# =====================================================================
# TEST 2: Round-trip O₂ (кривая Хилла)
# =====================================================================

def test_hill_roundtrip():
    print("\n" + "=" * 70)
    print("TEST 2: Round-trip O₂ (P → C → P) — кривая Хилла")
    print("=" * 70)
    ge = make_ge()

    print(f"  {'P_in':>8}  {'SaO2':>8}  {'C_out':>10}  {'P_back':>10}  {'err':>10}")
    max_err_P = 0.0
    for P_in in [20, 40, 60, 80, 100, 150, 300]:
        Sa = ge._SaO2_from_P(P_in)
        C  = ge._C_O2_from_P(P_in)
        P_back = ge._P_O2_from_C(C)
        err = abs(P_back - P_in) / P_in
        max_err_P = max(max_err_P, err)
        print(f"  {P_in:>8}  {Sa:>8.4f}  {C:>10.5f}  {P_back:>10.4f}  {err:>10.2e}")

    # Монотонность SaO2
    Ps = np.linspace(10, 200, 100)
    S  = np.array([ge._SaO2_from_P(p) for p in Ps])
    monotonic = bool(np.all(np.diff(S) > 0))

    # Ожидаемые значения из эталона
    Sa_100 = ge._SaO2_from_P(100.0)
    C_100  = ge._C_O2_from_P(100.0)
    ok_sa = abs(Sa_100 - 0.973) < 0.01
    ok_c  = abs(C_100  - 0.199) < 0.005

    print(f"\n  SaO2(100) = {Sa_100:.4f}   ожидание ~0.973  [{'OK' if ok_sa else 'FAIL'}]")
    print(f"  C_O2(100) = {C_100:.4f}   ожидание ~0.199  [{'OK' if ok_c else 'FAIL'}]")
    print(f"  max err_P round-trip = {max_err_P:.2e}")
    print(f"  SaO2 монотонно растёт: {monotonic}")

    ok = (max_err_P < 0.02) and monotonic and ok_sa and ok_c
    print(f"\n  [{'OK' if ok else 'FAIL'}] Hill curve корректна")
    return {"ok": ok, "max_err_P": float(max_err_P)}


# =====================================================================
# TEST 3: Round-trip CO₂ (линейная)
# =====================================================================

def test_co2_roundtrip():
    print("\n" + "=" * 70)
    print("TEST 3: Round-trip CO₂ (P → C → P) — линейная")
    print("=" * 70)
    ge = make_ge()

    print(f"  {'P_in':>8}  {'C_out':>10}  {'P_back':>10}  {'err':>10}")
    max_err = 0.0
    for P_in in [20, 30, 40, 50, 60, 80]:
        C = ge._C_CO2_from_P(P_in)
        P_back = ge._P_CO2_from_C(C)
        err = abs(P_back - P_in) / P_in
        max_err = max(max_err, err)
        print(f"  {P_in:>8}  {C:>10.5f}  {P_back:>10.4f}  {err:>10.2e}")

    C_40 = ge._C_CO2_from_P(40.0)
    ok_c = abs(C_40 - 0.480) < 0.001

    print(f"\n  C_CO2(P=40) = {C_40:.4f}   ожидание 0.480  [{'OK' if ok_c else 'FAIL'}]")
    print(f"  max err round-trip = {max_err:.2e}")

    ok = max_err < 1e-10 and ok_c
    print(f"\n  [{'OK' if ok else 'FAIL'}] CO₂ линейная кривая точна")
    return {"ok": ok, "max_err": float(max_err)}


# =====================================================================
# TEST 4: Здоровый (Q_shunt = 0)
# =====================================================================

def test_healthy():
    print("\n" + "=" * 70)
    print("TEST 4: Здоровый (Q_shunt = 0, C_v_O2 = 0.15, Q_p = 83)")
    print("=" * 70)
    ge = make_ge()
    out = ge.compute_effects(**INPUTS_HEALTHY)

    for k in ("C_v_O2", "C_pv_O2", "C_a_O2", "C_v_CO2", "C_pv_CO2", "C_a_CO2",
              "SaO2", "P_a_O2", "P_v_O2", "P_v_CO2",
              "Qp_Qs", "shunt_fraction_R2L", "f_bypass",
              "O2_uptake", "CO2_removal"):
        print(f"  {k:>20s} = {out[k]:+.6f}")

    checks = [
        ("C_a_O2 = C_pv_O2 (нет шунта)",
         abs(out["C_a_O2"] - out["C_pv_O2"]) < 1e-12),
        ("SaO2 ∈ [0.95, 0.99]", 0.95 <= out["SaO2"] <= 0.99),
        ("P_a_O2 ≈ 100 mmHg",
         abs(out["P_a_O2"] - 100.0) < 5.0),
        ("Qp_Qs = 1.0", abs(out["Qp_Qs"] - 1.0) < 1e-9),
        ("shunt_fraction_R2L = 0", out["shunt_fraction_R2L"] == 0.0),
        ("f_bypass = 0", out["f_bypass"] == 0.0),
        ("O2_uptake > 0", out["O2_uptake"] > 0),
        ("CO2_removal > 0", out["CO2_removal"] > 0),
    ]
    print()
    all_ok = True
    for name, ok in checks:
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return {"ok": all_ok}


# =====================================================================
# TEST 5: L→R шунт (Q_shunt > 0)
# =====================================================================

def test_LR_shunt():
    print("\n" + "=" * 70)
    print("TEST 5: L→R шунт (Q_shunt = +30)")
    print("=" * 70)
    ge = make_ge()
    inputs = dict(INPUTS_HEALTHY); inputs["Q_shunt"] = +30.0
    out = ge.compute_effects(**inputs)

    Q_s_expected = 83.0 - 30.0  # = 53
    Qp_Qs_expected = 83.0 / Q_s_expected

    print(f"  Q_p     = {inputs['Q_p']}")
    print(f"  Q_shunt = {inputs['Q_shunt']} (L→R)")
    print(f"  Q_s     = {Q_s_expected}")
    print(f"  Qp_Qs   = {out['Qp_Qs']:.6f}  ожидание {Qp_Qs_expected:.6f}")
    print(f"  SaO2    = {out['SaO2']:.4f}")
    print(f"  C_a_O2  = {out['C_a_O2']:.6f}")

    checks = [
        ("Qp_Qs > 1", out["Qp_Qs"] > 1.0),
        ("Qp_Qs = Q_p/(Q_p-Q_shunt)",
         abs(out["Qp_Qs"] - Qp_Qs_expected) < 1e-9),
        ("SaO2 не снижается (L→R)", out["SaO2"] > 0.95),
        ("C_a_O2 = C_pv_O2", abs(out["C_a_O2"] - out["C_pv_O2"]) < 1e-12),
        ("shunt_fraction_R2L = 0", out["shunt_fraction_R2L"] == 0.0),
    ]
    print()
    all_ok = True
    for name, ok in checks:
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return {"ok": all_ok}


# =====================================================================
# TEST 6: R→L шунт (Q_shunt < 0)
# =====================================================================

def test_RL_shunt():
    print("\n" + "=" * 70)
    print("TEST 6: R→L шунт (Q_shunt = −30)")
    print("=" * 70)
    ge = make_ge()
    inputs = dict(INPUTS_HEALTHY); inputs["Q_shunt"] = -30.0
    out = ge.compute_effects(**inputs)

    Q_s = 83.0 - (-30.0)  # = 113
    f_bypass_expected = 30.0 / Q_s
    C_a_O2_expected = ((1 - f_bypass_expected) * out["C_pv_O2"]
                       + f_bypass_expected * out["C_v_O2"])

    print(f"  Q_p      = {inputs['Q_p']}")
    print(f"  Q_shunt  = {inputs['Q_shunt']} (R→L)")
    print(f"  Q_s      = {Q_s}")
    print(f"  f_bypass = {out['f_bypass']:.6f}  ожидание {f_bypass_expected:.6f}")
    print(f"  Qp_Qs    = {out['Qp_Qs']:.4f}")
    print(f"  C_a_O2   = {out['C_a_O2']:.6f}  ожидание {C_a_O2_expected:.6f}")
    print(f"  SaO2     = {out['SaO2']:.4f}")

    checks = [
        ("Qp_Qs < 1 (R→L)", out["Qp_Qs"] < 1.0),
        ("f_bypass = |Q_shunt|/Q_s",
         abs(out["f_bypass"] - f_bypass_expected) < 1e-9),
        ("SaO2 снизилась", out["SaO2"] < 0.95),
        ("C_a_O2 = mix(C_pv, C_v)",
         abs(out["C_a_O2"] - C_a_O2_expected) < 1e-9),
        ("shunt_fraction_R2L > 0", out["shunt_fraction_R2L"] > 0),
    ]
    print()
    all_ok = True
    for name, ok in checks:
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return {"ok": all_ok, "SaO2": out["SaO2"]}


# =====================================================================
# TEST 7: Тяжёлый Эйзенменгер (Q_shunt = −60)
# =====================================================================

def test_severe_eisenmenger():
    print("\n" + "=" * 70)
    print("TEST 7: Тяжёлый Эйзенменгер (Q_shunt = −60, Q_p = 80)")
    print("=" * 70)
    ge = make_ge()
    inputs = dict(INPUTS_HEALTHY)
    inputs["Q_p"]     = 80.0
    inputs["Q_shunt"] = -60.0
    out = ge.compute_effects(**inputs)

    print(f"  Q_p               = {inputs['Q_p']}")
    print(f"  Q_shunt           = {inputs['Q_shunt']}")
    print(f"  Q_s               = {inputs['Q_p'] - inputs['Q_shunt']}")
    print(f"  f_bypass          = {out['f_bypass']:.4f}")
    print(f"  SaO2              = {out['SaO2']*100:.2f} %")
    print(f"  C_a_O2            = {out['C_a_O2']:.4f}")
    print(f"  P_a_O2            = {out['P_a_O2']:.1f} mmHg")
    print(f"  Qp_Qs             = {out['Qp_Qs']:.4f}")
    print(f"  shunt_fraction_R2L= {out['shunt_fraction_R2L']*100:.1f} %")

    checks = [
        ("SaO2 < 90% (цианоз)",         out["SaO2"] < 0.90),
        ("P_a_O2 < 60 mmHg",            out["P_a_O2"] < 60.0),
        ("Qp_Qs < 1",                   out["Qp_Qs"] < 1.0),
        ("shunt_fraction_R2L > 0.4",    out["shunt_fraction_R2L"] > 0.4),
    ]
    print()
    all_ok = True
    for name, ok in checks:
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return {"ok": all_ok}


# =====================================================================
# TEST 8: Физиологические диапазоны (Gayton)
# =====================================================================

def test_physio_ranges():
    print("\n" + "=" * 70)
    print("TEST 8: Физиологические диапазоны (Gayton)")
    print("=" * 70)
    ge = make_ge()
    out = ge.compute_effects(**INPUTS_HEALTHY)

    checks = [
        ("C_a_O2  (мл/мл)",    out["C_a_O2"],    0.18, 0.22),
        ("C_v_O2  (мл/мл)",    out["C_v_O2"],    0.12, 0.18),
        ("C_a_CO2 (мл/мл)",    out["C_a_CO2"],   0.45, 0.52),
        ("C_v_CO2 (мл/мл)",    out["C_v_CO2"],   0.48, 0.56),
        ("SaO2",                out["SaO2"],      0.94, 0.99),
        ("P_a_O2  (mmHg)",     out["P_a_O2"],    80.0, 110.0),
        ("P_v_O2  (mmHg)",     out["P_v_O2"],    30.0, 50.0),
        ("P_v_CO2 (mmHg)",     out["P_v_CO2"],   40.0, 52.0),
        ("O2_uptake (мл/с)",   out["O2_uptake"], 3.0,  6.0),
        ("CO2_removal (мл/с)", out["CO2_removal"],3.0,  6.0),
    ]
    all_ok = True
    for name, val, lo, hi in checks:
        ok = lo <= val <= hi
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name:<22s} = {val:>8.4f}  "
              f"Gayton [{lo}, {hi}]")
    return {"ok": all_ok}


# =====================================================================
# TEST 9: Граничные случаи (нет NaN/Inf)
# =====================================================================

def test_edge_cases():
    print("\n" + "=" * 70)
    print("TEST 9: Граничные случаи (нет NaN/Inf)")
    print("=" * 70)
    ge = make_ge()

    cases = [
        ("Q_p → 0",       {"C_v_O2": 0.15, "C_v_CO2": 0.52, "Q_p": 1e-6,   "Q_shunt": 0.0}),
        ("Q_shunt = Q_p", {"C_v_O2": 0.15, "C_v_CO2": 0.52, "Q_p": 83.0,   "Q_shunt": -83.0}),
        ("Q_shunt > Q_p", {"C_v_O2": 0.15, "C_v_CO2": 0.52, "Q_p": 83.0,   "Q_shunt": -200.0}),
        ("C_v = 0",       {"C_v_O2": 0.0,  "C_v_CO2": 0.0,  "Q_p": 83.0,   "Q_shunt": 0.0}),
        ("C_v max",       {"C_v_O2": 0.25, "C_v_CO2": 1.0,  "Q_p": 83.0,   "Q_shunt": 0.0}),
    ]
    all_ok = True
    for label, inputs in cases:
        try:
            out = ge.compute_effects(**inputs)
            non_finite = [k for k, v in out.items() if not np.isfinite(v)]
            if non_finite:
                print(f"  [FAIL] {label:<18s} → NaN/Inf в {non_finite}")
                all_ok = False
            else:
                print(f"  [OK]   {label:<18s} → C_a_O2 = {out['C_a_O2']:.4f}, "
                      f"SaO2 = {out['SaO2']:.4f}")
        except Exception as e:
            print(f"  [FAIL] {label:<18s} → exception {type(e).__name__}: {e}")
            all_ok = False

    print(f"\n  [{'OK' if all_ok else 'FAIL'}] все граничные случаи "
          f"дают конечные значения")
    return {"ok": all_ok}


# =====================================================================
# СВОДКА
# =====================================================================

def summary(results: list):
    print("\n" + "=" * 70)
    print("СВОДКА: физиологичен ли GasExchange?")
    print("=" * 70)
    names = [
        "Интерфейс OrganModel (state_size=0)",
        "Round-trip O₂ (Hill)",
        "Round-trip CO₂ (линейная)",
        "Здоровый (Q_shunt=0)",
        "L→R шунт (Q_shunt>0)",
        "R→L шунт (Q_shunt<0)",
        "Тяжёлый Эйзенменгер (Q_shunt=−60)",
        "Физиологические диапазоны Gayton",
        "Граничные случаи (без NaN)",
    ]
    print()
    for name, r in zip(names, results):
        flag = "OK" if r["ok"] else "FAIL"
        print(f"  [{flag}] {name}")

    print()
    if all(r["ok"] for r in results):
        print("ВЫВОД: GasExchange корректен и готов к интеграции с whole_body.")
        print()
        print("Примечание: из-за отсутствия ODE критерии nfev и drift")
        print("не применимы. Заменены на round-trip и физические границы.")
    else:
        print("ВЫВОД: GasExchange требует правки — см. выше.")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    print_banner()
    r1 = test_interface()
    r2 = test_hill_roundtrip()
    r3 = test_co2_roundtrip()
    r4 = test_healthy()
    r5 = test_LR_shunt()
    r6 = test_RL_shunt()
    r7 = test_severe_eisenmenger()
    r8 = test_physio_ranges()
    r9 = test_edge_cases()
    summary([r1, r2, r3, r4, r5, r6, r7, r8, r9])
    print("\n" + "=" * 70)
    print("Вход GasExchange: C_v_O2, C_v_CO2, Q_p, Q_shunt.")
    print("Выход GasExchange → whole_body: C_a_O2, C_a_CO2, SaO2, P_a_O2,")
    print("                                Qp_Qs, shunt_fraction_R2L, O2_uptake.")
    print("=" * 70)