#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_jugular_vein_1.py — изолированная проверка JugularVein (после Option A).

Запуск:  python tests/debug_jugular_vein_v2_1.py
Отчёт:   tests/results_debug_jugular_vein_v2_1.txt

Что проверяется:
  1. Интерфейс OrganModel
  2. YAML-консистентность (target_fraction/tau_target — deprecated)
  3. Steady state + аналитика V_ss = V0 + C·(Q_in·R_out + P_sv − P0)
  4. Трекинг газов (τ_gas = V_ss/Q_in)
  5. Отклик на Q_in
  6. Отклик на P_sv (V_jv РАСТЁТ с P_sv — новая семантика Option A)
  7. Mass balance: Q_in = Q_out в стационаре
  8. Мягкий пол
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

from jugular_vein import JugularVein
from physio_config import load_physiology

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_jugular_vein_v2_1.txt"


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, d):
        for s in self.s: s.write(d); s.flush()
    def flush(self):
        for s in self.s: s.flush()


CFG       = load_physiology()
JUG_CFG   = dict(CFG.get("jugular_vein", {}))

_JUG_KNOWN_KEYS = {"C", "P0", "V0", "R_out",
                   "C_O2_init", "C_CO2_init", "Hb"}
_JUG_DEPRECATED_KEYS = {"target_fraction", "tau_target"}


def make_jv() -> JugularVein:
    """Собирает JugularVein, отфильтровывая deprecated-ключи YAML."""
    kwargs = {k: v for k, v in JUG_CFG.items() if k in _JUG_KNOWN_KEYS}
    return JugularVein(**kwargs)


INPUTS_STD = {
    "Q_in":     12.0,    # мл/с, ~720 мл/мин мозгового кровотока
    "C_in_O2":  0.12,
    "C_in_CO2": 0.56,
    "P_sv":     5.0,
    "V_blood":  5800.0,  # больше НЕ читается внутри; оставлен для совместимости входов
}


def integrate(jv, inputs, t_end=600.0, max_step=0.05, rtol=1e-7, atol=1e-9):
    y0 = jv.get_initial_state()
    rhs = lambda t, y: jv.get_derivatives(t, y, inputs)
    return solve_ivp(rhs, (0.0, t_end), y0, method="LSODA",
                     rtol=rtol, atol=atol, max_step=max_step)


def out_at(jv, y, inputs):
    jv.get_derivatives(0.0, y, inputs)
    return jv.get_outputs(y)


# ---------------------------------------------------------------------
def test_interface():
    print("\n" + "=" * 70)
    print("TEST 1: Интерфейс OrganModel")
    print("=" * 70)
    jv = make_jv()
    sz = jv.get_state_size()
    y0 = jv.get_initial_state()
    d0 = jv.get_derivatives(0.0, y0, INPUTS_STD)
    print(f"  state_size = {sz}   (ожидание 3)")
    print(f"  y0         = {y0}")
    print(f"  dy/dt(0)   = {d0}")
    ok = (sz == 3 and y0.size == 3 and d0.size == 3
          and np.all(np.isfinite(y0)) and np.all(np.isfinite(d0)))
    print(f"  [{'OK' if ok else 'FAIL'}] интерфейс корректен")
    return {"ok": ok}


# ---------------------------------------------------------------------
def test_yaml_consistency():
    print("\n" + "=" * 70)
    print("TEST 2: YAML-консистентность (Option A)")
    print("=" * 70)
    jv = make_jv()
    checks = [
        ("V0",    jv.V0,    JUG_CFG.get("V0", 150.0)),
        ("C",     jv.C,     JUG_CFG.get("C", 20.0)),
        ("P0",    jv.P0,    JUG_CFG.get("P0", 6.0)),
        ("R_out", jv.R_out, JUG_CFG.get("R_out", 0.5)),
    ]
    ok = True
    for name, actual, expected in checks:
        same = abs(actual - expected) < 1e-9
        ok = ok and same
        print(f"  {name:8s}: class={actual:<8.2f}  yaml={expected:<8.2f}  [{'OK' if same else 'FAIL'}]")
    # Option A: параметры relaxation удалены
    has_tau = hasattr(jv, "tau_target")
    has_tf  = hasattr(jv, "target_fraction")
    print(f"  hasattr(jv, 'tau_target')     = {has_tau}  (ожидание False)")
    print(f"  hasattr(jv, 'target_fraction') = {has_tf}   (ожидание False)")
    ok = ok and not has_tau and not has_tf
    # deprecated-ключи YAML больше не пробрасываются в конструктор
    dep_present = [k for k in _JUG_DEPRECATED_KEYS if k in JUG_CFG]
    print(f"  deprecated-ключи в physiology.yaml: {dep_present or 'нет'}")
    if dep_present:
        print(f"  → make_jv() их отфильтровывает; это ожидаемо и безопасно.")
    print(f"  [{'OK' if ok else 'FAIL'}] параметры и семантика соответствуют Option A")
    return {"ok": ok}


# ---------------------------------------------------------------------
def test_steady_state():
    print("\n" + "=" * 70)
    print("TEST 3: Steady state + аналитика (без relaxation)")
    print("=" * 70)
    jv = make_jv()
    sol = integrate(jv, INPUTS_STD, t_end=600.0)
    y_ss = sol.y[:, -1]
    o = out_at(jv, y_ss, INPUTS_STD)

    # После Option A: dV/dt = Q_in − Q_out, Q_out = (P_jv − P_sv)/R_out
    # В стационаре Q_in = Q_out → V_ss = V0 + C·(Q_in·R_out + P_sv − P0)
    C, P0, V0, R_out = jv.C, jv.P0, jv.V0, jv.R_out
    Q_in, P_sv = INPUTS_STD["Q_in"], INPUTS_STD["P_sv"]
    V_pred = V0 + C * (Q_in * R_out + P_sv - P0)
    P_pred = P0 + (V_pred - V0) / C
    Qout_pred = (P_pred - P_sv) / R_out

    print(f"  Численное (t=600 с):")
    print(f"    V_jv   = {o['V_jv']:8.2f} мл")
    print(f"    P_jv   = {o['P_jv']:8.2f} мм рт.ст.")
    print(f"    Q_out  = {o['Q_jv_out']:8.3f} мл/с")
    print(f"    SjvO2  = {o['SjvO2']*100:.1f} %")
    print(f"  Аналитика (Q_in = Q_out):")
    print(f"    V_ss*  = {V_pred:8.2f} мл")
    print(f"    P_ss*  = {P_pred:8.2f} мм рт.ст.")
    print(f"    Qout*  = {Qout_pred:8.3f} мл/с")

    checks = [
        ("V_jv в [200, 700] мл",         200 < o["V_jv"] < 700),
        ("P_jv в [5, 30] мм рт.ст.",      5 < o["P_jv"] < 30),
        ("Q_out ≈ Q_in (±0.5)",          abs(o["Q_jv_out"] - Q_in) < 0.5),
        ("SjvO2 в [55, 75] %",           0.55 < o["SjvO2"] < 0.75),
        ("V_num vs V_pred < 2 %",        abs(o["V_jv"] - V_pred) / V_pred < 0.02),
        ("P_num vs P_pred < 2 %",        abs(o["P_jv"] - P_pred) / max(P_pred, 1e-6) < 0.02),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")
    return {"ok": ok}


# ---------------------------------------------------------------------
def test_gas_tracking():
    print("\n" + "=" * 70)
    print("TEST 4: Трекинг газов: C_jv → C_in")
    print("=" * 70)
    jv = make_jv()
    sol = integrate(jv, INPUTS_STD, t_end=600.0)
    C_O2 = sol.y[1, :]; C_CO2 = sol.y[2, :]; t = sol.t

    # τ_gas = V_ss/Q_in, где V_ss = V0 + C·(Q_in·R_out + P_sv − P0)
    V_ss = jv.V0 + jv.C * (INPUTS_STD["Q_in"] * jv.R_out + INPUTS_STD["P_sv"] - jv.P0)
    tau_gas = V_ss / max(INPUTS_STD["Q_in"], 1e-9)
    print(f"  τ_gas = V_ss/Q_in = {tau_gas:.1f} с  (V_ss={V_ss:.1f})")

    for tt in (0, tau_gas, 3 * tau_gas, 5 * tau_gas, 600):
        i = int(np.argmin(np.abs(t - tt)))
        print(f"    t={tt:5.0f}с  C_O2={C_O2[i]:.4f}  C_CO2={C_CO2[i]:.4f}")

    rel_O2  = abs(C_O2[-1]  - INPUTS_STD["C_in_O2"])  / INPUTS_STD["C_in_O2"]
    rel_CO2 = abs(C_CO2[-1] - INPUTS_STD["C_in_CO2"]) / INPUTS_STD["C_in_CO2"]
    print(f"  rel отклонение: O2={rel_O2:.2e}, CO2={rel_CO2:.2e}")
    ok = rel_O2 < 0.01 and rel_CO2 < 0.01
    print(f"  [{'OK' if ok else 'FAIL'}] C_jv сходится к C_in (< 1%)")
    return {"ok": ok}


# ---------------------------------------------------------------------
def test_qin_step():
    print("\n" + "=" * 70)
    print("TEST 5: Отклик на шаг Q_in (washout)")
    print("=" * 70)
    jv = make_jv()
    y0 = np.array([jv.V0, 0.15, 0.56])
    def rhs(t, y): return jv.get_derivatives(t, y, INPUTS_STD)
    sol = solve_ivp(rhs, (0, 300), y0, method="LSODA",
                    rtol=1e-7, atol=1e-9, max_step=0.05)
    target = 0.15 + (0.10 - 0.15) * 0.632
    idx = int(np.argmin(np.abs(sol.y[1] - target)))
    tau_obs = sol.t[idx]
    V_ss = jv.V0 + jv.C * (INPUTS_STD["Q_in"] * jv.R_out + INPUTS_STD["P_sv"] - jv.P0)
    tau_theor = V_ss / INPUTS_STD["Q_in"]
    print(f"  τ наблюдаемая = {tau_obs:.1f} с, теория V_ss/Q_in = {tau_theor:.1f} с")
    ok = 0.5 < tau_obs / tau_theor < 2.0
    print(f"  [{'OK' if ok else 'FAIL'}] τ в пределах 2× от V_ss/Q_in")
    return {"ok": ok}


# ---------------------------------------------------------------------
def test_psv_response():
    print("\n" + "=" * 70)
    print("TEST 6: Отклик на P_sv (Option A: V_jv РАСТЁТ с P_sv)")
    print("=" * 70)
    rows = []
    for P_sv in (2.0, 5.0, 10.0, 15.0):
        inp = dict(INPUTS_STD, P_sv=P_sv)
        jv = make_jv()
        sol = integrate(jv, inp, t_end=400.0)
        o = out_at(jv, sol.y[:, -1], inp)
        rows.append((P_sv, o["V_jv"], o["P_jv"], o["Q_jv_out"]))
        print(f"  P_sv={P_sv:5.1f}  →  V_jv={o['V_jv']:7.2f}  "
              f"P_jv={o['P_jv']:6.2f}  Q_out={o['Q_jv_out']:6.2f}")
    vs = [r[1] for r in rows]
    ps = [r[2] for r in rows]
    qs = [r[3] for r in rows]
    # V_jv монотонно растёт (V_ss = V0 + C·(Q_in·R_out + P_sv − P0))
    ok_v = all(vs[i] < vs[i+1] for i in range(len(vs)-1))
    # P_jv = P_sv + Q_in·R_out, тоже растёт ровно на ΔP_sv
    ok_p = all(ps[i] < ps[i+1] for i in range(len(ps)-1))
    # Q_out = Q_in во всех стационарах
    ok_q = all(abs(q - INPUTS_STD["Q_in"]) < 0.5 for q in qs)
    ok = ok_v and ok_p and ok_q
    print(f"  [{'OK' if ok_v else 'FAIL'}] V_jv монотонно растёт с P_sv")
    print(f"  [{'OK' if ok_p else 'FAIL'}] P_jv монотонно растёт с P_sv")
    print(f"  [{'OK' if ok_q else 'FAIL'}] Q_out = Q_in (steady) для всех P_sv")
    return {"ok": ok}


# ---------------------------------------------------------------------
def test_mass_balance():
    print("\n" + "=" * 70)
    print("TEST 7: Mass balance: Q_in = Q_out в стационаре")
    print("=" * 70)
    jv = make_jv()
    sol = integrate(jv, INPUTS_STD, t_end=600.0)
    y_ss = sol.y[:, -1]
    d = jv.get_derivatives(sol.t[-1], y_ss, INPUTS_STD)
    o = out_at(jv, y_ss, INPUTS_STD)
    print(f"  dV/dt       = {d[0]:+.3e} мл/с   (ожидание |·| < 1e-3)")
    print(f"  Q_in        = {INPUTS_STD['Q_in']:.4f} мл/с")
    print(f"  Q_out       = {o['Q_jv_out']:.4f} мл/с")
    print(f"  Q_in − Q_out = {INPUTS_STD['Q_in'] - o['Q_jv_out']:+.3e} мл/с")
    ok = abs(d[0]) < 1e-3 and abs(INPUTS_STD["Q_in"] - o["Q_jv_out"]) < 1e-3
    print(f"  [{'OK' if ok else 'FAIL'}] баланс сошёлся")
    return {"ok": ok}


# ---------------------------------------------------------------------
def test_soft_floor():
    print("\n" + "=" * 70)
    print("TEST 8: Мягкий пол (V не уходит в минус)")
    print("=" * 70)
    jv = make_jv()
    bad = {"Q_in": 0.0, "C_in_O2": 0.10, "C_in_CO2": 0.56,
           "P_sv": 30.0, "V_blood": 2000.0}
    sol = integrate(jv, bad, t_end=2000.0)
    V = sol.y[0, :]
    print(f"  V_min   = {V.min():.3f} мл  (0.5·V0 = {0.5*jv.V0:.1f})")
    print(f"  V_final = {V[-1]:.3f} мл")
    ok = V.min() > 0.0 and np.all(np.isfinite(V))
    print(f"  [{'OK' if ok else 'FAIL'}] V > 0 всюду, конечно")
    return {"ok": ok}


# ---------------------------------------------------------------------
def summary(results):
    print("\n" + "=" * 70)
    print("СВОДКА")
    print("=" * 70)
    names = ["Интерфейс", "YAML-консистентность", "Steady state",
             "Трекинг газов", "Отклик Q_in", "Отклик P_sv",
             "Mass balance", "Мягкий пол"]
    for n, r in zip(names, results):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    if all(r["ok"] for r in results):
        print("\nВЫВОД: JugularVein физиологичен в изоляции (Option A).")
        print("V_ss = V0 + C·(Q_in·R_out + P_sv − P0); Q_out = Q_in; ")
        print("V_jv не зависит от V_blood — relaxation удалён.")
    else:
        print("\nВЫВОД: см. FAIL выше.")


def run_all(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    orig = sys.stdout
    sys.stdout = Tee(orig, buf)
    try:
        print("=" * 70)
        print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("JugularVein: изолированный тест (v2.1, Option A)")
        print("=" * 70)
        print(f"  YAML-параметры (jugular_vein): {JUG_CFG}")
        print(f"  Фильтрованные kwargs        : "
              f"{ {k: v for k, v in JUG_CFG.items() if k in _JUG_KNOWN_KEYS} }")
        print(f"  Входы: {INPUTS_STD}")
        print("=" * 70)
        rs = [
            test_interface(),
            test_yaml_consistency(),
            test_steady_state(),
            test_gas_tracking(),
            test_qin_step(),
            test_psv_response(),
            test_mass_balance(),
            test_soft_floor(),
        ]
        summary(rs)
        print(f"\nФиниш: {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout = orig
    path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Отчёт: {path}")


if __name__ == "__main__":
    run_all(RESULT_FILE)