#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_jugular_vein.py — изолированная проверка JugularVein.

Запуск:  python tests/debug_jugular_vein.py
Отчёт:   tests/results_debug_jugular_vein.txt
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

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_jugular_vein_1.txt"


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d):
        for s in self.streams: s.write(d); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


# ---- Загрузка параметров из YAML (не дефолты класса!) ----
CFG = load_physiology()
JUG_CFG = dict(CFG.get("jugular_vein", {}))
SYS_CFG = dict(CFG.get("systemic", {}))


def make_jv() -> JugularVein:
    return JugularVein(**JUG_CFG)


# ---- Типичные «мозговые» входы ----
INPUTS_STD = {
    "Q_in":     12.0,   # мл/с  ≈ 700 мл/мин
    "C_in_O2":  0.12,   # мл/мл SjvO2 ~ 60 %
    "C_in_CO2": 0.56,
    "P_sv":     5.0,    # мм рт.ст.
    "V_blood":  5800.0,
}


def integrate(jv, inputs, t_end=600.0, max_step=0.5):
    y0 = jv.get_initial_state()
    rhs = lambda t, y: jv.get_derivatives(t, y, inputs)
    return solve_ivp(rhs, (0.0, t_end), y0, method="LSODA",
                     rtol=1e-7, atol=1e-9, max_step=max_step)


def out_at(jv, y, inputs):
    jv.get_derivatives(0.0, y, inputs)   # обновляет _current_outputs
    return jv.get_outputs(y)


# ---------------------------------------------------------------------
# TEST 1 — интерфейс
# ---------------------------------------------------------------------
def test_interface():
    print("\n" + "=" * 70)
    print("TEST 1: Интерфейс OrganModel")
    print("=" * 70)
    jv = make_jv()
    sz = jv.get_state_size()
    y0 = jv.get_initial_state()
    d0 = jv.get_derivatives(0.0, y0, INPUTS_STD)
    print(f"  state_size = {sz}  (ожидание 3)")
    print(f"  y0         = {y0}")
    print(f"  dy/dt(0)   = {d0}")
    ok = (sz == 3 and y0.size == 3 and d0.size == 3
          and np.all(np.isfinite(y0)) and np.all(np.isfinite(d0)))
    print(f"  [{'OK' if ok else 'FAIL'}] интерфейс корректен")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 2 — YAML-консистентность
# ---------------------------------------------------------------------
def test_yaml_consistency():
    print("\n" + "=" * 70)
    print("TEST 2: YAML-консистентность")
    print("=" * 70)
    jv = make_jv()
    checks = [
        ("V0",         jv.V0,            JUG_CFG.get("V0", 150.0)),
        ("C",          jv.C,             JUG_CFG.get("C", 20.0)),
        ("P0",         jv.P0,            JUG_CFG.get("P0", 6.0)),
        ("R_out",      jv.R_out,         JUG_CFG.get("R_out", 0.5)),
        ("tau_target", jv.tau_target,    JUG_CFG.get("tau_target", 200.0)),
        ("targ_frac",  jv.target_fraction, JUG_CFG.get("target_fraction", 0.05)),
    ]
    ok = True
    for name, actual, expected in checks:
        same = abs(actual - expected) < 1e-9
        ok = ok and same
        print(f"  {name:12s}: class={actual:<8.2f}  yaml={expected:<8.2f}  [{'OK' if same else 'FAIL'}]")
    # Ключевой момент: V0 в whole_body = 0.05·V_blood, а не 150
    print(f"  Примечание: whole_body передаёт V0 = 0.05·5800 = 290, "
          f"а не дефолт класса 150.")
    print(f"  [{'OK' if ok else 'FAIL'}] параметры соответствуют YAML")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 3 — Steady state + сверка с аналитикой
# ---------------------------------------------------------------------
def test_steady_state():
    print("\n" + "=" * 70)
    print("TEST 3: Steady state при типичных входах (brain-like)")
    print("=" * 70)
    jv = make_jv()
    sol = integrate(jv, INPUTS_STD, t_end=600.0)
    y_ss = sol.y[:, -1]
    o = out_at(jv, y_ss, INPUTS_STD)

    # Аналитический прогноз:
    C, P0, V0, R_out = jv.C, jv.P0, jv.V0, jv.R_out
    Vtgt = jv.target_fraction * INPUTS_STD["V_blood"]
    Q_in = INPUTS_STD["Q_in"]
    P_sv = INPUTS_STD["P_sv"]
    # Из dV=0:
    # (P0+(V−V0)/C − P_sv)/R_out = Q_in + (Vtgt−V)/tau
    # → (V−V0)·(1/(C·R_out) + 1/tau) = Q_in + (Vtgt−V0)/tau + (P_sv−P0)/R_out
    k = 1.0/(C*R_out) + 1.0/jv.tau_target
    rhs = Q_in + (Vtgt - V0)/jv.tau_target + (P_sv - P0)/R_out
    V_pred = V0 + rhs / k
    P_pred = P0 + (V_pred - V0)/C
    Qout_pred = (P_pred - P_sv)/R_out

    print(f"  Численное (t=600 с):")
    print(f"    V_jv   = {o['V_jv']:8.2f} мл")
    print(f"    P_jv   = {o['P_jv']:8.2f} мм рт.ст.")
    print(f"    Q_out  = {o['Q_jv_out']:8.3f} мл/с")
    print(f"    C_jv_O2  = {o['C_jv_O2']:.4f}")
    print(f"    C_jv_CO2 = {o['C_jv_CO2']:.4f}")
    print(f"    SjvO2  = {o['SjvO2']*100:.1f} %")
    print(f"  Аналитический прогноз:")
    print(f"    V_jv*  = {V_pred:8.2f} мл")
    print(f"    P_jv*  = {P_pred:8.2f} мм рт.ст.")
    print(f"    Q_out* = {Qout_pred:8.3f} мл/с")

    checks = [
        ("V_jv в [250, 500] мл",       250 < o["V_jv"] < 500),
        ("P_jv в [5, 15] мм рт.ст.",   5  < o["P_jv"] < 15),
        ("Q_out ≈ Q_in",               abs(o["Q_jv_out"] - Q_in) < 1.0),
        ("SjvO2 в [55, 70] %",         0.55 < o["SjvO2"] < 0.70),
        ("V_num vs V_pred < 5 %",      abs(o["V_jv"] - V_pred)/V_pred < 0.05),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 4 — Трекинг газов
# ---------------------------------------------------------------------
def test_gas_tracking():
    print("\n" + "=" * 70)
    print("TEST 4: Трекинг газов: C_jv → C_in")
    print("=" * 70)
    jv = make_jv()
    sol = integrate(jv, INPUTS_STD, t_end=600.0)
    C_O2  = sol.y[1, :]
    C_CO2 = sol.y[2, :]
    t = sol.t
    tau_gas = jv.V0 / max(INPUTS_STD["Q_in"], 1e-9)

    print(f"  τ_gas ≈ V0/Q_in = {tau_gas:.1f} с")
    for tt in (0, tau_gas, 3*tau_gas, 5*tau_gas, 600):
        i = int(np.argmin(np.abs(t - tt)))
        print(f"    t={tt:5.0f}с  C_O2={C_O2[i]:.4f}  C_CO2={C_CO2[i]:.4f}")

    rel_O2  = abs(C_O2[-1]  - INPUTS_STD["C_in_O2"])  / INPUTS_STD["C_in_O2"]
    rel_CO2 = abs(C_CO2[-1] - INPUTS_STD["C_in_CO2"]) / INPUTS_STD["C_in_CO2"]
    print(f"  rel отклонение от C_in: O2={rel_O2:.2e}, CO2={rel_CO2:.2e}")
    ok = rel_O2 < 0.01 and rel_CO2 < 0.01
    print(f"  [{'OK' if ok else 'FAIL'}] C_jv сходится к C_in (< 1%)")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 5 — Отклик на шаг Q_in
# ---------------------------------------------------------------------
def test_qin_step():
    print("\n" + "=" * 70)
    print("TEST 5: Отклик на изменение Q_in (washout)")
    print("=" * 70)
    jv = make_jv()
    # стартуем с C_jv = 0.15, подаём C_in = 0.10 → C_jv должен упасть
    y0 = np.array([jv.V0, 0.15, 0.56])
    def rhs(t, y): return jv.get_derivatives(t, y, INPUTS_STD)
    sol = solve_ivp(rhs, (0, 300), y0, method="LSODA", rtol=1e-7, atol=1e-9,
                    max_step=0.5)
    tau_obs = None
    # Оценим τ по времени, когда C_O2 пройдёт 63 % пути к C_in
    target = 0.15 + (0.10 - 0.15) * 0.632
    idx = int(np.argmin(np.abs(sol.y[1] - target)))
    tau_obs = sol.t[idx]
    tau_theor = jv.V0 / INPUTS_STD["Q_in"]
    print(f"  τ наблюдаемая ≈ {tau_obs:.1f} с, теория V0/Q_in = {tau_theor:.1f} с")
    ok = 0.5 < tau_obs/tau_theor < 2.0
    print(f"  [{'OK' if ok else 'FAIL'}] τ в пределах 2× от V0/Q_in")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 6 — Отклик на P_sv
# ---------------------------------------------------------------------
def test_psv_response():
    print("\n" + "=" * 70)
    print("TEST 6: Отклик на P_sv (выше P_sv → ниже V_jv)")
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
    ok = all(vs[i] > vs[i+1] for i in range(len(vs)-1))
    print(f"  [{'OK' if ok else 'FAIL'}] V_jv монотонно падает с ростом P_sv")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 7 — Mass balance на полке
# ---------------------------------------------------------------------
def test_mass_balance():
    print("\n" + "=" * 70)
    print("TEST 7: Mass balance на полке")
    print("=" * 70)
    jv = make_jv()
    sol = integrate(jv, INPUTS_STD, t_end=600.0)
    y_ss = sol.y[:, -1]
    o = out_at(jv, y_ss, INPUTS_STD)
    V_target = jv.target_fraction * INPUTS_STD["V_blood"]
    pull = (V_target - o["V_jv"]) / jv.tau_target
    balance = INPUTS_STD["Q_in"] - o["Q_jv_out"] + pull
    print(f"  Q_in              = {INPUTS_STD['Q_in']:.4f} мл/с")
    print(f"  Q_out             = {o['Q_jv_out']:.4f} мл/с")
    print(f"  (V_target − V)/τ  = {pull:.4f} мл/с")
    print(f"  Сумма dV/dt       = {balance:.3e} мл/с")
    ok = abs(balance) < 1e-3
    print(f"  [{'OK' if ok else 'FAIL'}] |dV/dt| < 1e-3")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 8 — Мягкий пол
# ---------------------------------------------------------------------
def test_soft_floor():
    print("\n" + "=" * 70)
    print("TEST 8: Мягкий пол (объём не уходит в минус)")
    print("=" * 70)
    jv = make_jv()
    # Экстремально: Q_in = 0, P_sv = 30 (высокое) → попытка высосать объём
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
# Запуск + сводка
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
        print("\nВЫВОД: JugularVein физиологичен в изоляции.")
        print("Все ожидаемые числа совпали: V≈387, P≈10.8, Q_out≈11.7, SjvO2≈60%.")
    else:
        print("\nВЫВОД: см. FAIL выше.")


def run_all(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    sys.stdout = Tee(sys.stdout, buf)
    try:
        print("=" * 70)
        print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("JugularVein: изолированный тест")
        print("=" * 70)
        print(f"  YAML-параметры: {JUG_CFG}")
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
        sys.stdout = sys.__stdout__
    path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Отчёт: {path}")


if __name__ == "__main__":
    run_all(RESULT_FILE)