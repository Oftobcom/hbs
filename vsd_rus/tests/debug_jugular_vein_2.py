#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_jugular_vein_2.py — расширенная проверка JugularVein (Option A).

Запуск:  python tests/debug_jugular_vein_v2_2.py
Отчёт:   tests/results_debug_jugular_vein_v2_2.txt

Тесты:
  1. Интерфейс OrganModel
  2. Steady state (аналитика совпадает с численным)
  3. Mass balance: Q_in = Q_out
  4. Washout: скачок Q_in 5→15, τ = V_ss/Q_in
  5. Венозное давление: P_sv 2→15 → P_jv растёт, V_jv растёт, Q_out не меняется
  6. V_blood independence — V_jv НЕ зависит от V_blood (Option A)
  7. Ишемия: C_in_O2 0.12→0.08 → SjvO2 < 50%
  8. Drift 600 с как t_calib
  9. Strict: положительность, Q_out ≥ 0, SjvO2 ∈ [0,1], конечность
 10. YAML-консистентность и отфильтрованные deprecated-ключи
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

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_jugular_vein_v2_2.txt"

_JUG_KNOWN_KEYS = {"C", "P0", "V0", "R_out",
                   "C_O2_init", "C_CO2_init", "Hb"}
_JUG_DEPRECATED_KEYS = {"target_fraction", "tau_target"}


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, d):
        for s in self.s: s.write(d); s.flush()
    def flush(self):
        for s in self.s: s.flush()


def load_cfg():
    cfg = load_physiology()
    return (dict(cfg.get("jugular_vein", {})),
            dict(cfg.get("brain", {})),
            dict(cfg.get("systemic", {})))


JV_CFG, BRAIN_CFG, SYS_CFG = load_cfg()


def make_jv() -> JugularVein:
    """JugularVein, отфильтровывая deprecated-ключи YAML (Option A)."""
    kwargs = {k: v for k, v in JV_CFG.items() if k in _JUG_KNOWN_KEYS}
    return JugularVein(**kwargs)


INPUTS_HEALTHY = {
    "Q_in": 12.0,
    "C_in_O2": 0.12,
    "C_in_CO2": 0.56,
    "P_sv": 5.0,
    "V_blood": 5800.0,   # не читается внутри, оставлен для совместимости
}


def banner():
    print("=" * 70)
    print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S} | NumPy {np.__version__}")
    print("JugularVein: изолированный тест (v2.2, Option A)")
    print("=" * 70)
    print(f"  Входы здоровые: {INPUTS_HEALTHY}")
    print(f"  Параметры YAML jugular_vein:")
    for k, v in JV_CFG.items():
        mark = " (deprecated)" if k in _JUG_DEPRECATED_KEYS else ""
        print(f"    {k:20s} = {v}{mark}")
    print(f"  → {RESULT_FILE}")
    print("=" * 70)


def integrate(jv, inputs, t_end=600, method="LSODA",
              rtol=1e-6, atol=1e-8, max_step=0.05):
    y0 = jv.get_initial_state()
    def rhs(t, y): return jv.get_derivatives(t, y, inputs)
    sol = solve_ivp(rhs, (0.0, t_end), y0, method=method,
                    rtol=rtol, atol=atol, max_step=max_step)
    jv.get_derivatives(sol.t[-1], sol.y[:, -1], inputs)
    return sol, jv.get_outputs(sol.y[:, -1])


def _v_ss(jv, inputs):
    """Аналитический стационар Option A: V_ss = V0 + C·(Q_in·R_out + P_sv − P0)."""
    return (jv.V0 + jv.C * (inputs["Q_in"] * jv.R_out
                            + inputs["P_sv"] - jv.P0))


# ---------------------------------------------------------------------
def test_interface():
    print("\n" + "=" * 70 + "\nTEST 1: Интерфейс OrganModel\n" + "=" * 70)
    jv = make_jv()
    sz = jv.get_state_size(); y0 = jv.get_initial_state()
    d = jv.get_derivatives(0.0, y0, INPUTS_HEALTHY)
    print(f"  state_size = {sz}   (ожидание 3)")
    print(f"  y0         = {y0}")
    print(f"  dy/dt(0)   = {d}")
    ok = sz == 3 and y0.size == 3 and d.size == 3
    print(f"  [{'OK' if ok else 'FAIL'}] интерфейс")
    return {"ok": ok}


def test_steady():
    print("\n" + "=" * 70
          + "\nTEST 2: Steady state + аналитика\n" + "=" * 70)
    jv = make_jv()
    sol, out = integrate(jv, INPUTS_HEALTHY, t_end=600)
    V_pred = _v_ss(jv, INPUTS_HEALTHY)
    P_pred = jv.P0 + (V_pred - jv.V0) / jv.C
    print(f"  V_jv  = {out['V_jv']:.1f} мл   (аналитика {V_pred:.1f})")
    print(f"  P_jv  = {out['P_jv']:.2f}      (аналитика {P_pred:.2f})")
    print(f"  Q_in  = {out['Q_in']:.2f}  Q_out = {out['Q_jv_out']:.2f}")
    print(f"  SjvO2 = {out['SjvO2']*100:.1f} %")
    ok = (abs(out['V_jv'] - V_pred) < 2.0
          and abs(out['Q_jv_out'] - out['Q_in']) < 0.5
          and 0.55 <= out['SjvO2'] <= 0.75)
    print(f"  [{'OK' if ok else 'FAIL'}] steady")
    return {"ok": ok, "out": out, "sol": sol}


def test_mass_balance():
    print("\n" + "=" * 70
          + "\nTEST 3: Mass balance Q_in = Q_out\n" + "=" * 70)
    jv = make_jv()
    sol, out = integrate(jv, INPUTS_HEALTHY, t_end=600)
    dydt = jv.get_derivatives(sol.t[-1], sol.y[:, -1], INPUTS_HEALTHY)
    print(f"  dV/dt  = {dydt[0]:+.3e} мл/с (|·| < 1e-3)")
    print(f"  Q_in − Q_out = {out['Q_in'] - out['Q_jv_out']:+.3e} мл/с")
    ok = abs(dydt[0]) < 1e-3 and abs(out["Q_in"] - out["Q_jv_out"]) < 1e-3
    print(f"  [{'OK' if ok else 'FAIL'}] mass balance")
    return {"ok": ok}


def test_washout():
    print("\n" + "=" * 70
          + "\nTEST 4: Washout — Q_in 5→15, τ = V_ss/Q_in\n" + "=" * 70)
    jv = make_jv()
    inputs_low = dict(INPUTS_HEALTHY, Q_in=5.0, C_in_O2=0.10)
    sol_low, _ = integrate(jv, inputs_low, t_end=300)
    y0 = sol_low.y[:, -1]
    inputs_high = dict(INPUTS_HEALTHY, Q_in=15.0, C_in_O2=0.13)
    def rhs(t, y): return jv.get_derivatives(t, y, inputs_high)
    sol = solve_ivp(rhs, (0, 120), y0, method="LSODA",
                    rtol=1e-6, atol=1e-8, max_step=0.05)
    jv.get_derivatives(sol.t[-1], sol.y[:, -1], inputs_high)
    out = jv.get_outputs(sol.y[:, -1])
    tau = out['V_jv'] / inputs_high["Q_in"]
    print(f"  C_in 0.10→0.13, после 120с C_jv_O2={out['C_jv_O2']:.3f} "
          f"(ожидание ~0.13)")
    print(f"  τ = V/Q = {tau:.1f} с")
    ok = abs(out['C_jv_O2'] - 0.13) < 0.01
    print(f"  [{'OK' if ok else 'FAIL'}] washout")
    return {"ok": ok}


def test_venous_pressure():
    print("\n" + "=" * 70
          + "\nTEST 5: P_sv 2→15 (V_jv растёт, Q_out инвариант)\n" + "=" * 70)
    vals = []
    for P_sv in [2, 5, 15]:
        inp = dict(INPUTS_HEALTHY, P_sv=P_sv)
        jv = make_jv()
        sol, out = integrate(jv, inp, t_end=600)
        vals.append((out['V_jv'], out['P_jv'], out['Q_jv_out']))
        print(f"  P_sv {P_sv:2d} → V_jv {out['V_jv']:6.1f}  "
              f"P_jv {out['P_jv']:5.2f}  Q_out {out['Q_jv_out']:5.2f}")
    vs = [v[0] for v in vals]; ps = [v[1] for v in vals]; qs = [v[2] for v in vals]
    ok_v = vs[0] < vs[1] < vs[2]
    ok_p = ps[0] < ps[1] < ps[2]
    ok_q = all(abs(q - INPUTS_HEALTHY["Q_in"]) < 0.5 for q in qs)
    ok = ok_v and ok_p and ok_q
    print(f"  [{'OK' if ok_v else 'FAIL'}] V_jv монотонно растёт")
    print(f"  [{'OK' if ok_p else 'FAIL'}] P_jv монотонно растёт")
    print(f"  [{'OK' if ok_q else 'FAIL'}] Q_out = Q_in во всех стационарах")
    return {"ok": ok}


def test_vblood_independence():
    print("\n" + "=" * 70
          + "\nTEST 6: V_blood independence — V_jv НЕ зависит от V_blood\n" + "=" * 70)
    rows = []
    for Vb in [4000, 5800, 7000]:
        inp = dict(INPUTS_HEALTHY, V_blood=Vb)
        jv = make_jv()
        sol, out = integrate(jv, inp, t_end=600)
        rows.append((Vb, out['V_jv']))
        print(f"  V_blood {Vb} → V_jv {out['V_jv']:.2f} мл")
    vs = [r[1] for r in rows]
    spread = max(vs) - min(vs)
    print(f"  Разброс V_jv по V_blood: {spread:.6f} мл  (ожидание ~0)")
    ok = spread < 1e-3
    print(f"  [{'OK' if ok else 'FAIL'}] V_jv инвариантен по V_blood (Option A)")
    return {"ok": ok}


def test_ischemia():
    print("\n" + "=" * 70
          + "\nTEST 7: Ишемия — C_in_O2 0.12→0.08 → SjvO2 < 50%\n" + "=" * 70)
    jv = make_jv()
    sol, out = integrate(jv, INPUTS_HEALTHY, t_end=300)
    print(f"  База SjvO2 {out['SjvO2']*100:.1f} %")
    inp = dict(INPUTS_HEALTHY, C_in_O2=0.08)
    y0 = sol.y[:, -1]
    def rhs(t, y): return jv.get_derivatives(t, y, inp)
    sol2 = solve_ivp(rhs, (0, 120), y0, method="LSODA",
                     rtol=1e-6, atol=1e-8, max_step=0.05)
    jv.get_derivatives(sol2.t[-1], sol2.y[:, -1], inp)
    out2 = jv.get_outputs(sol2.y[:, -1])
    print(f"  После 120 с: C_jv_O2={out2['C_jv_O2']:.3f}, "
          f"SjvO2={out2['SjvO2']*100:.1f} %")
    ok = out2['SjvO2'] < 0.5
    print(f"  [{'OK' if ok else 'FAIL'}] детекция ишемии")
    return {"ok": ok}


def test_drift():
    print("\n" + "=" * 70
          + "\nTEST 8: Drift 600 с (t_calib)\n" + "=" * 70)
    jv = make_jv()
    sol, _ = integrate(jv, INPUTS_HEALTHY, t_end=600)
    V = sol.y[0, :]
    m1 = np.mean(V[sol.t >= 580]); m2 = np.mean(V[sol.t >= 590])
    drift = abs(m2 - m1)
    print(f"  V[580..600] = {m1:.3f}  V[590..600] = {m2:.3f}  "
          f"drift = {drift:.4f}  nfev = {sol.nfev}")
    ok = drift < 1.0
    print(f"  [{'OK' if ok else 'FAIL'}] drift < 1 мл")
    return {"ok": ok}


def test_strict():
    print("\n" + "=" * 70
          + "\nTEST 9: Строгий — положительность / Q_out ≥ 0 / SjvO2 ∈ [0,1]\n"
          + "=" * 70)
    jv = make_jv()
    sol, out = integrate(jv, INPUTS_HEALTHY, t_end=600,
                         rtol=1e-9, atol=1e-12, max_step=0.05)
    V = sol.y[0, :]; C_O2 = sol.y[1, :]; C_CO2 = sol.y[2, :]
    checks = [
        ("V_jv > 0",                 np.all(V > 0)),
        ("V_jv > 0.5·V0",            np.min(V) > 0.5 * jv.V0),
        ("C_O2 ∈ [0, 0.3]",          np.all((C_O2 >= 0) & (C_O2 <= 0.3))),
        ("C_CO2 ∈ [0, 1]",           np.all((C_CO2 >= 0) & (C_CO2 <= 1))),
        ("Q_out ≥ 0",                out['Q_jv_out'] >= 0),
        ("SjvO2 ∈ [0, 1]",           0 <= out['SjvO2'] <= 1),
        ("P_jv ≥ 0",                 out['P_jv'] >= 0),
        ("finite",                   np.all(np.isfinite(sol.y))),
    ]
    ok_all = True
    for n, o in checks:
        ok_all = ok_all and o
        print(f"  [{'OK' if o else 'FAIL'}] {n}")
    print(f"  V_min {V.min():.1f} V_max {V.max():.1f} P_jv {out['P_jv']:.2f}")
    return {"ok": ok_all}


def test_yaml():
    print("\n" + "=" * 70
          + "\nTEST 10: YAML-консистентность и deprecated-ключи\n" + "=" * 70)
    jv = make_jv()
    checks = [
        ("V0",    jv.V0,    JV_CFG.get("V0", 150.0)),
        ("C",     jv.C,     JV_CFG.get("C", 20.0)),
        ("P0",    jv.P0,    JV_CFG.get("P0", 6.0)),
        ("R_out", jv.R_out, JV_CFG.get("R_out", 0.5)),
    ]
    ok = True
    for name, a, e in checks:
        same = abs(a - e) < 1e-9
        ok = ok and same
        print(f"  {name:8s}: class={a:<8.2f}  yaml={e:<8.2f}  "
              f"[{'OK' if same else 'FAIL'}]")
    dep_present = [k for k in _JUG_DEPRECATED_KEYS if k in JV_CFG]
    print(f"  deprecated-ключи в physiology.yaml: {dep_present or 'нет'}")
    print(f"  → make_jv() их фильтрует перед JugularVein(**kwargs).")
    print(f"  [{'OK' if ok else 'FAIL'}] параметры соответствуют YAML")
    return {"ok": ok}


# ---------------------------------------------------------------------
def summary(res):
    print("\n" + "=" * 70 + "\nСВОДКА\n" + "=" * 70)
    names = ["Интерфейс", "Steady", "Mass balance", "Washout",
             "Venous P_sv", "V_blood independence", "Ишемия SjvO2",
             "Drift 600с", "Strict", "YAML-консистентность"]
    for n, r in zip(names, res):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    print("\nВЫВОД:",
          "JugularVein корректна (Option A)"
          if all(r["ok"] for r in res) else "Есть FAIL")


def run_all(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO(); orig = sys.stdout
    sys.stdout = Tee(orig, buf)
    try:
        banner()
        rs = [
            test_interface(),
            test_steady(),
            test_mass_balance(),
            test_washout(),
            test_venous_pressure(),
            test_vblood_independence(),
            test_ischemia(),
            test_drift(),
            test_strict(),
            test_yaml(),
        ]
        summary(rs)
        print(f"\nФиниш {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout = orig
    path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Отчёт: {path}")


if __name__ == "__main__":
    run_all(RESULT_FILE)