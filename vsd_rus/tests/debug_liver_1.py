#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_liver.py — изолированная проверка Liver.

Liver — 6-состояние: [P_hv, C_bilirubin, C_ammonia, C_albumin, reserve, P_portal].
Проверяем:
  • steady-state гемодинамику (P_hv, P_portal, Q_ha, Q_pv, Q_out)
  • Kirchhoff: Q_ha + Q_pv = Q_out на стационаре
  • кинетику билирубина, аммиака, альбумина
  • масс-баланс билирубина/аммиака в системе кровь-печень
  • отклик на P_sa и Q_gut_out
  • строгие инварианты

Запуск:  python tests/debug_liver.py
Отчёт:   tests/results_debug_liver.txt
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
from liver import Liver
from physio_config import load_physiology

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_liver_1.txt"


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, d):
        for x in self.s: x.write(d); x.flush()
    def flush(self):
        for x in self.s: x.flush()


CFG = load_physiology()
LIV_CFG = dict(CFG.get("liver", {}))


def make_liver() -> Liver:
    return Liver(**LIV_CFG)


INPUTS_STD = {
    "P_sa": 85.0,
    "P_sv": 12.0,
    "C_bilirubin_blood": 0.5,
    "C_ammonia_blood":   0.3,
    "C_albumin_blood":   4.5,
    "C_lactate_blood":   0.10,
    "V_blood":           5800.0,
    "Q_gut_out":         10.0,   # мл/с — умеренный портальный поток
}


def integrate(liver, inputs, t_end=200.0, max_step=0.5):
    y0 = liver.get_initial_state()
    def rhs(t, y): return liver.get_derivatives(t, y, inputs)
    return solve_ivp(rhs, (0.0, t_end), y0, method="LSODA",
                     rtol=1e-7, atol=1e-9, max_step=max_step)


def out_at(liver, y, inputs):
    liver.get_derivatives(0.0, y, inputs)
    return liver.get_outputs(y)


def banner():
    print("=" * 78)
    print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S} | NumPy {np.__version__}")
    print("Liver: изолированный тест")
    print("=" * 78)
    for k, v in LIV_CFG.items():
        print(f"    {k:28s} = {v}")
    print(f"  Входы: {INPUTS_STD}")
    print(f"  → {RESULT_FILE}")
    print("=" * 78)


# ---------------------------------------------------------------------
# TEST 1 — интерфейс
# ---------------------------------------------------------------------
def test_interface():
    print("\n" + "=" * 78)
    print("TEST 1: Интерфейс OrganModel (6 состояний)")
    print("=" * 78)
    liver = make_liver()
    sz = liver.get_state_size()
    y0 = liver.get_initial_state()
    d0 = liver.get_derivatives(0.0, y0, INPUTS_STD)
    print(f"  state_size = {sz}  (ожидание 6)")
    print(f"  y0 = {y0}")
    print(f"  dy/dt(0) = {d0}")
    ok = (sz == 6 and y0.size == 6 and d0.size == 6
          and np.all(np.isfinite(y0)) and np.all(np.isfinite(d0)))
    print(f"  [{'OK' if ok else 'FAIL'}] интерфейс корректен")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 2 — гемодинамический стационар
# ---------------------------------------------------------------------
def test_steady_hemodynamics():
    print("\n" + "=" * 78)
    print("TEST 2: Стационар гемодинамики (P_sa=85, P_sv=12, Q_gut=10)")
    print("=" * 78)
    liver = make_liver()
    sol = integrate(liver, INPUTS_STD, t_end=200)
    y_ss = sol.y[:, -1]
    o = out_at(liver, y_ss, INPUTS_STD)

    P_hv, _, _, _, _, P_portal = y_ss
    print(f"  P_hv     = {P_hv:6.2f} мм рт.ст.")
    print(f"  P_portal = {P_portal:6.2f} мм рт.ст.")
    print(f"  Q_ha     = {o['Q_ha']:6.3f} мл/с = {o['Q_ha']*60:.0f} мл/мин")
    print(f"  Q_pv     = {o['Q_pv']:6.3f} мл/с (= Q_gut_out)")
    print(f"  Q_out    = {o['Q_liver_out']:6.3f} мл/с")

    # Аналитический прогноз
    # Q_pv = Q_gut_out = 10
    # (85 − P_hv)/17 + 10 = (P_hv − 12)/0.12
    R_ha, R_hv, R_pv = liver.R_ha, liver.R_hv_base, liver.R_pv_base
    Qg = INPUTS_STD["Q_gut_out"]
    Ps, Pv = INPUTS_STD["P_sa"], INPUTS_STD["P_sv"]
    # Решаем: (Ps − Phv)/R_ha + Qg = (Phv − Pv)/R_hv
    P_hv_pred = (Ps/R_ha + Qg + Pv/R_hv) / (1/R_ha + 1/R_hv)
    P_portal_pred = P_hv_pred + Qg * R_pv
    print(f"  Аналитика: P_hv* = {P_hv_pred:.2f}, P_portal* = {P_portal_pred:.2f}")

    checks = [
        ("P_hv ∈ [10, 18]",                    10 < P_hv < 18),
        ("P_portal ∈ [12, 20]",                12 < P_portal < 20),
        ("Q_ha ∈ [3, 8] мл/с",                 3 < o["Q_ha"] < 8),
        ("Q_pv ≈ Q_gut_out (rel<1%)",          abs(o["Q_pv"] - Qg) / Qg < 0.01),
        ("P_hv vs аналитика < 2%",             abs(P_hv - P_hv_pred)/P_hv_pred < 0.02),
        ("P_portal vs аналитика < 2%",         abs(P_portal - P_portal_pred)/P_portal_pred < 0.02),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")
    return {"ok": ok, "out": o, "y_ss": y_ss}


# ---------------------------------------------------------------------
# TEST 3 — Kirchhoff (flow balance)
# ---------------------------------------------------------------------
def test_kirchhoff():
    print("\n" + "=" * 78)
    print("TEST 3: Kirchhoff — Q_ha + Q_pv = Q_out на стационаре")
    print("=" * 78)
    liver = make_liver()
    sol = integrate(liver, INPUTS_STD, t_end=200)
    o = out_at(liver, sol.y[:, -1], INPUTS_STD)
    balance = o["Q_ha"] + o["Q_pv"] - o["Q_liver_out"]
    rel = abs(balance) / max(abs(o["Q_liver_out"]), 1e-6)
    print(f"  Q_ha + Q_pv − Q_out = {balance:.3e} мл/с (rel = {rel:.3e})")
    ok = rel < 1e-4
    print(f"  [{'OK' if ok else 'FAIL'}] Kirchhoff < 0.01%")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 4 — кинетика метаболитов
# ---------------------------------------------------------------------
def test_metabolite_kinetics():
    print("\n" + "=" * 78)
    print("TEST 4: Кинетика метаболитов (билирубин, аммиак, альбумин)")
    print("=" * 78)
    liver = make_liver()
    sol = integrate(liver, INPUTS_STD, t_end=200)
    C_bil, C_amm, C_alb = sol.y[1, -1], sol.y[2, -1], sol.y[3, -1]

    # Аналитические прогнозы
    # dC_bil = 0.1*(C_b - C_l) − 0.2*C_l = 0 → C_l = 0.1*C_b/0.3
    C_bil_pred = 0.1 * 0.5 / (0.1 + 0.2)
    # 0.1*(0.3 − C_l) = 0.15*C_l → C_l = 0.1*0.3/(0.1+0.15)
    C_amm_pred = 0.1 * 0.3 / (0.1 + 0.15)
    # 0.1 − 0.01*C_l − 0.05*(C_l − C_blood) = 0
    # 0.1 + 0.05*C_blood = 0.06*C_l
    C_alb_pred = (0.1 + 0.05 * 4.5) / 0.06

    print(f"  C_bilirubin = {C_bil:.4f}  (аналитика {C_bil_pred:.4f})")
    print(f"  C_ammonia   = {C_amm:.4f}  (аналитика {C_amm_pred:.4f})")
    print(f"  C_albumin   = {C_alb:.4f}  (аналитика {C_alb_pred:.4f})")

    checks = [
        ("C_bilirubin близко к 0.167",  abs(C_bil - C_bil_pred) / C_bil_pred < 0.02),
        ("C_ammonia близко к 0.120",    abs(C_amm - C_amm_pred) / C_amm_pred < 0.02),
        ("C_albumin близко к 5.42",     abs(C_alb - C_alb_pred) / C_alb_pred < 0.02),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 5 — масс-баланс билирубина (кровь + печень)
# ---------------------------------------------------------------------
def test_bilirubin_mass_balance():
    """
    ВНИМАНИЕ: этот тест проверяет корректность масс-баланса
    в системе «кровь + печень». В коде liver.py:
        dC_bil_blood = −clearance_bil / V_blood
    но ФИЗИЧЕСКИ правильно:
        dC_bil_blood = −uptake_bil / V_blood
    Тест выявляет расхождение.
    """
    print("\n" + "=" * 78)
    print("TEST 5: Масс-баланс билирубина (кровь + печень)")
    print("=" * 78)
    liver = make_liver()
    # Возьмём момент сразу после старта, где uptake ≠ clearance
    y0 = liver.get_initial_state()
    d0 = liver.get_derivatives(0.0, y0, INPUTS_STD)
    C_bil_blood = INPUTS_STD["C_bilirubin_blood"]
    V_blood = INPUTS_STD["V_blood"]
    C_bil_liver = y0[1]                      # 0.0
    dC_bil_liver = d0[1]
    dC_bil_blood = -liver.bilirubin_clearance_base * C_bil_liver / V_blood

    # Физически: кровь теряет uptake_bil, а не clearance_bil
    uptake_bil = 0.1 * (C_bil_blood - C_bil_liver)
    clearance_bil = liver.bilirubin_clearance_base * C_bil_liver
    print(f"  При t=0: C_bil_liver={C_bil_liver}, C_bil_blood={C_bil_blood}")
    print(f"  uptake_bil    = {uptake_bil:.4e}")
    print(f"  clearance_bil = {clearance_bil:.4e}")
    print(f"  dC_bil_liver  = {dC_bil_liver:.4e}")
    print(f"  dC_bil_blood  = {dC_bil_blood:.4e} (в коде liver.py)")
    print(f"  Правильно было бы: −uptake_bil/V_blood = {-uptake_bil/V_blood:.4e}")

    # Масс-баланс системы кровь+печень:
    # V_blood*dC_blood + V_liver*dC_liver (V_liver=1) = ?
    total_change = V_blood * dC_bil_blood + dC_bil_liver
    expected_change = -clearance_bil              # bilirubin уходит с желчью
    rel_err = abs(total_change - expected_change) / max(abs(expected_change), 1e-12)
    print(f"  d/dt_total = V_b·dC_blood + dC_liver = {total_change:.4e}")
    print(f"  Ожидание (только -clearance)          = {expected_change:.4e}")
    print(f"  rel_err = {rel_err:.3e}")

    ok = rel_err < 1e-3
    print(f"  [{'OK' if ok else 'FAIL'}] масс-баланс корректен")
    if not ok:
        print("  ↑ Причина: dC_bil_blood = −clearance/V_blood вместо −uptake/V_blood")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 6 — отклик на P_sa
# ---------------------------------------------------------------------
def test_psa_response():
    print("\n" + "=" * 78)
    print("TEST 6: Отклик на P_sa (Q_ha растёт с P_sa)")
    print("=" * 78)
    rows = []
    for P_sa in (60.0, 85.0, 120.0):
        inp = dict(INPUTS_STD, P_sa=P_sa)
        liver = make_liver()
        sol = integrate(liver, inp, t_end=200)
        o = out_at(liver, sol.y[:, -1], inp)
        rows.append((P_sa, o["Q_ha"], o["Q_liver_out"], sol.y[0, -1]))
        print(f"  P_sa={P_sa:6.1f} → Q_ha={o['Q_ha']:6.3f}, "
              f"Q_out={o['Q_liver_out']:6.3f}, P_hv={sol.y[0, -1]:6.2f}")
    qha = [r[1] for r in rows]
    ok = qha[0] < qha[1] < qha[2]
    print(f"  [{'OK' if ok else 'FAIL'}] Q_ha монотонно растёт с P_sa")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 7 — отклик на Q_gut_out
# ---------------------------------------------------------------------
def test_qgut_response():
    print("\n" + "=" * 78)
    print("TEST 7: Отклик на Q_gut_out (P_portal растёт с Q_gut)")
    print("=" * 78)
    rows = []
    for Qg in (5.0, 10.0, 15.0):
        inp = dict(INPUTS_STD, Q_gut_out=Qg)
        liver = make_liver()
        sol = integrate(liver, inp, t_end=200)
        P_hv, _, _, _, _, P_portal = sol.y[:, -1]
        rows.append((Qg, P_portal, P_hv))
        print(f"  Q_gut={Qg:5.1f} → P_portal={P_portal:6.2f}, P_hv={P_hv:6.2f}")
    pp = [r[1] for r in rows]
    ok = pp[0] < pp[1] < pp[2]
    print(f"  [{'OK' if ok else 'FAIL'}] P_portal монотонно растёт с Q_gut_out")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 8 — строгие инварианты
# ---------------------------------------------------------------------
def test_strict():
    print("\n" + "=" * 78)
    print("TEST 8: Строгие инварианты на сетке (P_sa × Q_gut × C_bil_blood)")
    print("=" * 78)
    liver = make_liver()
    violations = []
    for P_sa in (40.0, 85.0, 150.0):
        for Qg in (3.0, 10.0, 20.0):
            for Cb in (0.1, 0.5, 2.0):
                inp = dict(INPUTS_STD, P_sa=P_sa, Q_gut_out=Qg, C_bilirubin_blood=Cb)
                sol = integrate(liver, inp, t_end=100)
                if not np.all(np.isfinite(sol.y)):
                    violations.append(f"non-finite @ ({P_sa},{Qg},{Cb})")
                    continue
                y = sol.y[:, -1]
                if y[0] < 0 or y[5] < 0:
                    violations.append(f"P_hv/P_portal<0 @ ({P_sa},{Qg},{Cb})")
                if y[1] < 0 or y[2] < 0 or y[3] < 0:
                    violations.append(f"conc<0 @ ({P_sa},{Qg},{Cb})")
    if violations:
        for v in violations[:10]:
            print(f"  [FAIL] {v}")
    else:
        print(f"  Все проверки пройдены (27 комбинаций)")
    ok = len(violations) == 0
    print(f"  [{'OK' if ok else 'FAIL'}] строгие инварианты")
    return {"ok": ok}


# ---------------------------------------------------------------------
# Сводка
# ---------------------------------------------------------------------
def summary(results):
    print("\n" + "=" * 78)
    print("СВОДКА: Liver")
    print("=" * 78)
    names = [
        "Интерфейс 6-state",
        "Стационар гемодинамики",
        "Kirchhoff flow balance",
        "Кинетика метаболитов",
        "Масс-баланс билирубина",
        "Отклик на P_sa",
        "Отклик на Q_gut",
        "Строгие инварианты",
    ]
    for n, r in zip(names, results):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    if all(r["ok"] for r in results):
        print("\nВЫВОД: Liver физиологичен в изоляции.")
    else:
        print("\nВЫВОД: см. FAIL выше.")


def run_all(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    sys.stdout = Tee(sys.__stdout__, buf)
    try:
        banner()
        rs = [
            test_interface(),
            test_steady_hemodynamics(),
            test_kirchhoff(),
            test_metabolite_kinetics(),
            test_bilirubin_mass_balance(),
            test_psa_response(),
            test_qgut_response(),
            test_strict(),
        ]
        summary(rs)
        print(f"\nФиниш: {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout = sys.__stdout__
    path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Отчёт сохранён: {path}")


if __name__ == "__main__":
    run_all(RESULT_FILE)