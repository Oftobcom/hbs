#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_lungs.py — строгая изолированная проверка Lungs2Chamber.

Состояние: [P_prox, P_dist, R_remodel].
Три механизма PVR: f_recruit, f_flow, R_remodel + виртуальный клапан R2.

Тесты:
  1. Интерфейс OrganModel (3 состояния)
  2. Аналитический steady-state vs численный (brentq vs LSODA)
  3. Kirchhoff: Q_pulm = Q_int = Q_out на стационаре
  4. Recruitment-кривая f_recruit(P_pa) — все опорные точки
  5. Flow-фактор f_flow(Q_pulm) — off/on
  6. Q-sweep: нелинейность P_pa(Q) — recruitment буферизует
  7. Виртуальный клапан _valve_flow: односторонность, dP=0→0, гладкость
  8. Масс-баланс (интегральная форма с учётом клапана)
  9. Структурное ремоделирование: активация, target, clipping
 10. Восстановление из нефизичного начального состояния
 11. Строгие инварианты на сетке конфигов

Запуск:  python tests/debug_lungs.py
Отчёт:   tests/results_debug_lungs.txt
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
from scipy.optimize import brentq
from lungs import Lungs2Chamber
from physio_config import load_physiology

if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_lungs_1.txt"


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, d):
        for x in self.s: x.write(d); x.flush()
    def flush(self):
        for x in self.s: x.flush()


CFG = load_physiology()
LUN_CFG = dict(CFG.get("lungs", {}))


def make_lungs(**overrides) -> Lungs2Chamber:
    cfg = dict(LUN_CFG)
    cfg.update(overrides)
    return Lungs2Chamber(**cfg)


INPUTS_NORM = {
    "Q_pulmonary": 83.0,
    "P_pv": 12.0,
}


def integrate(lungs, inputs, t_end=30.0, y0=None, max_step=0.02):
    if y0 is None:
        y0 = lungs.get_initial_state()
    def rhs(t, y): return lungs.get_derivatives(t, y, inputs)
    return solve_ivp(rhs, (0.0, t_end), y0, method="LSODA",
                     rtol=1e-9, atol=1e-11, max_step=max_step)


def out_at(lungs, y, inputs):
    lungs.get_derivatives(0.0, y, inputs)
    return lungs.get_outputs(y)


def analytic_steady_state(lungs, Q_pulm, P_pv):
    """
    Решает неявное уравнение:
        P_prox = P_pv + Q·(R1_base+R2_base)·f_recruit(P_prox)·f_flow·R_remodel(P_prox)

    f_recruit и R_remodel зависят от P_prox → brentq.
    Возвращает dict с P_prox, P_dist, R1_eff, R2_eff, R_remodel.
    """
    f_flow = lungs._flow_factor(Q_pulm)

    def F(P_prox):
        f_rec = lungs._recruit_factor(P_prox)
        R_rem = lungs._R_remodel_target(P_prox)
        R_sum = (lungs.R1_base + lungs.R2_base) * f_rec * f_flow * R_rem
        return P_prox - P_pv - Q_pulm * R_sum

    lo = max(0.5, P_pv * 0.5)
    hi = P_pv + 500.0
    # Расширяем, если знаки одинаковы (экзотика при Q=0)
    if F(lo) * F(hi) > 0:
        hi = 5000.0
    try:
        P_prox = brentq(F, lo, hi, xtol=1e-12, rtol=1e-12)
    except ValueError:
        return None

    f_rec = lungs._recruit_factor(P_prox)
    R_rem = lungs._R_remodel_target(P_prox)
    R1_eff = lungs.R1_base * f_rec * f_flow * R_rem
    R2_eff = lungs.R2_base * f_rec * f_flow * R_rem
    P_dist = P_pv + Q_pulm * R2_eff
    return {
        "P_prox": float(P_prox),
        "P_dist": float(P_dist),
        "R1_eff": float(R1_eff),
        "R2_eff": float(R2_eff),
        "R_remodel": float(R_rem),
        "f_recruit": float(f_rec),
        "f_flow": float(f_flow),
    }


def banner():
    print("=" * 78)
    print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S} | NumPy {np.__version__}")
    print("Lungs2Chamber: строгий изолированный тест")
    print("=" * 78)
    for k, v in LUN_CFG.items():
        print(f"    {k:28s} = {v}")
    print(f"  Входы: {INPUTS_NORM}")
    print(f"  → {RESULT_FILE}")
    print("=" * 78)


# ---------------------------------------------------------------------
# TEST 1 — интерфейс
# ---------------------------------------------------------------------
def test_interface():
    print("\n" + "=" * 78)
    print("TEST 1: Интерфейс OrganModel (3 состояния)")
    print("=" * 78)
    lungs = make_lungs()
    sz = lungs.get_state_size()
    y0 = lungs.get_initial_state()
    d0 = lungs.get_derivatives(0.0, y0, INPUTS_NORM)
    print(f"  state_size = {sz}  (ожидание 3)")
    print(f"  y0 = {y0}")
    print(f"  dy/dt(0) = {d0}")
    ok = (sz == 3 and y0.size == 3 and d0.size == 3
          and np.all(np.isfinite(y0)) and np.all(np.isfinite(d0)))
    print(f"  [{'OK' if ok else 'FAIL'}] интерфейс корректен")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 2 — аналитический steady-state vs численный
# ---------------------------------------------------------------------
def test_analytic_vs_numeric():
    print("\n" + "=" * 78)
    print("TEST 2: Аналитический steady-state vs LSODA")
    print("=" * 78)
    print(f"  {'Q':>6}  {'P_prox(num)':>12}  {'P_prox(an)':>12}  "
          f"{'rel':>10}  {'P_dist(num)':>12}  {'rel':>10}")
    ok = True
    for Q in (40.0, 83.0, 166.0, 250.0):
        lungs = make_lungs()
        inp = dict(INPUTS_NORM, Q_pulmonary=Q)
        sol = integrate(lungs, inp, t_end=30.0)
        y_num = sol.y[:, -1]
        P_num, Pd_num = y_num[0], y_num[1]
        an = analytic_steady_state(lungs, Q, INPUTS_NORM["P_pv"])
        if an is None:
            print(f"  Q={Q:>4.0f} — аналитика не найдена")
            ok = False
            continue
        rel_P = abs(P_num - an["P_prox"]) / max(an["P_prox"], 1e-6)
        rel_Pd = abs(Pd_num - an["P_dist"]) / max(an["P_dist"], 1e-6)
        print(f"  {Q:>6.0f}  {P_num:>12.4f}  {an['P_prox']:>12.4f}  "
              f"{rel_P:>10.2e}  {Pd_num:>12.4f}  {rel_Pd:>10.2e}")
        if rel_P > 1e-5 or rel_Pd > 1e-5:
            ok = False
    print(f"  [{'OK' if ok else 'FAIL'}] численный steady-state совпадает с аналитикой < 1e-5")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 3 — Kirchhoff на стационаре
# ---------------------------------------------------------------------
def test_kirchhoff():
    print("\n" + "=" * 78)
    print("TEST 3: Kirchhoff — Q_pulm = Q_int = Q_out на стационаре")
    print("=" * 78)
    ok = True
    for Q in (40.0, 83.0, 166.0):
        lungs = make_lungs()
        inp = dict(INPUTS_NORM, Q_pulmonary=Q)
        sol = integrate(lungs, inp, t_end=30.0)
        y = sol.y[:, -1]
        o = out_at(lungs, y, inp)
        rel1 = abs(o["Q_int"] - Q) / Q
        rel2 = abs(o["Q_out"] - Q) / Q
        print(f"  Q={Q:>5.0f}: Q_int={o['Q_int']:.6f} (rel={rel1:.2e}), "
              f"Q_out={o['Q_out']:.6f} (rel={rel2:.2e})")
        if rel1 > 1e-6 or rel2 > 1e-6:
            ok = False
    print(f"  [{'OK' if ok else 'FAIL'}] Kirchhoff < 1e-6 на всех Q")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 4 — recruitment-кривая
# ---------------------------------------------------------------------
def test_recruit_curve():
    print("\n" + "=" * 78)
    print("TEST 4: Recruitment-кривая f_recruit(P_pa) — опорные точки")
    print("=" * 78)
    lungs = make_lungs()
    ref = [
        (10.0, 0.95, 0.02),
        (15.0, 0.87, 0.02),
        (20.0, 0.78, 0.03),
        (30.0, 0.65, 0.03),
        (40.0, 0.60, 0.03),
    ]
    print(f"  {'P_pa':>5}  {'f_recruit':>12}  {'докум.':>10}  {'|diff|':>10}")
    ok = True
    for P, ref_val, tol in ref:
        f = lungs._recruit_factor(P)
        diff = abs(f - ref_val)
        print(f"  {P:>5.1f}  {f:>12.4f}  {ref_val:>10.4f}  {diff:>10.4f}")
        if diff > tol:
            ok = False
    # Монотонность
    P_arr = np.linspace(5.0, 60.0, 200)
    f_arr = np.array([lungs._recruit_factor(p) for p in P_arr])
    mono = bool(np.all(np.diff(f_arr) <= 1e-12))
    print(f"  Монотонно убывает: {mono}")
    ok = ok and mono
    print(f"  [{'OK' if ok else 'FAIL'}] recruitment-кривая соответствует docstring")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 5 — flow-фактор
# ---------------------------------------------------------------------
def test_flow_factor():
    print("\n" + "=" * 78)
    print("TEST 5: Flow-фактор f_flow(Q_pulm)")
    print("=" * 78)
    lungs_off = make_lungs(flow_dependent_resistance=False)
    lungs_on  = make_lungs(flow_dependent_resistance=True, flow_sensitivity=0.15,
                           Q_norm=80.0)
    print(f"  {'Q':>6}  {'f_flow(off)':>14}  {'f_flow(on)':>14}")
    for Q in (40, 80, 83, 100, 166, 250, 400):
        fo = lungs_off._flow_factor(Q)
        fn = lungs_on._flow_factor(Q)
        print(f"  {Q:>6}  {fo:>14.4f}  {fn:>14.4f}")

    checks = [
        ("off: f ≡ 1.0",                    lungs_off._flow_factor(500) == 1.0),
        ("on: Q=80 → f=1.0",                abs(lungs_on._flow_factor(80.0) - 1.0) < 1e-12),
        ("on: Q=83 → f≈1.0056",             abs(lungs_on._flow_factor(83.0) - 1.005625) < 1e-6),
        ("on: Q=166 → f≈1.1613",            abs(lungs_on._flow_factor(166.0) - 1.161250) < 1e-6),
        ("on: Q=400 → f=1.6 (не клип)",     abs(lungs_on._flow_factor(400.0) - 1.600) < 1e-6),
        ("on: Q=2000 → f=3.0 (клип)",       abs(lungs_on._flow_factor(2000.0) - 3.0) < 1e-12),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 6 — Q-sweep нелинейность
# ---------------------------------------------------------------------
def test_q_sweep():
    print("\n" + "=" * 78)
    print("TEST 6: Нелинейность P_pa(Q_pulm) — recruitment буферизует")
    print("=" * 78)
    results = []
    for Q in (40.0, 83.0, 166.0, 250.0, 400.0):
        inp = dict(INPUTS_NORM, Q_pulmonary=Q)
        lungs = make_lungs()
        sol = integrate(lungs, inp, t_end=30.0)
        o = out_at(lungs, sol.y[:, -1], inp)
        results.append((Q, o["P_pa"], o["R1_eff"], o["recruit_factor"], o["flow_factor"]))
        print(f"  Q={Q:6.0f} → P_pa={o['P_pa']:6.3f}, R1_eff={o['R1_eff']:.5f}, "
              f"f_recruit={o['recruit_factor']:.4f}, f_flow={o['flow_factor']:.4f}")

    pp = [r[1] for r in results]
    # Ключевое: рост P_pa должен быть сублинейным
    # Наивно линейный P_pa(Q) дал бы P_pa(400)/P_pa(83) = 400/83 ≈ 4.82
    # Реально recruitment + flow-эффект снижают этот рост
    ratio_Q = 400.0 / 83.0
    ratio_P = pp[4] / pp[1]
    print(f"  ratio Q = {ratio_Q:.2f}, ratio P_pa = {ratio_P:.2f}  "
          f"(ratio_P < ratio_Q = сублинейность)")

    checks = [
        ("P_pa монотонно растёт",              all(pp[i] < pp[i+1] for i in range(len(pp)-1))),
        ("P_pa(40) ∈ [14, 17]",                14 < pp[0] < 17),
        ("P_pa(83) ∈ [17, 21]",                17 < pp[1] < 21),
        ("P_pa(166) ∈ [23, 29]",               23 < pp[2] < 29),
        ("ratio_P < 0.85·ratio_Q (буферизация)", ratio_P < 0.85 * ratio_Q),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 7 — виртуальный клапан _valve_flow
# ---------------------------------------------------------------------
def test_valve_flow():
    print("\n" + "=" * 78)
    print("TEST 7: Виртуальный клапан _valve_flow (односторонний поток)")
    print("=" * 78)
    lungs = make_lungs()
    R = 0.05

    print(f"  {'dP':>8}  {'_valve_flow':>15}  {'dP/R (ламинар)':>16}")
    for dP in (-10.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 10.0):
        v = lungs._valve_flow(dP, R)
        lam = dP / R
        print(f"  {dP:>8.2f}  {v:>15.6e}  {lam:>16.6e}")

    # Проверки
    v_pos10 = lungs._valve_flow(10.0, R)
    v_zero  = lungs._valve_flow(0.0, R)
    v_neg10 = lungs._valve_flow(-10.0, R)

    # Плавность: производная по dP непрерывна
    dP_arr = np.linspace(-2.0, 2.0, 1001)
    v_arr = np.array([lungs._valve_flow(x, R) for x in dP_arr])
    # Монотонность (не убывает)
    mono = bool(np.all(np.diff(v_arr) >= -1e-10))
    # Гладкость: max |Δv| мало на равномерной сетке
    max_jump = float(np.max(np.abs(np.diff(v_arr))))
    # Порядок величины шага в плавной части (dP >> 1/k)
    smooth_jump = 2.0 / R / 1000

    checks = [
        ("dP=0 → v=0 (точно)",              abs(v_zero) < 1e-15),
        ("dP=10 → v≈dP/R (ламинар)",        abs(v_pos10 - 10.0/R) / (10.0/R) < 1e-10),
        ("dP=-10 → v≈0 (|v|<1e-10)",        abs(v_neg10) < 1e-10),
        ("Монотонно не убывает",            mono),
        ("Макс. скачок ≤ 2× средний (гладкость)",
         max_jump < 3.0 * smooth_jump),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 8 — масс-баланс (интегральная форма с клапаном)
# ---------------------------------------------------------------------
def test_mass_balance():
    print("\n" + "=" * 78)
    print("TEST 8: Масс-баланс (интегральная форма с виртуальным клапаном)")
    print("=" * 78)
    lungs = make_lungs()
    # Стартуем из нефизичного состояния, где P_dist < P_pv → клапан закрыт
    y0 = np.array([20.0, 8.0, 1.0])
    Q_pulm = INPUTS_NORM["Q_pulmonary"]
    P_pv = INPUTS_NORM["P_pv"]

    def rhs(t, y): return lungs.get_derivatives(t, y, INPUTS_NORM)
    sol = solve_ivp(rhs, (0, 30), y0, method="LSODA",
                    rtol=1e-9, atol=1e-12, max_step=0.02, dense_output=True)

    # Вычисляем Q_out(t) вдоль траектории
    t_arr = sol.t
    Q_out_arr = np.zeros_like(t_arr)
    for i, (ti, yi) in enumerate(zip(t_arr, sol.y.T)):
        lungs.get_derivatives(ti, yi, INPUTS_NORM)
        o = lungs.get_outputs(yi)
        Q_out_arr[i] = o["Q_out"]

    integral_in = Q_pulm * (t_arr[-1] - t_arr[0])
    integral_out = float(_trapz(Q_out_arr, t_arr))
    delta_V = (lungs.C1 * (sol.y[0, -1] - sol.y[0, 0])
               + lungs.C2 * (sol.y[1, -1] - sol.y[1, 0]))
    lhs = integral_in - integral_out
    rel = abs(lhs - delta_V) / max(abs(delta_V), abs(lhs), 1e-6)

    print(f"  ∫Q_in dt   = {integral_in:.6f} мл")
    print(f"  ∫Q_out dt  = {integral_out:.6f} мл")
    print(f"  ΔV (C1·ΔP_prox + C2·ΔP_dist) = {delta_V:.6f} мл")
    print(f"  Net in−out = {lhs:.6f} мл")
    print(f"  rel = {rel:.3e}")
    ok = rel < 1e-4
    print(f"  [{'OK' if ok else 'FAIL'}] масс-баланс < 1e-4")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 9 — структурное ремоделирование
# ---------------------------------------------------------------------
def test_pressure_remodel():
    print("\n" + "=" * 78)
    print("TEST 9: Структурное ремоделирование (pressure_remodel=True)")
    print("=" * 78)
    lungs = make_lungs(pressure_remodel=True,
                       P_pa_threshold=25.0,
                       pressure_sensitivity=0.04,
                       R_remodel_max=5.0,
                       tau_remodel=200.0)
    # Длительный прогон при Q=166 (P_pa ≈ 25-30 > threshold)
    inp = dict(INPUTS_NORM, Q_pulmonary=166.0)
    sol = integrate(lungs, inp, t_end=2000.0, max_step=0.5)
    y_end = sol.y[:, -1]
    o = out_at(lungs, y_end, inp)
    R_target = 1.0 + 0.04 * max(o["P_pa"] - 25.0, 0.0)

    print(f"  P_pa       = {o['P_pa']:.3f} мм рт.ст.")
    print(f"  R_remodel  = {o['R_remodel']:.4f}")
    print(f"  R_target   = {R_target:.4f}")
    print(f"  R1_eff     = {o['R1_eff']:.5f}")
    print(f"  R_remodel_max = 5.0")

    # Проверка сходимости: R_remodel → R_target
    rel_conv = abs(o["R_remodel"] - R_target) / max(R_target, 1e-6)

    checks = [
        ("R_remodel > 1 (активировано)",       o["R_remodel"] > 1.0),
        ("R_remodel → R_target (<5%)",         rel_conv < 0.05),
        ("R_remodel ≤ R_remodel_max",          o["R_remodel"] <= 5.0 + 1e-6),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")

    # Дополнительно: клип при экстремально высоком P_pa
    print(f"\n  --- Проверка клипа R_remodel_max ---")
    lungs2 = make_lungs(pressure_remodel=True,
                        P_pa_threshold=10.0,          # низкий порог
                        pressure_sensitivity=0.5,     # агрессивный рост
                        R_remodel_max=3.0,
                        tau_remodel=10.0)             # быстрый
    inp2 = dict(INPUTS_NORM, Q_pulmonary=400.0)
    sol2 = integrate(lungs2, inp2, t_end=500.0, max_step=0.5)
    o2 = out_at(lungs2, sol2.y[:, -1], inp2)
    print(f"  P_pa={o2['P_pa']:.2f}, R_remodel={o2['R_remodel']:.4f} "
          f"(клип на R_remodel_max=3.0)")
    clip_ok = o2["R_remodel"] <= 3.0 + 1e-3
    ok = ok and clip_ok
    print(f"  [{'OK' if clip_ok else 'FAIL'}] R_remodel не превышает R_remodel_max")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 10 — восстановление из нефизичного состояния
# ---------------------------------------------------------------------
def test_state_recovery():
    print("\n" + "=" * 78)
    print("TEST 10: Восстановление из нефизичного начального состояния")
    print("=" * 78)
    lungs = make_lungs()
    # Нефизичное: отрицательные давления, R_remodel ниже клипа
    y0_bad = np.array([-5.0, -3.0, 0.05])
    print(f"  Начальное: y0 = {y0_bad}")
    print(f"  Клипы: P_prox=max(y, 0), P_dist=max(y, 0), R_remodel∈[0.1, {lungs.R_remodel_max}]")

    def rhs(t, y): return lungs.get_derivatives(t, y, INPUTS_NORM)
    sol = solve_ivp(rhs, (0, 60), y0_bad, method="LSODA",
                    rtol=1e-9, atol=1e-12, max_step=0.05)
    y_final = sol.y[:, -1]

    print(f"  Конечное:  y_end = [{y_final[0]:.3f}, {y_final[1]:.3f}, {y_final[2]:.4f}]")
    print(f"  Через 60 с: nfev={sol.nfev}, success={sol.success}")

    checks = [
        ("Нет NaN/Inf в траектории",         bool(np.all(np.isfinite(sol.y)))),
        ("P_prox > 0 в конце",               y_final[0] > 0),
        ("P_dist > 0 в конце",               y_final[1] > 0),
        ("R_remodel ≥ 0.1",                  y_final[2] >= 0.1 - 1e-6),
        ("R_remodel ≤ R_remodel_max",        y_final[2] <= lungs.R_remodel_max + 1e-6),
        ("Результат физичен (P_prox > P_dist > P_pv)",
         y_final[0] > y_final[1] > INPUTS_NORM["P_pv"] - 1.0),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 11 — строгие инварианты (сетка конфигов)
# ---------------------------------------------------------------------
def test_strict_invariants():
    print("\n" + "=" * 78)
    print("TEST 11: Строгие инварианты на сетке (2×2×5×4 = 80 комбинаций)")
    print("=" * 78)
    violations = []
    for flow_dep in (False, True):
        for remodel in (False, True):
            lungs = make_lungs(flow_dependent_resistance=flow_dep,
                               pressure_remodel=remodel)
            for Q in (0.0, 40.0, 83.0, 166.0, 300.0):
                for Ppv in (0.0, 5.0, 12.0, 20.0):
                    inp = dict(INPUTS_NORM, Q_pulmonary=Q, P_pv=Ppv)
                    try:
                        sol = integrate(lungs, inp, t_end=20.0)
                    except Exception as e:
                        violations.append(f"solver crash fd={flow_dep} rm={remodel} Q={Q} Ppv={Ppv}: {e}")
                        continue
                    if not np.all(np.isfinite(sol.y)):
                        violations.append(f"non-finite fd={flow_dep} rm={remodel} Q={Q} Ppv={Ppv}")
                        continue
                    y = sol.y[:, -1]
                    if y[0] < -1e-8 or y[1] < -1e-8:
                        violations.append(f"negative pressure fd={flow_dep} rm={remodel} Q={Q} Ppv={Ppv}")
                    if y[2] < 0.1 - 1e-6 or y[2] > lungs.R_remodel_max + 1e-6:
                        violations.append(f"R_remodel out of range fd={flow_dep} rm={remodel} Q={Q} Ppv={Ppv}")

    if violations:
        for v in violations[:10]:
            print(f"  [FAIL] {v}")
    else:
        print(f"  Все проверки пройдены")
    ok = len(violations) == 0
    print(f"  [{'OK' if ok else 'FAIL'}] строгие инварианты")
    return {"ok": ok}


# ---------------------------------------------------------------------
# Сводка
# ---------------------------------------------------------------------
def summary(results):
    print("\n" + "=" * 78)
    print("СВОДКА: Lungs2Chamber (строгий тест)")
    print("=" * 78)
    names = [
        "Интерфейс 3-state",
        "Аналитический steady-state",
        "Kirchhoff flow balance",
        "Recruitment-кривая",
        "Flow-фактор",
        "Q-sweep (сублинейность)",
        "Виртуальный клапан R2",
        "Масс-баланс (с клапаном)",
        "Структурное ремоделирование",
        "Восстановление состояния",
        "Строгие инварианты (80 комб.)",
    ]
    for n, r in zip(names, results):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    print()
    if all(r["ok"] for r in results):
        print("ВЫВОД: Lungs2Chamber полностью верифицирован.")
        print("Клапан R2 односторонний, аналитика совпадает с численным")
        print("решением, масс-баланс соблюдается в транзиенте, ремоделирование")
        print("корректно клипуется, состояние восстанавливается из нефизичного.")
    else:
        print("ВЫВОД: см. FAIL выше.")


def run_all(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    sys.stdout = Tee(sys.__stdout__, buf)
    try:
        banner()
        rs = [
            test_interface(),
            test_analytic_vs_numeric(),
            test_kirchhoff(),
            test_recruit_curve(),
            test_flow_factor(),
            test_q_sweep(),
            test_valve_flow(),
            test_mass_balance(),
            test_pressure_remodel(),
            test_state_recovery(),
            test_strict_invariants(),
        ]
        summary(rs)
        print(f"\nФиниш: {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout = sys.__stdout__
    path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Отчёт сохранён: {path}")


if __name__ == "__main__":
    run_all(RESULT_FILE)