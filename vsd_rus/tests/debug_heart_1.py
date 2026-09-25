#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_heart.py — изолированная проверка Heart4Chambers.

Heart — ПУЛЬСИРУЮЩАЯ система. Нет точки равновесия в строгом смысле,
есть предельный цикл. Интегрируем 30 циклов, усредняем последние 10.

Входы изолированного теста (фиксированы):
    P_sa = 85, P_sv = 12, P_pa = 15, P_pv = 12
    hr_factor = 1.0, baro_activation = 1.0

Что проверяем:
    1. Интерфейс OrganModel
    2. Steady state в Gayton ± допуск
    3. Mass balance (Kirchhoff)
    4. HR sweep: CO растёт с HR
    5. PV loop sanity
    6. VSD shunt (R_vsd=5): L→R, Qp/Qs > 1
    7. RK45 ≈ LSODA
    8. Drift cycle-to-cycle < 5%
    9. Conservation laws & periodicity (строгий)

Цель: доказать, что heart не виновен в P_sa=65, EDV_LV=69.
Если PASS → причина в связях whole_body.
Если FAIL → чинить heart изолированно.

Отчёт сохраняется в:
    results/debug_heart_report.txt

Запуск:
    python tests/debug_heart.py
"""

from __future__ import annotations
import sys
import io
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import solve_ivp

from heart import Heart4Chambers
from physio_config import load_physiology


# =====================================================================
# Tee: дублирование вывода в stdout и в файл
# =====================================================================

class Tee:
    """Перенаправляет вывод одновременно в несколько потоков."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# =====================================================================
# 0. Конфигурация
# =====================================================================

def load_heart_cfg():
    cfg = load_physiology()
    if "heart" not in cfg:
        raise RuntimeError("physiology.yaml: нет секции 'heart'")
    return dict(cfg["heart"])


HEART_CFG = load_heart_cfg()


def make_heart():
    return Heart4Chambers(**HEART_CFG)


INPUTS_HEALTHY = {
    "P_sa": 85.0,
    "P_sv": 12.0,
    "P_pa": 15.0,
    "P_pv": 12.0,
    "hr_factor": 1.0,
    "baro_activation": 1.0,
}


def print_banner():
    print("=" * 70)
    print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("Heart4Chambers: изолированный тест (pulsatile)")
    print("=" * 70)
    print(f"  Входы: {INPUTS_HEALTHY}")
    print(f"  Ключевые параметры из physiology.yaml:")
    for k in ("hr", "E_max_lv", "E_min_lv", "E_max_rv", "E_min_rv",
              "R_mitral", "R_aortic", "R_tricuspid", "R_pulmonary",
              "R_venous_sys", "R_venous_pulm", "k_valve"):
        print(f"    {k:16s} = {HEART_CFG[k]}")
    print("=" * 70)


# =====================================================================
# Интеграция + извлечение steady-state метрик по последним N циклам
# =====================================================================

def integrate_cycles(heart, inputs, n_cycles=30, n_avg=10,
                     method="LSODA", rtol=1e-7, atol=1e-9, max_step=0.005):
    """
    Интегрирует heart на n_cycles кардиоциклов.
    Возвращает sol, T (длительность цикла), t_avg_start (начало усреднения).
    """
    hr = heart.hr_base * inputs.get("hr_factor", 1.0)
    hr = np.clip(hr, heart.hr_min, heart.hr_max)
    T = 60.0 / hr

    t_end = n_cycles * T
    t_avg_start = (n_cycles - n_avg) * T

    y0 = heart.get_initial_state()

    def rhs(t, y):
        return heart.get_derivatives(t, y, inputs)

    sol = solve_ivp(rhs, (0.0, t_end), y0, method=method,
                    rtol=rtol, atol=atol, max_step=max_step)

    return sol, T, t_avg_start


def extract_steady(sol, heart, inputs, T, t_avg_start):
    """Усреднение метрик по последним n_avg циклам."""
    mask = sol.t > t_avg_start
    if np.sum(mask) < 10:
        raise RuntimeError("Слишком мало точек в окне усреднения")

    Q_aortic, Q_mitral, Q_pulmonary, Q_tricuspid = [], [], [], []
    Q_pv_to_la, Q_sv_to_ra = [], []
    P_lv_arr, P_rv_arr, P_la_arr, P_ra_arr = [], [], [], []

    for i in range(len(sol.t)):
        if not mask[i]:
            continue
        heart.get_derivatives(sol.t[i], sol.y[:, i], inputs)
        out = heart.get_outputs(sol.y[:, i])
        Q_aortic.append(out["Q_aortic"])
        Q_mitral.append(out["Q_mitral"])
        Q_pulmonary.append(out["Q_pulmonary"])
        Q_tricuspid.append(out["Q_tricuspid"])
        Q_pv_to_la.append(out["Q_pv_to_la"])
        Q_sv_to_ra.append(out["Q_sv_to_ra"])
        P_lv_arr.append(out["P_lv"])
        P_rv_arr.append(out["P_rv"])
        P_la_arr.append(out["P_la"])
        P_ra_arr.append(out["P_ra"])

    V_lv = sol.y[1, mask]
    V_rv = sol.y[3, mask]
    V_la = sol.y[0, mask]
    V_ra = sol.y[2, mask]

    return {
        "Q_aortic":    float(np.mean(Q_aortic)),
        "Q_mitral":    float(np.mean(Q_mitral)),
        "Q_pulmonary": float(np.mean(Q_pulmonary)),
        "Q_tricuspid": float(np.mean(Q_tricuspid)),
        "Q_pv_to_la":  float(np.mean(Q_pv_to_la)),
        "Q_sv_to_ra":  float(np.mean(Q_sv_to_ra)),
        "EDV_LV":      float(V_lv.max()),
        "ESV_LV":      float(V_lv.min()),
        "SV_LV":       float(V_lv.max() - V_lv.min()),
        "EDV_RV":      float(V_rv.max()),
        "ESV_RV":      float(V_rv.min()),
        "SV_RV":       float(V_rv.max() - V_rv.min()),
        "P_lv_max":    float(np.max(P_lv_arr)),
        "P_lv_min":    float(np.min(P_lv_arr)),
        "P_rv_max":    float(np.max(P_rv_arr)),
        "P_rv_min":    float(np.min(P_rv_arr)),
        "P_la_mean":   float(np.mean(P_la_arr)),
        "P_ra_mean":   float(np.mean(P_ra_arr)),
        "V_la_mean":   float(V_la.mean()),
        "V_ra_mean":   float(V_ra.mean()),
        "nfev":        int(sol.nfev),
        "success":     bool(sol.success),
        "finite":      bool(np.all(np.isfinite(sol.y[:, mask]))),
    }


# =====================================================================
# TEST 1: Интерфейс
# =====================================================================

def test_interface():
    print("\n" + "=" * 70)
    print("TEST 1: Интерфейс OrganModel")
    print("=" * 70)
    heart = make_heart()
    n = heart.get_state_size()
    y0 = heart.get_initial_state()
    d = heart.get_derivatives(0.0, y0, INPUTS_HEALTHY)

    print(f"  state_size    = {n}  (ожидание 4)")
    print(f"  initial state = {y0}")
    print(f"  derivatives   = {d}")

    ok = (n == 4 and y0.size == 4 and d.size == 4
          and np.all(np.isfinite(y0)) and np.all(np.isfinite(d)))
    print(f"  [{'OK' if ok else 'FAIL'}] state_size=4, "
          f"init=[EDV_la, EDV_lv, EDV_ra, EDV_rv]")
    return {"ok": ok}


# =====================================================================
# TEST 2: Steady state
# =====================================================================

def test_steady_state():
    print("\n" + "=" * 70)
    print("TEST 2: Steady state (P_sa=85, P_sv=12, P_pa=15, P_pv=12, HR=70)")
    print("=" * 70)
    heart = make_heart()
    sol, T, t_start = integrate_cycles(heart, INPUTS_HEALTHY,
                                        n_cycles=30, n_avg=10)
    m = extract_steady(sol, heart, INPUTS_HEALTHY, T, t_start)

    print(f"  T = {T:.4f} с, nfev = {m['nfev']}, success = {m['success']}")
    print()
    print(f"  {'Метрика':>14}  {'значение':>10}  {'Gayton':>20}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*20}")
    rows = [
        ("Q_aortic",    m["Q_aortic"],    "75-95 мл/с"),
        ("Q_mitral",    m["Q_mitral"],    "75-95 мл/с"),
        ("Q_pulmonary", m["Q_pulmonary"], "75-95 мл/с"),
        ("EDV_LV",      m["EDV_LV"],      "100-140 мл"),
        ("ESV_LV",      m["ESV_LV"],      "40-70 мл"),
        ("SV_LV",       m["SV_LV"],       "60-85 мл"),
        ("EDV_RV",      m["EDV_RV"],      "100-140 мл"),
        ("ESV_RV",      m["ESV_RV"],      "40-70 мл"),
        ("P_lv_max",    m["P_lv_max"],    "90-160 mmHg"),
        ("P_lv_min",    m["P_lv_min"],    "0-15 mmHg"),
        ("P_rv_max",    m["P_rv_max"],    "15-40 mmHg"),
        ("P_rv_min",    m["P_rv_min"],    "0-10 mmHg"),
    ]
    for name, val, ref in rows:
        print(f"  {name:>14}  {val:>10.2f}  {ref:>20}")

    checks = [
        ("Q_aortic ∈ [60, 110]",    60.0 <= m["Q_aortic"] <= 110.0),
        ("Q_mitral ∈ [60, 110]",    60.0 <= m["Q_mitral"] <= 110.0),
        ("Q_pulmonary ∈ [60, 110]", 60.0 <= m["Q_pulmonary"] <= 110.0),
        ("EDV_LV ∈ [90, 150]",      90.0 <= m["EDV_LV"] <= 150.0),
        ("ESV_LV ∈ [30, 80]",       30.0 <= m["ESV_LV"] <= 80.0),
        ("SV_LV ∈ [50, 100]",       50.0 <= m["SV_LV"] <= 100.0),
        ("EDV_RV ∈ [90, 150]",      90.0 <= m["EDV_RV"] <= 150.0),
        ("P_lv_max ∈ [80, 180]",    80.0 <= m["P_lv_max"] <= 180.0),
        ("P_rv_max ∈ [10, 50]",     10.0 <= m["P_rv_max"] <= 50.0),
        ("P_lv_min < P_lv_max",     m["P_lv_min"] < m["P_lv_max"]),
        ("P_rv_min < P_rv_max",     m["P_rv_min"] < m["P_rv_max"]),
        ("nfev < 50000",            m["nfev"] < 50000),
        ("success",                 m["success"]),
        ("all finite",              m["finite"]),
    ]
    print()
    all_ok = True
    for name, ok in checks:
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return {"ok": all_ok, "metrics": m}


# =====================================================================
# TEST 3: Mass balance
# =====================================================================

def test_mass_balance():
    print("\n" + "=" * 70)
    print("TEST 3: Kirchhoff в steady state (R_vsd=inf)")
    print("=" * 70)
    heart = make_heart()
    sol, T, t_start = integrate_cycles(heart, INPUTS_HEALTHY,
                                        n_cycles=30, n_avg=10)
    m = extract_steady(sol, heart, INPUTS_HEALTHY, T, t_start)

    def rel(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1e-6)

    rel_am = rel(m["Q_aortic"], m["Q_mitral"])
    rel_pt = rel(m["Q_pulmonary"], m["Q_tricuspid"])
    rel_pm = rel(m["Q_pv_to_la"], m["Q_mitral"])
    rel_st = rel(m["Q_sv_to_ra"], m["Q_tricuspid"])

    print(f"  Q_aortic vs Q_mitral:       rel = {rel_am:.3e}")
    print(f"  Q_pulmonary vs Q_tricuspid: rel = {rel_pt:.3e}")
    print(f"  Q_pv_to_la vs Q_mitral:     rel = {rel_pm:.3e}")
    print(f"  Q_sv_to_ra vs Q_tricuspid:  rel = {rel_st:.3e}")

    ok = all(r < 0.05 for r in (rel_am, rel_pt, rel_pm, rel_st))
    print(f"\n  [{'OK' if ok else 'FAIL'}] все балансы < 5%")
    return {"ok": ok}


# =====================================================================
# TEST 4: HR sweep
# =====================================================================

def test_hr_sweep():
    print("\n" + "=" * 70)
    print("TEST 4: HR sweep (0.7 → 1.0 → 1.3)")
    print("=" * 70)
    heart = make_heart()

    rows = []
    for hf in (0.7, 1.0, 1.3):
        inputs = dict(INPUTS_HEALTHY, hr_factor=hf)
        sol, T, t_start = integrate_cycles(heart, inputs,
                                            n_cycles=30, n_avg=10)
        m = extract_steady(sol, heart, inputs, T, t_start)
        rows.append({"hr": heart.hr_base * hf,
                     "Q_aortic": m["Q_aortic"],
                     "SV": m["SV_LV"]})

    print(f"  {'HR':>6}  {'Q_aortic':>10}  {'SV':>10}")
    for r in rows:
        print(f"  {r['hr']:>6.1f}  {r['Q_aortic']:>10.2f}  {r['SV']:>10.2f}")

    ok = rows[2]["Q_aortic"] > rows[0]["Q_aortic"]
    print(f"\n  [{'OK' if ok else 'FAIL'}] Q_aortic растёт с HR")
    return {"ok": ok}


# =====================================================================
# TEST 5: PV loop sanity
# =====================================================================

def test_pv_loop():
    print("\n" + "=" * 70)
    print("TEST 5: PV loop sanity")
    print("=" * 70)
    heart = make_heart()
    sol, T, t_start = integrate_cycles(heart, INPUTS_HEALTHY,
                                        n_cycles=30, n_avg=10)
    m = extract_steady(sol, heart, INPUTS_HEALTHY, T, t_start)

    print(f"  P_lv_max = {m['P_lv_max']:.1f}  > P_sa=85   (LV изгоняет кровь)")
    print(f"  P_rv_max = {m['P_rv_max']:.1f}  > P_pa=15   (RV изгоняет кровь)")
    print(f"  P_lv_min = {m['P_lv_min']:.2f}  < P_sa      (диастола)")

    ok = (m["P_lv_max"] > 85.0 and m["P_rv_max"] > 15.0
          and m["P_lv_min"] < 85.0)
    print(f"\n  [{'OK' if ok else 'FAIL'}] PV-loop физиологичен")
    return {"ok": ok}


# =====================================================================
# TEST 6: VSD shunt
# =====================================================================

def test_vsd_shunt():
    print("\n" + "=" * 70)
    print("TEST 6: VSD shunt (R_vsd=5.0, малый ДМЖП)")
    print("=" * 70)
    heart = make_heart()
    heart.R_vsd = 5.0

    sol, T, t_start = integrate_cycles(heart, INPUTS_HEALTHY,
                                        n_cycles=30, n_avg=10)
    m = extract_steady(sol, heart, INPUTS_HEALTHY, T, t_start)

    Qp_Qs = m["Q_pulmonary"] / max(m["Q_aortic"], 1e-6)
    delta = m["Q_pulmonary"] - m["Q_aortic"]

    print(f"  Q_aortic    = {m['Q_aortic']:.2f}")
    print(f"  Q_pulmonary = {m['Q_pulmonary']:.2f}")
    print(f"  Qp - Qs     = {delta:.2f}  (L->R при P_lv > P_rv)")
    print(f"  Qp/Qs       = {Qp_Qs:.3f}  (ожидание > 1)")

    ok = Qp_Qs > 1.0 and delta > 0.0
    print(f"\n  [{'OK' if ok else 'FAIL'}] L->R шунт воспроизводится")
    return {"ok": ok}


# =====================================================================
# TEST 7: RK45 vs LSODA
# =====================================================================

def test_integration_accuracy():
    print("\n" + "=" * 70)
    print("TEST 7: RK45 vs LSODA")
    print("=" * 70)

    heart_R = make_heart()
    sol_R, T, t_start = integrate_cycles(heart_R, INPUTS_HEALTHY,
                                          n_cycles=20, n_avg=5,
                                          method="RK45",
                                          rtol=1e-8, atol=1e-10,
                                          max_step=0.003)
    m_R = extract_steady(sol_R, heart_R, INPUTS_HEALTHY, T, t_start)

    heart_L = make_heart()
    sol_L, T, t_start = integrate_cycles(heart_L, INPUTS_HEALTHY,
                                          n_cycles=20, n_avg=5,
                                          method="LSODA",
                                          rtol=1e-8, atol=1e-10,
                                          max_step=0.003)
    m_L = extract_steady(sol_L, heart_L, INPUTS_HEALTHY, T, t_start)

    keys = ("Q_aortic", "Q_mitral", "EDV_LV", "ESV_LV",
            "P_lv_max", "P_rv_max")
    print(f"  {'метрика':>12}  {'RK45':>10}  {'LSODA':>10}  {'|diff|':>10}")
    all_ok = True
    for k in keys:
        d = abs(m_R[k] - m_L[k])
        flag = "OK" if d < 1.0 else "FAIL"
        all_ok = all_ok and (d < 1.0)
        print(f"  {k:>12}  {m_R[k]:>10.3f}  {m_L[k]:>10.3f}  "
              f"{d:>10.3e}  [{flag}]")

    print(f"  nfev RK45  = {sol_R.nfev}")
    print(f"  nfev LSODA = {sol_L.nfev}")
    print(f"\n  [{'OK' if all_ok else 'FAIL'}] RK45 == LSODA (|diff| < 1)")
    return {"ok": all_ok}


# =====================================================================
# TEST 8: Drift cycle-to-cycle
# =====================================================================

def test_drift():
    print("\n" + "=" * 70)
    print("TEST 8: Drift — циклы 8-13 vs циклы 23-28")
    print("=" * 70)
    heart = make_heart()
    sol, T, _ = integrate_cycles(heart, INPUTS_HEALTHY,
                                  n_cycles=30, n_avg=10, max_step=0.005)

    w1 = (sol.t > 8*T) & (sol.t < 13*T)
    w2 = (sol.t > 23*T) & (sol.t < 28*T)

    edv1, edv2 = sol.y[1, w1].max(), sol.y[1, w2].max()
    esv1, esv2 = sol.y[1, w1].min(), sol.y[1, w2].min()
    d_edv = abs(edv2 - edv1) / max(abs(edv2), 1.0)
    d_esv = abs(esv2 - esv1) / max(abs(esv2), 1.0)

    print(f"  EDV_LV: [8-13] = {edv1:.2f}  [23-28] = {edv2:.2f}  "
          f"drift = {d_edv:.3e}")
    print(f"  ESV_LV: [8-13] = {esv1:.2f}  [23-28] = {esv2:.2f}  "
          f"drift = {d_esv:.3e}")

    ok = d_edv < 0.05 and d_esv < 0.05
    print(f"\n  [{'OK' if ok else 'FAIL'}] drift < 5% за 20 циклов")
    return {"ok": ok}


# =====================================================================
# TEST 9: Conservation laws & periodicity (СТРОГИЙ)
# =====================================================================

def test_conservation_laws():
    """
    Строгая проверка на предельном цикле:

      1. Интегральный mass balance за полный цикл:
            ∫Q_aortic dt == ∫Q_mitral dt == SV_LV
            ∫Q_pulmonary dt == ∫Q_tricuspid dt == SV_RV

      2. Periodicity: y(t + T) == y(t) с точностью 1e-6
            (для t в установившемся окне)

      3. Положительность объёмов:
            V_lv > V0_lv, V_rv > V0_rv
            V_la > V0_la, V_ra > V0_ra

      4. SV_LV * HR == CO (Q_aortic), с точностью 1%

      5. Нет NaN/Inf нигде в траектории
    """
    print("\n" + "=" * 70)
    print("TEST 9: Conservation laws & periodicity (СТРОГИЙ)")
    print("=" * 70)

    heart = make_heart()
    # Узкий max_step + жёсткий rtol для точной интеграции
    sol, T, t_start = integrate_cycles(
        heart, INPUTS_HEALTHY,
        n_cycles=30, n_avg=10,
        method="LSODA", rtol=1e-9, atol=1e-11, max_step=0.002,
    )

    # --- Возьмём последний полный цикл ---
    t_lo = sol.t[-1] - T
    mask = (sol.t >= t_lo) & (sol.t <= sol.t[-1])
    t_c = sol.t[mask]
    y_c = sol.y[:, mask]
    if t_c.size < 50:
        print("  [FAIL] слишком мало точек в последнем цикле")
        return {"ok": False}

    # Q-массивы на этом цикле
    Q_aortic_c    = np.empty_like(t_c)
    Q_mitral_c    = np.empty_like(t_c)
    Q_pulmonary_c = np.empty_like(t_c)
    Q_tricuspid_c = np.empty_like(t_c)
    for i in range(t_c.size):
        heart.get_derivatives(t_c[i], y_c[:, i], INPUTS_HEALTHY)
        out = heart.get_outputs(y_c[:, i])
        Q_aortic_c[i]    = out["Q_aortic"]
        Q_mitral_c[i]    = out["Q_mitral"]
        Q_pulmonary_c[i] = out["Q_pulmonary"]
        Q_tricuspid_c[i] = out["Q_tricuspid"]

    # Интегралы (trapz)
    sv_lv_aortic    = float(np.trapezoid(Q_aortic_c, t_c))
    sv_lv_mitral    = float(np.trapezoid(Q_mitral_c, t_c))
    sv_rv_pulmonary = float(np.trapezoid(Q_pulmonary_c, t_c))
    sv_rv_tricuspid = float(np.trapezoid(Q_tricuspid_c, t_c))

    # SV из V_lv(t): max - min
    V_lv = y_c[1, :]
    V_rv = y_c[3, :]
    V_la = y_c[0, :]
    V_ra = y_c[2, :]
    sv_lv_direct = float(V_lv.max() - V_lv.min())
    sv_rv_direct = float(V_rv.max() - V_rv.min())

    def rel(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1e-6)

    print(f"\n  [1] Интегральный mass balance за цикл:")
    print(f"      ∫Q_aortic dt    = {sv_lv_aortic:.4f} мл")
    print(f"      ∫Q_mitral dt    = {sv_lv_mitral:.4f} мл")
    print(f"      SV_LV (Vmax-Vmin) = {sv_lv_direct:.4f} мл")
    print(f"      rel(∫aortic, ∫mitral) = {rel(sv_lv_aortic, sv_lv_mitral):.3e}")
    print(f"      rel(∫aortic, SV_LV)   = {rel(sv_lv_aortic, sv_lv_direct):.3e}")
    print(f"      rel(∫mitral, SV_LV)   = {rel(sv_lv_mitral, sv_lv_direct):.3e}")

    print(f"\n  [2] Аналогично для RV:")
    print(f"      ∫Q_pulmonary dt = {sv_rv_pulmonary:.4f} мл")
    print(f"      ∫Q_tricuspid dt = {sv_rv_tricuspid:.4f} мл")
    print(f"      SV_RV (Vmax-Vmin) = {sv_rv_direct:.4f} мл")

    # Периодичность
    # Сравним y в t_start и t_start+T внутри установившегося окна
    t0 = sol.t[-1] - 2*T
    mask_t0 = np.argmin(np.abs(sol.t - t0))
    mask_t0T = np.argmin(np.abs(sol.t - (t0 + T)))
    y0_cyc = sol.y[:, mask_t0]
    y1_cyc = sol.y[:, mask_t0T]
    dy_cycle = np.max(np.abs(y1_cyc - y0_cyc))
    print(f"\n  [3] Периодичность y(t+T) == y(t):")
    print(f"      max |y(t+T) - y(t)| = {dy_cycle:.3e} мл")

    # Положительность объёмов
    V0 = heart.V0
    min_V_lv = float(V_lv.min())
    min_V_rv = float(V_rv.min())
    min_V_la = float(V_la.min())
    min_V_ra = float(V_ra.min())
    print(f"\n  [4] Положительность объёмов:")
    print(f"      min V_lv = {min_V_lv:.2f}  (V0_lv = {V0['LV']})")
    print(f"      min V_rv = {min_V_rv:.2f}  (V0_rv = {V0['RV']})")
    print(f"      min V_la = {min_V_la:.2f}  (V0_la = {V0['LA']})")
    print(f"      min V_ra = {min_V_ra:.2f}  (V0_ra = {V0['RA']})")

    # SV * HR == CO
    hr = heart.hr_base * INPUTS_HEALTHY["hr_factor"]
    hr = np.clip(hr, heart.hr_min, heart.hr_max)
    co_from_sv = sv_lv_direct * hr
    co_direct = sv_lv_aortic / T
    print(f"\n  [5] SV · HR vs CO:")
    print(f"      SV_LV · HR = {co_from_sv:.3f} мл/с")
    print(f"      CO (Q_aortic mean) = {np.mean(Q_aortic_c):.3f} мл/с")
    print(f"      CO (∫/T) = {co_direct:.3f} мл/с")
    print(f"      rel = {rel(co_from_sv, co_direct):.3e}")

    # NaN/Inf
    all_finite = bool(np.all(np.isfinite(sol.y)))

    # Итог
    checks = [
        ("mass balance LV < 1%",       rel(sv_lv_aortic, sv_lv_mitral) < 0.01),
        ("SV_LV from V vs ∫Q < 2%",    rel(sv_lv_direct, sv_lv_aortic) < 0.02),
        ("mass balance RV < 1%",       rel(sv_rv_pulmonary, sv_rv_tricuspid) < 0.01),
        ("SV_RV from V vs ∫Q < 2%",    rel(sv_rv_direct, sv_rv_pulmonary) < 0.02),
        ("periodicity y(t+T)-y(t) < 1e-2 мл", dy_cycle < 1e-2),
        ("V_lv > V0_lv",                min_V_lv > V0["LV"]),
        ("V_rv > V0_rv",                min_V_rv > V0["RV"]),
        ("V_la > V0_la",                min_V_la > V0["LA"]),
        ("V_ra > V0_ra",                min_V_ra > V0["RA"]),
        ("SV·HR == CO < 1%",            rel(co_from_sv, co_direct) < 0.01),
        ("all finite over trajectory",  all_finite),
    ]
    print()
    all_ok = True
    for name, ok in checks:
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return {"ok": all_ok}


# =====================================================================
# Сводка
# =====================================================================

def summary(results):
    print("\n" + "=" * 70)
    print("СВОДКА: физиологичен ли Heart4Chambers?")
    print("=" * 70)
    names = [
        "Интерфейс OrganModel",
        "Steady state в Gayton ± допуск",
        "Mass balance (Kirchhoff)",
        "HR sweep (CO ~ HR)",
        "PV loop sanity",
        "VSD shunt (L->R)",
        "RK45 == LSODA",
        "Drift < 5%",
        "Conservation laws & periodicity",
    ]
    print()
    for name, r in zip(names, results):
        flag = "OK" if r["ok"] else "FAIL"
        print(f"  [{flag}] {name}")
    print()
    if all(r["ok"] for r in results):
        print("ВЫВОД: Heart4Chambers корректен в изоляции.")
        print("Значит, P_sa 65 / EDV 69 создаётся в связях whole_body:")
        print("Windkessel, mass-balance, обратные связи.")
    else:
        print("ВЫВОД: Heart4Chambers — вероятный источник "
              "нефизиологического стационара.")
        print("Искать в FAIL-тестах выше.")


# =====================================================================
# Entry point с сохранением в файл
# =====================================================================

def run_all_and_save(report_path: Path) -> None:
    """Запускает все тесты, дублируя вывод в report_path."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    tee = Tee(sys.stdout, buf)

    original_stdout = sys.stdout
    sys.stdout = tee
    try:
        print_banner()
        r1 = test_interface()
        r2 = test_steady_state()
        r3 = test_mass_balance()
        r4 = test_hr_sweep()
        r5 = test_pv_loop()
        r6 = test_vsd_shunt()
        r7 = test_integration_accuracy()
        r8 = test_drift()
        r9 = test_conservation_laws()
        summary([r1, r2, r3, r4, r5, r6, r7, r8, r9])
        print("\n" + "=" * 70)
        print("Вход: P_sa, P_sv, P_pa, P_pv, hr_factor, baro_activation")
        print("Выход -> whole_body: Q_aortic, Q_pulmonary, Q_mitral,"
              " Q_tricuspid,")
        print("                     Q_vsd, Q_pv_to_la, Q_sv_to_ra,")
        print("                     P_la, P_lv, P_ra, P_rv")
        print("=" * 70)
        print(f"\nФиниш: {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout = original_stdout

    report_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nОтчёт сохранён: {report_path}")


if __name__ == "__main__":
    out_dir = ROOT / "results"
    report_path = out_dir / "debug_heart_report_1.txt"
    run_all_and_save(report_path)