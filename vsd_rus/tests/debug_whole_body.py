#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_whole_body.py — изолированная диагностика WholeBodyModel
для выяснения причин инверсии Qp/Qs, низкого P_sa и большого nfev.

Запуск:
    python debug_whole_body.py
    python debug_whole_body.py --variant vsd_r5            # по умолчанию
    python debug_whole_body.py --variant healthy
    python debug_whole_body.py --variant vsd_r1
    python debug_whole_body.py --variant all

Проверяет три сценария:
    1. healthy       — R_vsd = inf (нет ДМЖП)
    2. vsd_r5        — R_vsd = 5.0 (малый ДМЖП, как в Stage 1)
    3. vsd_r1        — R_vsd = 1.0 (большой ДМЖП)

Для каждого сценария печатает:
    • Сходимость y0 после калибровки
    • Динамику P_sa, V_lv, Qp, Qa, Q_vsd по окнам времени
    • Знак и величину Q_vsd (проверка направления шунта)
    • P_lv vs P_rv на систолическом пике (причина направления шунта)
    • Сравнение V_sv и V_sv_target (проверка Windkessel-релаксации)
    • Физиологичность итоговой точки
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whole_body import WholeBodyModel


# =====================================================================
# Сценарии
# =====================================================================

SCENARIOS = {
    "healthy": {
        "label": "Здоровый (R_vsd=inf)",
        "R_vsd": np.inf,
        "flow_dependent_lungs": False,
    },
    "vsd_r5": {
        "label": "Малый ДМЖП (R_vsd=5.0)",
        "R_vsd": 5.0,
        "flow_dependent_lungs": False,
    },
    "vsd_r1": {
        "label": "Большой ДМЖП (R_vsd=1.0)",
        "R_vsd": 1.0,
        "flow_dependent_lungs": True,
    },
}


# =====================================================================
# Вспомогательные функции
# =====================================================================

def _build_model(scenario: dict) -> WholeBodyModel:
    return WholeBodyModel(
        heart_params={
            'hr': 70,
            'R_vsd': scenario["R_vsd"],
            'R_venous_sys':  0.08,
            'R_venous_pulm': 0.03,
        },
        flow_dependent_lungs=scenario["flow_dependent_lungs"],
        peripheral_params={'VO2_base': 1.5},
        target_MAP=85.0, target_CO=83.0,
    )


def _collect_data(model: WholeBodyModel, sol) -> dict:
    """Прогоняет compute_outputs по всем точкам решения."""
    keys = None
    rows = []
    for i in range(sol.t.size):
        out = model.compute_outputs(sol.t[i], sol.y[:, i])
        if keys is None:
            keys = list(out.keys())
        rows.append([out[k] for k in keys])
    data = {k: np.array([r[j] for r in rows], dtype=float)
            for j, k in enumerate(keys)}
    data["t"] = np.asarray(sol.t, dtype=float)
    return data


def _window_mean(data: dict, key: str, mask: np.ndarray) -> float:
    arr = data.get(key)
    if arr is None:
        return float("nan")
    m = mask & np.isfinite(arr)
    if not np.any(m):
        return float("nan")
    return float(np.mean(arr[m]))


def _print_convergence(data: dict, label: str) -> None:
    """Печатает средние по временным окнам — оценка сходимости."""
    print(f"\n--- {label}: CONVERGENCE WINDOWS ---")
    print(f"{'window':>14}  {'P_sa':>7}  {'V_lv':>7}  {'V_rv':>7}  "
          f"{'HR':>5}  {'Qa':>7}  {'Qp':>7}  {'Q_vsd':>8}  {'P_sv':>6}")
    t = data["t"]
    for t_lo, t_hi in [(0, 150), (150, 300), (300, 450), (450, 600)]:
        m = (t >= t_lo) & (t <= t_hi)
        if not np.any(m):
            continue
        print(f"[{t_lo:4d}-{t_hi:4d}]  "
              f"{_window_mean(data, 'P_sa', m):7.2f}  "
              f"{_window_mean(data, 'V_lv', m):7.1f}  "
              f"{_window_mean(data, 'V_rv', m):7.1f}  "
              f"{_window_mean(data, 'HR', m):5.1f}  "
              f"{_window_mean(data, 'Q_aortic', m):7.2f}  "
              f"{_window_mean(data, 'Q_pulmonary', m):7.2f}  "
              f"{_window_mean(data, 'Q_vsd', m):+8.2f}  "
              f"{_window_mean(data, 'P_sv', m):6.2f}")


def _print_steady(data: dict, label: str, model=None, sol=None) -> None:
    """Средние за последние 10 циклов + ключевые физиологические метрики."""
    t = data["t"]
    HR_mean = _window_mean(data, "HR", t > t[-1] * 0.5)
    T = 60.0 / max(HR_mean, 1e-6) if np.isfinite(HR_mean) else 0.86
    win = t > (t[-1] - 10.0 * T)

    P_sa = _window_mean(data, "P_sa", win)
    P_pa = _window_mean(data, "P_pa", win)
    Qa = _window_mean(data, "Q_aortic", win)
    Qp = _window_mean(data, "Q_pulmonary", win)
    Q_vsd = _window_mean(data, "Q_vsd", win)
    P_sv = _window_mean(data, "P_sv", win)
    P_lv = _window_mean(data, "P_lv", win)
    P_rv = _window_mean(data, "P_rv", win)
    EDV_LV = float(np.max(data["V_lv"][win])) if np.any(win) else float("nan")
    EDV_RV = float(np.max(data["V_rv"][win])) if np.any(win) else float("nan")
    V_sv = _window_mean(data, "V_sv", win)
    V_sv_target = _window_mean(data, "V_sv_target", win)
    Qp_Qs = Qp / max(Qa, 1e-6)

    print(f"\n--- {label}: STEADY (last 10 cycles, T={T:.3f} s) ---")
    print(f"  P_sa          = {P_sa:7.2f} мм рт.ст.   (target ≈ 85)")
    print(f"  P_pa          = {P_pa:7.2f} мм рт.ст.   (норма 15–20)")
    print(f"  P_sv          = {P_sv:7.2f} мм рт.ст.")
    print(f"  Q_aortic (Qs) = {Qa:7.2f} мл/с        (target ≈ 83)")
    print(f"  Q_pulmonary   = {Qp:7.2f} мл/с")
    print(f"  Q_vsd         = {Q_vsd:+7.2f} мл/с        (+L→R, −R→L)")
    print(f"  Qp / Qs       = {Qp_Qs:7.3f}           (норма ≈ 1.0, L→R > 1, R→L < 1)")
    print(f"  HR            = {HR_mean:7.2f} уд/мин")
    print(f"  EDV_LV        = {EDV_LV:7.1f} мл")
    print(f"  EDV_RV        = {EDV_RV:7.1f} мл")
    print(f"  V_sv          = {V_sv:7.1f} мл        (target {V_sv_target:.1f})")
    print(f"  P_lv (mean)   = {P_lv:7.2f} мм рт.ст.")
    print(f"  P_rv (mean)   = {P_rv:7.2f} мм рт.ст.")

    Q_mitral = _window_mean(data, "Q_mitral", win)
    Q_tricuspid = _window_mean(data, "Q_tricuspid", win)

    # Стационарный баланс: ∮ dV_lv = 0 и ∮ dV_rv = 0 за цикл
    balance_lv = Q_mitral - Qa - Q_vsd
    balance_rv = Q_tricuspid + Q_vsd - Qp

    print()
    print(f"  --- MASS BALANCE (should be ~0 in steady state) ---")
    print(f"  Q_mitral      = {Q_mitral:7.2f} мл/с")
    print(f"  Q_tricuspid   = {Q_tricuspid:7.2f} мл/с")
    print(f"  LV: Q_mitral - Q_aortic - Q_vsd = {balance_lv:+.2f} мл/с")
    print(f"  RV: Q_tricuspid + Q_vsd - Qp    = {balance_rv:+.2f} мл/с")
    if abs(balance_lv) > 2.0 or abs(balance_rv) > 2.0:
        print(f"  ⚠ Mass balance violated → система НЕ в стационаре")

    # Дополнительная диагностика: мгновенный срез цикла с фиксированным y_end
    # НЕ заменяет t-усреднение, только перекрёстная проверка.
    t_end_ss = float(data["t"][-1])
    y_end_ss = sol.y[:, -1]
    cycle_info = model.cycle_averaged_flows(t_end_ss, y_end_ss, n_pts=60)
    print()
    print(f"  --- CYCLE-AVERAGED (fixed y_end, cross-check) ---")
    print(f"  Qp_cycle_mean    = {cycle_info['Qp_cycle_mean']:7.2f} мл/с")
    print(f"  Qs_cycle_mean    = {cycle_info['Qs_cycle_mean']:7.2f} мл/с")
    print(f"  Qp/Qs_cycle      = {cycle_info['Qp_Qs_cycle']:7.3f}")
    print(f"  mass_balance_err = {cycle_info['mass_balance_error']:+.2f}")

    # --- Диагностика направления шунта ---
    print()
    if Q_vsd > 1.0:
        print(f"  ► ШУНТ: L→R  ({Q_vsd:+.1f} мл/с), Qp/Qs={Qp_Qs:.3f}")
    elif Q_vsd < -1.0:
        print(f"  ► ШУНТ: R→L  ({Q_vsd:+.1f} мл/с), Qp/Qs={Qp_Qs:.3f}")
    else:
        print(f"  ► ШУНТ: гемодинамически незначим ({Q_vsd:+.1f} мл/с)")

    # --- Физиологичность ---
    print()
    warnings = []
    if not (40.0 < P_sa < 180.0):
        warnings.append(f"P_sa={P_sa:.1f} вне [40, 180]")
    if not (5.0 < P_pa < 80.0):
        warnings.append(f"P_pa={P_pa:.1f} вне [5, 80]")
    if Qa < 25.0:
        warnings.append(f"Q_aortic={Qa:.1f} < 25 (системный коллапс)")
    if Qp_Qs > 5.0:
        warnings.append(f"Qp/Qs={Qp_Qs:.2f} > 5 (экстремальный шунт)")
    if Qp_Qs < 0.5:
        warnings.append(f"Qp/Qs={Qp_Qs:.2f} < 0.5 (нефизиологичная инверсия)")
    if EDV_LV < 50.0:
        warnings.append(f"EDV_LV={EDV_LV:.1f} < 50 (недонаполнение ЛЖ)")
    if abs(V_sv - V_sv_target) > 0.1 * max(V_sv_target, 1.0):
        warnings.append(
            f"V_sv={V_sv:.0f} ≠ target {V_sv_target:.0f} "
            f"(Windkessel не сошёлся)"
        )
    if warnings:
        print("  ⚠ ФИЗИОЛОГИЧЕСКИЕ ПРЕДУПРЕЖДЕНИЯ:")
        for w in warnings:
            print(f"     - {w}")
    else:
        print("  ✓ Все физиологические метрики в норме")


def _print_one_cycle(data: dict, label: str) -> None:
    """Детали одного последнего кардиоцикла: P_lv vs P_rv для проверки знака Q_vsd."""
    t = data["t"]
    HR_mean = _window_mean(data, "HR", t > t[-1] * 0.5)
    T = 60.0 / max(HR_mean, 1e-6)
    cycle = t > (t[-1] - T)
    if not np.any(cycle):
        return

    print(f"\n--- {label}: ONE CYCLE (last T={T:.3f} s) ---")
    for k in ("V_lv", "V_rv", "P_lv", "P_rv", "Q_mitral", "Q_tricuspid",
              "Q_aortic", "Q_pulmonary", "Q_vsd"):
        if k not in data:
            continue
        arr = data[k][cycle]
        print(f"  {k:14s}  min={arr.min():+8.3f}  "
              f"max={arr.max():+8.3f}  mean={arr.mean():+8.3f}")

    # Систолический пик P_lv vs P_rv — решающий момент для направления шунта
    if "P_lv" in data and "P_rv" in data:
        P_lv_c = data["P_lv"][cycle]
        P_rv_c = data["P_rv"][cycle]
        i_peak = int(np.argmax(P_lv_c))
        print(f"\n  Систолический пик (i={i_peak}):")
        print(f"    P_lv = {P_lv_c[i_peak]:+7.2f} мм рт.ст.")
        print(f"    P_rv = {P_rv_c[i_peak]:+7.2f} мм рт.ст.")
        print(f"    ΔP   = {P_lv_c[i_peak] - P_rv_c[i_peak]:+7.2f} мм рт.ст.  "
              f"(>0 → L→R, <0 → R→L в эту фазу)")


# =====================================================================
# Один сценарий
# =====================================================================

def run_scenario(name: str, scenario: dict) -> None:
    label = scenario["label"]
    print("\n" + "=" * 78)
    print(f"  SCENARIO: {name}  —  {label}")
    print("=" * 78)

    # --- Построение модели ---
    try:
        model = _build_model(scenario)
    except Exception as e:
        print(f"  ✗ build_model FAILED: {e}")
        return

    # --- Калибровка ---
    t_calib_eff = 400.0
    print(f"\n  Калибровка t_calib={t_calib_eff:.0f} с...")
    try:
        y0 = model.calibrate_initial_state(t_calib=t_calib_eff)
    except Exception as e:
        print(f"  ✗ calibrate_initial_state FAILED: {e}")
        return

    heart_slc = model.idx['heart']
    V_blood = y0[model.idx['blood']][0]
    P_sa_0 = y0[model.idx['sys_art']][0]
    print(f"  y0: heart={y0[heart_slc]}  "
          f"V_blood={V_blood:.1f}  P_sa={P_sa_0:.2f}")

    # --- Симуляция ---
    t_end_sim = 600.0
    n_pts = 12000
    print(f"\n  Симуляция 0..{t_end_sim:.0f} с, LSODA")
    t_eval = np.linspace(0.0, t_end_sim, n_pts)
    try:
        sol = model.simulate(
            (0.0, t_end_sim), t_eval=t_eval, y0=y0,
            method='LSODA', rtol=1e-4, atol=1e-5, max_step=0.15,
        )
    except Exception as e:
        print(f"  ✗ simulate FAILED: {e}")
        return

    print(f"  solver: success={getattr(sol, 'success', False)}  "
          f"nfev={sol.nfev}  njev={getattr(sol, 'njev', 0)}  "
          f"t_end={sol.t[-1]:.1f}  n_points={sol.t.size}")

    if not getattr(sol, "success", False):
        print("  ✗ Solver did not converge")
        return

    # --- Сбор выходов ---
    try:
        data = _collect_data(model, sol)
    except Exception as e:
        print(f"  ✗ collect_outputs FAILED: {e}")
        return

    # --- Печать ---
    _print_convergence(data, label)
    _print_one_cycle(data, label)
    _print_steady(data, label, model=model, sol=sol)


# =====================================================================
# main
# =====================================================================

def main(variant: str) -> None:
    print("=" * 78)
    print(f"  DEBUG WholeBodyModel — variant: {variant}")
    print("=" * 78)

    if variant == "all":
        names = list(SCENARIOS.keys())
    elif variant in SCENARIOS:
        names = [variant]
    else:
        print(f"  ✗ Unknown variant '{variant}'. "
              f"Доступно: {list(SCENARIOS.keys())} или 'all'.")
        sys.exit(1)

    for name in names:
        run_scenario(name, SCENARIOS[name])

    print("\n" + "=" * 78)
    print("  Done.")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Isolated diagnostic of WholeBodyModel."
    )
    parser.add_argument(
        "--variant", "-V", type=str, default="vsd_r5",
        help="Сценарий: healthy | vsd_r5 | vsd_r1 | all (default: vsd_r5)",
    )
    args = parser.parse_args()
    main(variant=args.variant)