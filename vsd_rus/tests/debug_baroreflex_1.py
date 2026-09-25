#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_baroreflex.py — изолированный тест Baroreflex.

Запуск:
    python debug_baroreflex.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.integrate import solve_ivp
from baroreflex import Baroreflex

# =====================================================================
# 1. Параметры для проверки
# =====================================================================

HR_BASE = 70.0
P_SET   = 80.0
TAU     = 2.0

# Проверяем несколько наборов (gain, k_inotropy):
# текущий и «физиологический»
CASES = [
    {"name": "current (gain=0.002)",   "gain": 0.002, "k_inotropy": 0.5},
    {"name": "physiological (gain=0.015)", "gain": 0.015, "k_inotropy": 1.5},
]


def make_baro(gain, k_inotropy):
    return Baroreflex(P_set=P_SET, HR_base=HR_BASE,
                      gain=gain, tau=TAU, k_inotropy=k_inotropy)


def simulate_step(baro, P_profile, t_end=60.0, dt=0.05):
    """
    P_profile(t) → P_sa(t). Интегрируем dHR/dt = (target - HR)/tau.
    """
    t_eval = np.arange(0.0, t_end, dt)
    HR = np.zeros_like(t_eval)
    HR_target_arr = np.zeros_like(t_eval)
    baro_act_arr = np.zeros_like(t_eval)

    y = baro.get_initial_state()
    for i, t in enumerate(t_eval):
        P_sa = P_profile(t)
        baro.get_derivatives(t, y, {'P_sa': P_sa})
        out = baro.get_outputs(y)
        HR[i] = out['HR']
        HR_target_arr[i] = out['HR_target']
        baro_act_arr[i] = out['baro_activation']
        # Euler (dt=0.05, tau=2 → устойчиво)
        dy = baro.get_derivatives(t, y, {'P_sa': P_sa})
        y = y + dt * dy

    return t_eval, HR, HR_target_arr, baro_act_arr


# =====================================================================
# 2. Тест 1: аналитическая проверка gain
# =====================================================================

def test_static_gain():
    print("=" * 70)
    print("Тест 1: статический gain (HR_target = HR_base·(1 − gain·ΔP))")
    print("=" * 70)

    P_range = [40, 60, 70, 80, 90, 100, 120, 140]

    for case in CASES:
        baro = make_baro(case["gain"], case["k_inotropy"])
        print(f"\n  {case['name']}:")
        print(f"    {'P_sa':>6}  {'HR_target':>10}  {'ΔHR':>7}  {'bpm/mmHg':>10}")
        for P_sa in P_range:
            y = baro.get_initial_state()
            baro.get_derivatives(0, y, {'P_sa': P_sa})
            out = baro.get_outputs(y)
            HRt = out['HR_target']
            dHR = HRt - HR_BASE
            slope = dHR / (P_sa - P_SET) if P_sa != P_SET else 0.0
            print(f"    {P_sa:>6}  {HRt:>10.2f}  {dHR:>+7.2f}  {slope:>10.4f}")

    print("\n  [Ожидание] Норма baroreflex sensitivity: 0.5–1.5 bpm/mmHg.")
    print("              gain·HR_base = чувствительность:")
    for case in CASES:
        sens = case["gain"] * HR_BASE
        flag = "OK" if 0.5 <= sens <= 1.5 else "WEAK"
        print(f"                {case['name']:<35} {sens:.3f} bpm/mmHg  [{flag}]")


# =====================================================================
# 3. Тест 2: step response — динамика
# =====================================================================

def test_step_response():
    print("\n" + "=" * 70)
    print("Тест 2: step response (P_sa: 80 → 60 → 100 → 80)")
    print("=" * 70)

    def P_profile(t):
        if t < 20:
            return P_SET
        elif t < 40:
            return 60.0   # гипотензия
        elif t < 55:
            return 100.0  # гипертензия
        else:
            return P_SET

    for case in CASES:
        print(f"\n  {case['name']}:")
        baro = make_baro(case["gain"], case["k_inotropy"])
        t, HR, HRt, ba = simulate_step(baro, P_profile, t_end=80.0)

        # Средние значения в окнах через 15 с после каждого step
        for t_lo, t_hi, label in [(15, 20, "t=20, P=80 baseline"),
                                   (35, 40, "t=40, P=60"),
                                   (50, 55, "t=55, P=100"),
                                   (75, 80, "t=80, P=80 return")]:
            m = (t >= t_lo) & (t <= t_hi)
            print(f"    {label:<26}  HR = {HR[m].mean():6.2f}  "
                  f"target = {HRt[m].mean():6.2f}  "
                  f"baro_act = {ba[m].mean():.4f}")


# =====================================================================
# 4. Тест 3: барo_activation
# =====================================================================

def test_baro_activation():
    print("\n" + "=" * 70)
    print("Тест 3: baro_activation(P_sa) — только при P_sa < P_set")
    print("=" * 70)

    baro = make_baro(0.002, 0.5)
    print(f"    {'P_sa':>6}  {'baro_activation':>18}")
    for P_sa in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
        y = baro.get_initial_state()
        baro.get_derivatives(0, y, {'P_sa': P_sa})
        out = baro.get_outputs(y)
        print(f"    {P_sa:>6}  {out['baro_activation']:>18.4f}")

    print("\n  [Ожидание] 1.0 при P_sa ≥ 80, растёт до 1.5 при P_sa → 0.")


# =====================================================================
# 5. Тест 4: time constant
# =====================================================================

def test_time_constant():
    print("\n" + "=" * 70)
    print("Тест 4: time constant (должен совпасть с tau=2.0)")
    print("=" * 70)

    baro = make_baro(0.002, 0.5)
    y0 = np.array([HR_BASE])
    P_new = 60.0
    y = baro.get_initial_state()
    baro.get_derivatives(0, y, {'P_sa': P_new})
    HR_target = baro.get_outputs(y)['HR_target']

    # Аналитика: HR(t) = HR_target + (HR_0 − HR_target)·exp(−t/tau)
    for t_check in [2, 4, 6, 10]:
        y_num = np.array([HR_BASE])
        dt = 0.01
        for _ in range(int(t_check / dt)):
            dy = baro.get_derivatives(0, y_num, {'P_sa': P_new})
            y_num = y_num + dt * dy
        HR_num = y_num[0]
        HR_ana = HR_target + (HR_BASE - HR_target) * np.exp(-t_check / TAU)
        err = abs(HR_num - HR_ana)
        print(f"    t = {t_check:2d}s: numeric = {HR_num:.4f}, "
              f"analytic = {HR_ana:.4f}, err = {err:.4e}")


# =====================================================================
# 6. Сводный отчёт
# =====================================================================

def summary():
    print("\n" + "=" * 70)
    print("СВОДКА: физиологичен ли Baroreflex в текущей конфигурации?")
    print("=" * 70)

    checks = []

    # 1. Gain
    gain_default = 0.002
    sens = gain_default * HR_BASE
    checks.append(("Gain-чувствительность (0.5–1.5 bpm/mmHg)",
                   sens, 0.5, 1.5, sens >= 0.5))

    # 2. HR_range at P_sa = [40, 140]
    baro = make_baro(gain_default, 0.5)
    HRs = []
    for P in [40, 140]:
        y = baro.get_initial_state()
        baro.get_derivatives(0, y, {'P_sa': P})
        HRs.append(baro.get_outputs(y)['HR_target'])
    hr_range = max(HRs) - min(HRs)
    checks.append(("HR range over P_sa ∈ [40, 140] ≥ 20 bpm",
                   hr_range, 20, 100, hr_range >= 20))

    # 3. baro_activation max
    y = baro.get_initial_state()
    baro.get_derivatives(0, y, {'P_sa': 20})
    ba_max = baro.get_outputs(y)['baro_activation']
    checks.append(("baro_activation max ≥ 1.5",
                   ba_max, 1.5, 3.0, ba_max >= 1.5))

    print()
    for name, val, lo, hi, ok in checks:
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}] {name:<50} = {val:.3f}  "
              f"(норма [{lo}, {hi}])")

    print()


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    test_static_gain()
    test_step_response()
    test_baro_activation()
    test_time_constant()
    summary()