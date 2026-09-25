#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_baroreflex_2.py — изолированная проверка Baroreflex.

Версия v2: параметры читаются из ../config/physiology.yaml, а не
зашиты в код. Это устраняет расхождение между debug-скриптом и
реальным прогоном whole_body (который использует load_physiology).

Запуск:
    python tests/debug_baroreflex_2.py

Что подаем на вход:
  - P_sa (мм рт.ст.) — единственный вход baroreflex
  - state [HR] — текущая ЧСС

Что ожидаем на выходе:
  - HR_target = HR_base * (1 - gain*(P_sa - P_set)) clip 40-180
  - dHR/dt = (HR_target - HR)/tau
  - hr_factor = HR / HR_base (для Heart)
  - baro_activation = 1 + k_inotropy * max(1 - P_sa/P_set, 0)

Физиологические критерии:
  - При P_sa = P_set → HR_target = HR_base, baro_activation = 1.0
  - Gain-чувствительность gain * HR_base: норма 0.5–1.5 bpm/mmHg
  - HR range over P_sa ∈ [40, 140]: ≥ 20 bpm
  - baro_activation max при P_sa → 0: ≥ 1.5
  - tau = 2 с: 63 % за 2 с, 95 % за 6 с
"""

from __future__ import annotations
import sys
from pathlib import Path

# --- Путь к корню проекта (там, где whole_body.py, physio_config.py) ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import solve_ivp

from baroreflex import Baroreflex
from physio_config import load_physiology


# =====================================================================
# Загрузка параметров из physiology.yaml
# =====================================================================

def load_baroreflex_cfg() -> dict:
    """
    Возвращает словарь параметров для Baroreflex из physiology.yaml.

    Если load_physiology() по какой-то причине не находит файл, падаем
    с понятной ошибкой, а не с невнятным KeyError внутри Baroreflex.
    """
    cfg = load_physiology()             # читает config/physiology.yaml
    if "baroreflex" not in cfg:
        raise RuntimeError(
            f"physiology.yaml найден, но в нём нет секции 'baroreflex'.\n"
            f"Проверь: {ROOT / 'config' / 'physiology.yaml'}"
        )
    br_cfg = dict(cfg["baroreflex"])
    # Явно перечислим обязательные ключи, чтобы ошибка была понятной
    required = ("P_set", "HR_base", "gain", "tau", "k_inotropy")
    missing = [k for k in required if k not in br_cfg]
    if missing:
        raise RuntimeError(
            f"В baroreflex physiology.yaml отсутствуют ключи: {missing}"
        )
    return br_cfg


BARO_CFG = load_baroreflex_cfg()

# Удобные алиасы для тестов
P_SET     = float(BARO_CFG["P_set"])
HR_BASE   = float(BARO_CFG["HR_base"])
GAIN      = float(BARO_CFG["gain"])
TAU       = float(BARO_CFG["tau"])
K_INOTROPY = float(BARO_CFG["k_inotropy"])


def make_baro() -> Baroreflex:
    """Создаёт Baroreflex с параметрами из yaml."""
    return Baroreflex(**BARO_CFG)


# =====================================================================
# Заголовок с параметрами
# =====================================================================

def print_config_banner():
    print("=" * 70)
    print("Baroreflex: параметры загружены из config/physiology.yaml")
    print("=" * 70)
    print(f"  P_set       = {P_SET:.2f} мм рт.ст.")
    print(f"  HR_base     = {HR_BASE:.2f} уд/мин")
    print(f"  gain        = {GAIN:.5f} (1/мм рт.ст.)")
    print(f"  tau         = {TAU:.2f} с")
    print(f"  k_inotropy  = {K_INOTROPY:.2f}")
    print(f"  sensitivity = gain·HR_base = {GAIN * HR_BASE:.3f} bpm/mmHg")
    print("=" * 70)


# =====================================================================
# TEST 1: статическая кривая HR_target(P_sa)
# =====================================================================

def test_static_curve():
    print("\n" + "=" * 70)
    print("TEST 1: Статическая кривая HR_target(P_sa)")
    print("=" * 70)
    br = make_baro()

    header = f"{'P_sa':>6}  {'HR_target':>10}  {'ΔHR':>8}  {'slope':>8}  {'baro_act':>9}  tag"
    print(header)
    print("-" * len(header))

    for P_sa in [40, 60, 70, 80, 90, 100, 120, 140]:
        state = np.array([HR_BASE])
        br.get_derivatives(0, state, {"P_sa": P_sa})
        out = br.get_outputs(state)
        HRt = out["HR_target"]
        dHR = HRt - HR_BASE
        slope = dHR / (P_sa - P_SET) if P_sa != P_SET else 0.0
        tag = "TACHY" if HRt > HR_BASE else "BRADY" if HRt < HR_BASE else "NORM"
        print(f"{P_sa:>6}  {HRt:>10.2f}  {dHR:>+8.2f}  {slope:>+8.4f}  "
              f"{out['baro_activation']:>9.3f}  {tag}")

    sens = GAIN * HR_BASE
    flag = "OK" if 0.5 <= sens <= 1.5 else "WEAK"
    print(f"\nЧувствительность = {sens:.3f} bpm/mmHg  [{flag}]  "
          f"(норма 0.5–1.5)")


# =====================================================================
# TEST 2: step response — динамика
# =====================================================================

def test_step_response():
    print("\n" + "=" * 70)
    print("TEST 2: Ступенька давления 80 → 60 мм рт.ст. в t = 10 с")
    print("=" * 70)
    br = make_baro()

    def P_sa_func(t):
        return P_SET if t < 10.0 else 60.0

    def rhs(t, y):
        return br.get_derivatives(t, y, {"P_sa": P_sa_func(t)})

    y0 = br.get_initial_state()
    t_eval = np.linspace(0.0, 30.0, 301)
    sol = solve_ivp(rhs, (0.0, 30.0), y0, t_eval=t_eval, method="RK45")

    for t_check in [0.0, 9.9, 10.5, 12.0, 16.0, 30.0]:
        idx = int(np.argmin(np.abs(sol.t - t_check)))
        HR = float(sol.y[0, idx])
        P = P_sa_func(sol.t[idx])
        br.get_derivatives(sol.t[idx], [HR], {"P_sa": P})
        out = br.get_outputs([HR])
        print(f"t={sol.t[idx]:5.1f}s  P_sa={P:5.1f}  "
              f"HR={HR:6.2f}  HR_target={out['HR_target']:6.2f}  "
              f"baro_act={out['baro_activation']:.3f}")

    # Проверка экспоненциальной динамики
    idx_12 = int(np.argmin(np.abs(sol.t - 12.0)))
    HR_12 = float(sol.y[0, idx_12])
    HR_target_new = HR_BASE * (1.0 - GAIN * (60.0 - P_SET))
    HR_expected = HR_target_new + (HR_BASE - HR_target_new) * np.exp(-(12.0 - 10.0) / TAU)
    ok = "OK" if abs(HR_12 - HR_expected) < 0.5 else "FAIL"
    print(f"\nФакт HR@12с = {HR_12:.2f}, ожидание ~ {HR_expected:.2f}  [{ok}]")


# =====================================================================
# TEST 3: variability — синусоида P_sa
# =====================================================================

def test_variability():
    print("\n" + "=" * 70)
    print("TEST 3: Синусоида P_sa = 80 ± 20 мм рт.ст., T = 20 с")
    print("=" * 70)
    br = make_baro()

    def rhs(t, y):
        P = 80.0 + 20.0 * np.sin(2 * np.pi * t / 20.0)
        return br.get_derivatives(t, y, {"P_sa": P})

    y0 = br.get_initial_state()
    t_eval = np.linspace(0.0, 60.0, 601)
    sol = solve_ivp(rhs, (0.0, 60.0), y0, t_eval=t_eval, method="RK45")
    HR_min, HR_max = float(sol.y[0].min()), float(sol.y[0].max())
    amplitude = (HR_max - HR_min) / 2.0
    print(f"HR колеблется {HR_min:.1f}–{HR_max:.1f} уд/мин, "
          f"амплитуда ± {amplitude:.1f}")
    # Ожидание: амплитуда HR_target = gain·HR_base·(P_sa − P_set) ~ 1.05·20 = 21 bpm,
    # но HR фильтруется через tau=2с и период 20с, поэтому амплитуда меньше.
    # Реалистично: 50–90 % от амплитуды HR_target.
    print(f"Ожидаемая амплитуда ± {GAIN * HR_BASE * 20 * 0.7:.1f} ± 30 %")


# =====================================================================
# TEST 4: sanity-check
# =====================================================================

def test_physio_sanity():
    print("\n" + "=" * 70)
    print("TEST 4: Санити-чек формул Baroreflex с текущими параметрами")
    print("=" * 70)
    br = make_baro()

    # 4.1 При P_sa = P_set → HR_target = HR_base, baro_act = 1
    br.get_derivatives(0, [HR_BASE], {"P_sa": P_SET})
    out = br.get_outputs([HR_BASE])
    assert abs(out["HR_target"] - HR_BASE) < 1e-6
    assert abs(out["baro_activation"] - 1.0) < 1e-6
    print("✓ P_sa = P_set → HR_target = HR_base, baro_act = 1.0")

    # 4.2 Клиппинг
    br.get_derivatives(0, [HR_BASE], {"P_sa": 0.0})
    out = br.get_outputs([HR_BASE])
    assert 40.0 <= out["HR_target"] <= 180.0
    print(f"✓ Клиппинг: P_sa = 0 → HR_target = {out['HR_target']:.2f} (clip 40–180)")

    # 4.3 baro_activation не растёт при гипертензии
    br.get_derivatives(0, [HR_BASE], {"P_sa": 120.0})
    out = br.get_outputs([HR_BASE])
    assert abs(out["baro_activation"] - 1.0) < 1e-6
    print(f"✓ При P_sa = 120 baro_act = {out['baro_activation']:.3f} (не растёт)")

    # 4.4 baro_activation при гипотензии
    br.get_derivatives(0, [HR_BASE], {"P_sa": 40.0})
    out = br.get_outputs([HR_BASE])
    expected = 1.0 + K_INOTROPY * max(1.0 - 40.0 / P_SET, 0.0)
    assert abs(out["baro_activation"] - expected) < 1e-6
    print(f"✓ При P_sa = 40 baro_act = {out['baro_activation']:.3f} "
          f"(формула 1 + k·(1 − P/P_set))")

    # 4.5 Чувствительность
    sens = GAIN * HR_BASE
    flag = "OK" if 0.5 <= sens <= 1.5 else "WEAK"
    print(f"{'✓' if flag == 'OK' else '⚠'} Чувствительность = {sens:.3f} bpm/mmHg  [{flag}]")

    # 4.6 HR range на [40, 140]
    HRs = []
    for P in (40.0, 140.0):
        br.get_derivatives(0, [HR_BASE], {"P_sa": P})
        HRs.append(br.get_outputs([HR_BASE])["HR_target"])
    hr_range = max(HRs) - min(HRs)
    flag = "OK" if hr_range >= 20.0 else "FAIL"
    print(f"{'✓' if flag == 'OK' else '✗'} HR range over [40, 140] = "
          f"{hr_range:.1f} уд/мин  [{flag}, норма ≥ 20]")

    # 4.7 baro_activation max при P_sa = 0
    br.get_derivatives(0, [HR_BASE], {"P_sa": 0.0})
    ba_max = br.get_outputs([HR_BASE])["baro_activation"]
    flag = "OK" if ba_max >= 1.5 else "FAIL"
    print(f"{'✓' if flag == 'OK' else '✗'} baro_activation max = "
          f"{ba_max:.3f}  [{flag}, норма ≥ 1.5]")


# =====================================================================
# СВОДКА
# =====================================================================

def summary():
    print("\n" + "=" * 70)
    print("СВОДКА: физиологичен ли Baroreflex с текущими параметрами?")
    print("=" * 70)

    checks = []

    # 1. Gain-чувствительность
    sens = GAIN * HR_BASE
    checks.append(("Gain-чувствительность (0.5–1.5 bpm/mmHg)",
                   sens, 0.5, 1.5, 0.5 <= sens <= 1.5))

    # 2. HR range [40, 140]
    br = make_baro()
    HRs = []
    for P in (40.0, 140.0):
        br.get_derivatives(0, [HR_BASE], {"P_sa": P})
        HRs.append(br.get_outputs([HR_BASE])["HR_target"])
    hr_range = max(HRs) - min(HRs)
    checks.append(("HR range over P_sa ∈ [40, 140] ≥ 20 bpm",
                   hr_range, 20.0, 200.0, hr_range >= 20.0))

    # 3. baro_activation max
    br.get_derivatives(0, [HR_BASE], {"P_sa": 20.0})
    ba_max = br.get_outputs([HR_BASE])["baro_activation"]
    checks.append(("baro_activation max ≥ 1.5",
                   ba_max, 1.5, 5.0, ba_max >= 1.5))

    # 4. tau-соответствие (проверяем аналитически, что модель с tau)
    y_num = np.array([HR_BASE])
    P_new = 60.0
    br.get_derivatives(0, y_num, {"P_sa": P_new})
    HR_target = br.get_outputs(y_num)["HR_target"]
    dt = 0.001
    t_check = TAU
    for _ in range(int(t_check / dt)):
        dy = br.get_derivatives(0, y_num, {"P_sa": P_new})
        y_num = y_num + dt * dy
    HR_analytic = HR_target + (HR_BASE - HR_target) * np.exp(-t_check / TAU)
    err = abs(y_num[0] - HR_analytic)
    checks.append((f"tau = {TAU:.2f} с (err после 1·tau)",
                   err, 0.0, 0.05, err < 0.05))

    print()
    for name, val, lo, hi, ok in checks:
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}] {name:<48} = {val:.3f}  (норма [{lo}, {hi}])")
    print()

    all_ok = all(c[4] for c in checks)
    if all_ok:
        print("ВЫВОД: Baroreflex физиологичен и готов к интеграции с Heart.")
    else:
        print("ВЫВОД: параметры требуют правки в physiology.yaml.")
        print("  Рекомендации:")
        if not checks[0][4]:
            print(f"    • gain: текущий {GAIN}, попробуй 0.015 "
                  f"(sens = {0.015 * HR_BASE:.3f} bpm/mmHg)")
        if not checks[2][4]:
            print(f"    • k_inotropy: текущий {K_INOTROPY}, попробуй 1.5 "
                  f"(baro_act max = {1.0 + 1.5:.2f})")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    print_config_banner()
    test_static_curve()
    test_step_response()
    test_variability()
    test_physio_sanity()
    summary()
    print("\n" + "=" * 70)
    print("Вход Baroreflex: P_sa (float).")
    print("Выход Baroreflex → Heart._update_parameters(): "
          "hr_factor, baro_activation.")
    print("=" * 70)