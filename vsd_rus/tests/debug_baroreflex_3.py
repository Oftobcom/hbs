#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_baroreflex_3.py — изолированная проверка Baroreflex (v3).

Что нового относительно v2:
  - TEST 2 (step response) интегрируется в ДВА этапа: [0, 10] и [10, 30],
    чтобы разрыв P_sa(t) в t=10 не попадал внутрь шага RK45.
  - Используются жёсткие допуски (rtol=1e-8, atol=1e-10) для RK45.
  - Сравнение идёт не с «наивной» аналитикой от HR(0)=HR_base,
    а с аналитикой от ТОЧНОГО значения HR(10) из первого этапа.
  - Добавлена сверка RK45 vs LSODA — если оба совпали, артефакта нет.
  - Проверка [OK/FAIL] использует порог 0.05 уд/мин (типичная точность
    RK45 с rtol=1e-8 при tau=2 с).

Запуск:
    python tests/debug_baroreflex_3.py
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import solve_ivp

from baroreflex import Baroreflex
from physio_config import load_physiology


# =====================================================================
# 0. Загрузка параметров из config/physiology.yaml
# =====================================================================

def load_baroreflex_cfg() -> dict:
    cfg = load_physiology()
    if "baroreflex" not in cfg:
        raise RuntimeError(
            f"physiology.yaml найден, но без секции 'baroreflex'.\n"
            f"Проверь: {ROOT / 'config' / 'physiology.yaml'}"
        )
    br_cfg = dict(cfg["baroreflex"])
    required = ("P_set", "HR_base", "gain", "tau", "k_inotropy")
    missing = [k for k in required if k not in br_cfg]
    if missing:
        raise RuntimeError(
            f"В baroreflex physiology.yaml отсутствуют ключи: {missing}"
        )
    return br_cfg


BARO_CFG   = load_baroreflex_cfg()
P_SET      = float(BARO_CFG["P_set"])
HR_BASE    = float(BARO_CFG["HR_base"])
GAIN       = float(BARO_CFG["gain"])
TAU        = float(BARO_CFG["tau"])
K_INOTROPY = float(BARO_CFG["k_inotropy"])


def make_baro() -> Baroreflex:
    return Baroreflex(**BARO_CFG)


def print_config_banner():
    print("=" * 70)
    print("Baroreflex v3: параметры загружены из config/physiology.yaml")
    print("=" * 70)
    print(f"  P_set       = {P_SET:.2f} мм рт.ст.")
    print(f"  HR_base     = {HR_BASE:.2f} уд/мин")
    print(f"  gain        = {GAIN:.5f} (1/мм рт.ст.)")
    print(f"  tau         = {TAU:.2f} с")
    print(f"  k_inotropy  = {K_INOTROPY:.2f}")
    print(f"  sensitivity = gain·HR_base = {GAIN * HR_BASE:.3f} bpm/mmHg")
    print("=" * 70)


# =====================================================================
# 1. Статическая кривая
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
    print(f"\nЧувствительность = {sens:.3f} bpm/mmHg  [{flag}]  (норма 0.5–1.5)")


# =====================================================================
# 2. Step response — ДВУХЭТАПНАЯ интеграция
# =====================================================================

def simulate_step_two_stage(br: Baroreflex,
                            t_step: float = 10.0,
                            t_end: float = 30.0,
                            P_before: float = None,
                            P_after: float = None,
                            method: str = "RK45",
                            rtol: float = 1e-8,
                            atol: float = 1e-10,
                            n_per_stage: int = 300):
    """
    Интегрирует dHR/dt в два этапа:
        stage 1: [0, t_step]  при P = P_before
        stage 2: [t_step, t_end] при P = P_after

    Это устраняет разрыв в правой части от попадания внутрь одного шага
    RK45. Возвращает объединённый (t, y) numpy-массив.
    """
    if P_before is None:
        P_before = P_SET
    if P_after is None:
        P_after = 60.0

    y0 = br.get_initial_state()

    def rhs_before(t, y):
        return br.get_derivatives(t, y, {"P_sa": P_before})

    def rhs_after(t, y):
        return br.get_derivatives(t, y, {"P_sa": P_after})

    # --- Stage 1 ---
    t_eval_1 = np.linspace(0.0, t_step, n_per_stage)
    sol1 = solve_ivp(rhs_before, (0.0, t_step), y0,
                     t_eval=t_eval_1, method=method, rtol=rtol, atol=atol)

    # --- Stage 2, начальное состояние — точный конец stage 1 ---
    y1 = sol1.y[:, -1]
    t_eval_2 = np.linspace(t_step, t_end, n_per_stage)
    sol2 = solve_ivp(rhs_after, (t_step, t_end), y1,
                     t_eval=t_eval_2, method=method, rtol=rtol, atol=atol)

    t_all = np.concatenate([sol1.t, sol2.t])
    y_all = np.concatenate([sol1.y, sol2.y], axis=1)

    nfev_total = sol1.nfev + sol2.nfev
    return t_all, y_all, y1, nfev_total


def test_step_response():
    print("\n" + "=" * 70)
    print("TEST 2: Ступенька 80 → 60 мм рт.ст. в t = 10 с")
    print("        ДВУХЭТАПНАЯ интеграция, rtol=1e-8, atol=1e-10")
    print("=" * 70)

    br = make_baro()

    t, y, y_at_step, nfev = simulate_step_two_stage(
        br, t_step=10.0, t_end=30.0,
        P_before=P_SET, P_after=60.0,
        method="RK45", rtol=1e-8, atol=1e-10, n_per_stage=300,
    )
    HR = y[0, :]

    # Показываем ключевые точки
    for t_check in [0.0, 9.9, 10.0, 10.5, 12.0, 16.0, 30.0]:
        idx = int(np.argmin(np.abs(t - t_check)))
        HRt = float(HR[idx])
        P = P_SET if t[idx] < 10.0 else 60.0
        br.get_derivatives(t[idx], [HRt], {"P_sa": P})
        out = br.get_outputs([HRt])
        print(f"t={t[idx]:5.1f}s  P_sa={P:5.1f}  "
              f"HR={HRt:6.3f}  HR_target={out['HR_target']:6.2f}  "
              f"baro_act={out['baro_activation']:.3f}")

    # Аналитика: от ТОЧНОГО HR(10) из stage 1
    HR_at_step = float(y_at_step[0])
    HR_target_new = HR_BASE * (1.0 - GAIN * (60.0 - P_SET))
    t_rel = 12.0 - 10.0
    HR_analytic = HR_target_new + (HR_at_step - HR_target_new) * np.exp(-t_rel / TAU)

    idx_12 = int(np.argmin(np.abs(t - 12.0)))
    HR_12 = float(HR[idx_12])
    err = abs(HR_12 - HR_analytic)
    tol = 0.05  # типичная точность RK45 с rtol=1e-8 на 20 с
    flag = "OK" if err < tol else "FAIL"

    print(f"\nHR(t=10) — конец stage 1          = {HR_at_step:.6f}")
    print(f"HR_target после step (P_sa=60)    = {HR_target_new:.6f}")
    print(f"Аналитический HR(12) от HR(10)    = {HR_analytic:.6f}")
    print(f"Численный  HR(12)                 = {HR_12:.6f}")
    print(f"|num − analytic|                  = {err:.2e}  "
          f"(порог {tol})  [{flag}]")
    print(f"nfev (суммарно два этапа)         = {nfev}")

    # --- Контроль: RK45 vs LSODA на тех же двух этапах ---
    print("\n--- Контроль: RK45 vs LSODA ---")
    t_L, y_L, _, nfev_L = simulate_step_two_stage(
        br, t_step=10.0, t_end=30.0,
        P_before=P_SET, P_after=60.0,
        method="LSODA", rtol=1e-8, atol=1e-10, n_per_stage=300,
    )
    idx_12_L = int(np.argmin(np.abs(t_L - 12.0)))
    HR_12_L = float(y_L[0, idx_12_L])
    diff = abs(HR_12 - HR_12_L)
    print(f"HR(12) RK45  = {HR_12:.6f}  nfev = {nfev}")
    print(f"HR(12) LSODA = {HR_12_L:.6f}  nfev = {nfev_L}")
    print(f"|RK45 − LSODA| = {diff:.2e}  "
          f"[{'OK' if diff < 1e-3 else 'FAIL'}]")


# =====================================================================
# 3. Variability — синусоида
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
    t_eval = np.linspace(0.0, 60.0, 1201)
    sol = solve_ivp(rhs, (0.0, 60.0), y0, t_eval=t_eval,
                    method="LSODA", rtol=1e-8, atol=1e-10)

    HR_min, HR_max = float(sol.y[0].min()), float(sol.y[0].max())
    amplitude = (HR_max - HR_min) / 2.0
    raw_amp = GAIN * HR_BASE * 20.0
    # Фильтр tau=2с на периоде T=20с: 1/sqrt(1+(ωτ)^2), ω = 2π/T
    omega = 2 * np.pi / 20.0
    filter_amp = 1.0 / np.sqrt(1.0 + (omega * TAU) ** 2)
    predicted_amp = raw_amp * filter_amp
    err = abs(amplitude - predicted_amp) / predicted_amp

    print(f"HR колеблется {HR_min:.2f}–{HR_max:.2f} уд/мин, "
          f"амплитуда ± {amplitude:.2f}")
    print(f"Сырая амплитуда HR_target   = ± {raw_amp:.2f}")
    print(f"С учётом фильтра tau={TAU}с на T=20с (множитель {filter_amp:.3f}): "
          f"± {predicted_amp:.2f}")
    print(f"Относительная разница       = {err * 100:.1f}%  "
          f"[{'OK' if err < 0.10 else 'WARN'}]")


# =====================================================================
# 4. Sanity-check
# =====================================================================

def test_physio_sanity():
    print("\n" + "=" * 70)
    print("TEST 4: Санити-чек формул Baroreflex с текущими параметрами")
    print("=" * 70)
    br = make_baro()

    # 4.1
    br.get_derivatives(0, [HR_BASE], {"P_sa": P_SET})
    out = br.get_outputs([HR_BASE])
    assert abs(out["HR_target"] - HR_BASE) < 1e-9
    assert abs(out["baro_activation"] - 1.0) < 1e-9
    print("✓ P_sa = P_set → HR_target = HR_base, baro_act = 1.0")

    # 4.2
    br.get_derivatives(0, [HR_BASE], {"P_sa": 0.0})
    out = br.get_outputs([HR_BASE])
    assert 40.0 <= out["HR_target"] <= 180.0
    print(f"✓ Клиппинг: P_sa=0 → HR_target = {out['HR_target']:.2f} (clip 40–180)")

    # 4.3
    br.get_derivatives(0, [HR_BASE], {"P_sa": 120.0})
    out = br.get_outputs([HR_BASE])
    assert abs(out["baro_activation"] - 1.0) < 1e-9
    print(f"✓ При P_sa=120 baro_act = {out['baro_activation']:.3f} (не растёт)")

    # 4.4
    br.get_derivatives(0, [HR_BASE], {"P_sa": 40.0})
    out = br.get_outputs([HR_BASE])
    expected = 1.0 + K_INOTROPY * max(1.0 - 40.0 / P_SET, 0.0)
    assert abs(out["baro_activation"] - expected) < 1e-9
    print(f"✓ При P_sa=40 baro_act = {out['baro_activation']:.3f} "
          f"(= 1 + k·(1 − P/P_set))")

    # 4.5
    sens = GAIN * HR_BASE
    flag = "OK" if 0.5 <= sens <= 1.5 else "WEAK"
    print(f"{'✓' if flag == 'OK' else '⚠'} Чувствительность = "
          f"{sens:.3f} bpm/mmHg  [{flag}]")

    # 4.6
    HRs = []
    for P in (40.0, 140.0):
        br.get_derivatives(0, [HR_BASE], {"P_sa": P})
        HRs.append(br.get_outputs([HR_BASE])["HR_target"])
    hr_range = max(HRs) - min(HRs)
    flag = "OK" if hr_range >= 20.0 else "FAIL"
    print(f"{'✓' if flag == 'OK' else '✗'} HR range over [40,140] = "
          f"{hr_range:.1f}  [{flag}, норма ≥ 20]")

    # 4.7
    br.get_derivatives(0, [HR_BASE], {"P_sa": 0.0})
    ba_max = br.get_outputs([HR_BASE])["baro_activation"]
    flag = "OK" if ba_max >= 1.5 else "FAIL"
    print(f"{'✓' if flag == 'OK' else '✗'} baro_activation max = "
          f"{ba_max:.3f}  [{flag}, норма ≥ 1.5]")


# =====================================================================
# 5. Сводка
# =====================================================================

def summary():
    print("\n" + "=" * 70)
    print("СВОДКА: физиологичен ли Baroreflex с текущими параметрами?")
    print("=" * 70)

    checks = []

    sens = GAIN * HR_BASE
    checks.append(("Gain-чувствительность (0.5–1.5 bpm/mmHg)",
                   sens, 0.5, 1.5, 0.5 <= sens <= 1.5))

    br = make_baro()
    HRs = []
    for P in (40.0, 140.0):
        br.get_derivatives(0, [HR_BASE], {"P_sa": P})
        HRs.append(br.get_outputs([HR_BASE])["HR_target"])
    hr_range = max(HRs) - min(HRs)
    checks.append(("HR range over P_sa ∈ [40, 140] ≥ 20 bpm",
                   hr_range, 20.0, 200.0, hr_range >= 20.0))

    br.get_derivatives(0, [HR_BASE], {"P_sa": 20.0})
    ba_max = br.get_outputs([HR_BASE])["baro_activation"]
    checks.append(("baro_activation max ≥ 1.5",
                   ba_max, 1.5, 5.0, ba_max >= 1.5))

    # tau-соответствие на двух этапах
    br = make_baro()
    t, y, y_step, _ = simulate_step_two_stage(
        br, t_step=10.0, t_end=30.0,
        P_before=P_SET, P_after=60.0,
        method="RK45", rtol=1e-10, atol=1e-12, n_per_stage=400,
    )
    idx = int(np.argmin(np.abs(t - (10.0 + TAU))))
    HR_num = float(y[0, idx])
    HR_target_new = HR_BASE * (1.0 - GAIN * (60.0 - P_SET))
    HR_ana = HR_target_new + (float(y_step[0]) - HR_target_new) * np.exp(-1.0)
    err = abs(HR_num - HR_ana)
    checks.append((f"tau = {TAU:.2f} с (err после 1·tau, 2-stage)",
                   err, 0.0, 0.05, err < 0.05))

    print()
    for name, val, lo, hi, ok in checks:
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}] {name:<48} = {val:.4f}  (норма [{lo}, {hi}])")
    print()

    all_ok = all(c[4] for c in checks)
    if all_ok:
        print("ВЫВОД: Baroreflex физиологичен и готов к интеграции с Heart.")
    else:
        print("ВЫВОД: параметры требуют правки в physiology.yaml.")
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