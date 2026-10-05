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
    python debug_whole_body.py --variant eisenmenger_comp
    python debug_whole_body.py --variant eisenmenger_decomp

Логирование:
    По умолчанию весь вывод дублируется в
        <ROOT>/results/debug_whole_body_<variant>_<YYYYmmdd_HHMMSS>.log
    Отключить:      --no-log
    Свой путь:      --log path/to/file.log
    Без timestamp:  --log-flat   (имя без даты/времени, перезаписывается)

Доступные алиасы (из aliases в config/patient_*.yaml):
    healthy              — R_vsd = inf (нет ДМЖП)
    vsd_r5               — R_vsd = 5.0 (малый ДМЖП)
    vsd_r1               — R_vsd = 1.0 (большой ДМЖП)
    eisenmenger_comp     — R_remodel_max=5.5, R_vsd=0.4
    eisenmenger_decomp   — R_remodel_max=10.0, R_vsd=0.3

Для каждого сценария печатает:
    • Сходимость y0 после калибровки
    • Динамику P_sa, V_lv, Qp, Qa, Q_vsd по окнам времени
    • Знак и величину Q_vsd (проверка направления шунта)
    • P_lv vs P_rv на систолическом пике (причина направления шунта)
    • Разбивку объёмов по компартментам (volume breakdown)
    • Физиологичность итоговой точки
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
from scipy.integrate import solve_ivp

# --- ROOT: родительская директория тестов, откуда импортируются модули ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whole_body import WholeBodyModel

# =====================================================================
# Сценарии
# =====================================================================

from physio_config import load_all_patients, load_physiology
from sim_builder import build_model_from_params, extract_simulation_config

_PHYS = load_physiology()
_DIAG = _PHYS.get('diagnostics', {})
_G1   = _DIAG.get('goal1', {})
_PHYS_R = _DIAG.get('physiological', {})

_G1_R_REM_MIN  = float(_G1.get('R_remodel_min', 4.0))
_G1_P_PA_MIN   = float(_G1.get('P_pa_min', 40.0))
_G1_BARO_MIN   = float(_G1.get('baro_activation_rv_min', 1.5))

_P_SA_LO, _P_SA_HI = _PHYS_R.get('P_sa_range', [40.0, 180.0])
_P_PA_LO, _P_PA_HI = _PHYS_R.get('P_pa_range', [5.0, 80.0])
_QA_MIN            = float(_PHYS_R.get('Q_aortic_min', 25.0))
_QPQ_S_LO, _QPQ_S_HI = _PHYS_R.get('Qp_Qs_range', [0.5, 5.0])
_EDV_LV_MIN        = float(_PHYS_R.get('EDV_LV_min', 50.0))
_BAL_TOL           = float(_PHYS_R.get('mass_balance_tol', 2.0))
_ALL_PATIENTS = load_all_patients(base_physiology=_PHYS)

SCENARIOS = {}
for label, p in _ALL_PATIENTS.items():
    for alias in p.get('aliases', []):
        SCENARIOS[alias] = p

if not SCENARIOS:
    raise RuntimeError(
        "debug_whole_body: ни у одного patient_*.yaml нет поля `aliases`. "
        "Добавьте aliases в config/patient_*.yaml, например:\n"
        "  aliases: [healthy]\n"
        "  aliases: [vsd_r5]\n"
        "  aliases: [vsd_r1]\n"
        "  aliases: [eisenmenger_comp]\n"
        "  aliases: [eisenmenger_decomp]"
    )

# --- Проверка, что все ожидаемые варианты доступны ---
_EXPECTED_VARIANTS = ('healthy', 'vsd_r5', 'vsd_r1',
                      'eisenmenger_comp', 'eisenmenger_decomp')
_missing_variants = [v for v in _EXPECTED_VARIANTS if v not in SCENARIOS]
if _missing_variants:
    warnings.warn(
        f"debug_whole_body: отсутствуют ожидаемые алиасы "
        f"{_missing_variants}. Доступные: {sorted(SCENARIOS.keys())}",
        RuntimeWarning,
    )      

# =====================================================================
# Tee-логгер: дублирование вывода в консоль и в файл
# =====================================================================

class Tee:
    """
    Перенаправляет вывод одновременно в несколько потоков.

    Свойства:
      • encoding — берётся у первого потока.
      • fileno() — проксирует fileno первого потока.
      • write()  — буферизация по строкам: flush только при '\\n'.
      • flush()  — явный сброс всех потоков.
    """
    def __init__(self, *streams):
        if not streams:
            raise ValueError("Tee: нужен хотя бы один поток")
        self.streams = streams
        self.encoding = getattr(streams[0], 'encoding', 'utf-8')

    def write(self, data):
        for s in self.streams:
            s.write(data)
        if '\n' in data:
            for s in self.streams:
                s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def fileno(self):
        return self.streams[0].fileno()


def _resolve_log_path(variant: str,
                      explicit: str | None,
                      flat: bool,
                      no_log: bool) -> Path | None:
    """
    Определяет путь лог-файла.

    Приоритеты:
      no_log=True          → None (логирование отключено)
      explicit != None     → этот путь
      flat=True            → results/debug_whole_body_<variant>.log (перезапись)
      иначе                → results/debug_whole_body_<variant>_<timestamp>.log
    """
    if no_log:
        return None
    if explicit:
        p = Path(explicit)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if flat:
        return out_dir / f"debug_whole_body_{variant}.log"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"debug_whole_body_{variant}_{ts}.log"


# =====================================================================
# Вспомогательные функции
# =====================================================================

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

# =====================================================================
# Счётчик RHS-вызовов по времени
# =====================================================================

class _RHSCounter:
    """
    Обёртка model.derivatives: считает вызовы RHS по временным бинам.

    solve_ivp с t_eval возвращает только выходную сетку — внутренние
    шаги остаются скрытыми. Счётчик показывает, где LSODA тратит
    шаги: на переходных процессах (transient spike) или равномерно
    (жёсткая система везде).

    Атрибуты:
      total    — общее число вызовов RHS
      t_min/t_max — границы по времени
      _bins    — dict: bin_idx -> count (bin = int(t / bin_size))
    """
    def __init__(self, model: WholeBodyModel, bin_size: float = 1.0):
        self.model = model
        self.bin_size = float(bin_size)
        self.total = 0
        self._bins: dict[int, int] = {}
        self.t_min: float | None = None
        self.t_max: float | None = None

    def __call__(self, t, y):
        self.total += 1
        key = int(t / self.bin_size)
        self._bins[key] = self._bins.get(key, 0) + 1
        if self.t_min is None or t < self.t_min:
            self.t_min = t
        if self.t_max is None or t > self.t_max:
            self.t_max = t
        return self.model.derivatives(t, y)


def simulate_with_counter(model, t_span, t_eval, y0, bin_size=1.0, **kwargs):
    """solve_ivp с обёрткой RHS-счётчика. Возвращает (sol, counter)."""
    counter = _RHSCounter(model, bin_size=bin_size)
    sol = solve_ivp(counter, t_span, y0, t_eval=t_eval, **kwargs)
    return sol, counter


def _print_solver_cost(sol, counter, t_end_sim: float, label: str,
                       T_cycle_hint: float | None = None,
                       wall_time: float | None = None,
                       n_blocks: int = 24,
                       sparkline_width: int = 80) -> None:
    """
    Диагностика стоимости LSODA: где и как тратятся шаги.

    Метрики:
      nfev/cycle    — сколько RHS-вызовов на один кардиоцикл.
                      Здоровый ~100–300. > 1000 → жёсткая динамика.
      njev/nfev     — частота обновления якобиана. > 0.05 → система
                      часто меняет режим жёсткости (клапаны!).
      nlu/nfev      — частота LU-разложений. Растёт при жёсткости.
      wall/nfev     — стоимость одного RHS-вызова. > 20 us → Python-оверхед
                      доминирует, оптимизируйте RHS, а не метод.
    """
    nfev = int(getattr(sol, 'nfev', 0))
    njev = int(getattr(sol, 'njev', 0))
    nlu  = int(getattr(sol, 'nlu', 0))

    if not T_cycle_hint or T_cycle_hint <= 0:
        T_cycle_hint = 0.86
    n_cycles = t_end_sim / T_cycle_hint

    print(f"\n--- {label}: SOLVER COST ---")
    print(f"  nfev                     = {nfev:>13,}")
    print(f"  njev                     = {njev:>13,}")
    print(f"  nlu                      = {nlu:>13,}")
    print(f"  t_end                    = {t_end_sim:>13.1f} s")
    print(f"  T_cycle (hint)           = {T_cycle_hint:>13.4f} s")
    print(f"  n_cycles                 = {n_cycles:>13.1f}")
    print(f"  nfev / s                 = {nfev / max(t_end_sim, 1e-9):>13.1f}")
    print(f"  nfev / cycle             = {nfev / max(n_cycles, 1e-9):>13.1f}")
    print(f"  njev / nfev              = {njev / max(nfev, 1):>13.5f}")
    print(f"  nlu  / nfev              = {nlu  / max(nfev, 1):>13.5f}")
    if wall_time is not None:
        print(f"  wall time                = {wall_time:>13.2f} s")
        print(f"  wall / nfev              = "
              f"{wall_time * 1e6 / max(nfev, 1):>13.2f} us")
    if counter is not None and counter.total > 0:
        print(f"  RHS calls (counter)      = {counter.total:>13,}")
        print(f"  counter / nfev           = "
              f"{counter.total / max(nfev, 1):>13.4f}  "
              f"(≈1.0 — счётчик видит всё)")

    # --- Гистограмма RHS-вызовов по времени ---
    if counter is None or not counter._bins or counter.t_max is None:
        return

    t_end = counter.t_max or t_end_sim
    block_w = max(t_end / n_blocks, 1e-9)
    counts = [0] * n_blocks
    for b, c in counter._bins.items():
        t_bin = b * counter.bin_size
        idx = min(int(t_bin / block_w), n_blocks - 1)
        counts[idx] += c
    total = sum(counts) or 1
    cmax = max(counts) or 1

    print(f"\n  RHS calls per time block "
          f"({n_blocks} blocks of {block_w:.1f}s):")
    for i, c in enumerate(counts):
        t_lo = i * block_w
        t_hi = (i + 1) * block_w
        pct = 100.0 * c / total
        bar = "#" * int(50 * c / cmax)
        print(f"    [{t_lo:7.1f}–{t_hi:7.1f}]  {c:>11,}  "
              f"{pct:5.1f}%  {bar}")

    # --- Sparkline по всей длительности ---
    SPARK = " .:-=+*#%@"
    width = min(int(sparkline_width), 120)
    bucket = max(t_end / width, 1e-9)
    spark_bins = [0] * width
    for b, c in counter._bins.items():
        t_bin = b * counter.bin_size
        idx = min(int(t_bin / bucket), width - 1)
        spark_bins[idx] += c
    m = max(spark_bins) or 1
    spark = "".join(SPARK[min(int(9 * c / m), 9)] for c in spark_bins)
    print(f"\n  RHS/s sparkline (0..{t_end:.0f}s, "
          f"max={m}, min={min(spark_bins)}, mean={sum(spark_bins)/len(spark_bins):.0f}):")
    print(f"    {spark}")
    print(f"    {'^' * max(1, width // 4)}  "
          f"(левая четверть = переходный процесс; "
          f"правая = стационарный режим)")


def _print_convergence(data: dict, label: str) -> None:
    """Печатает средние по временным окнам — оценка сходимости."""
    print(f"\n--- {label}: CONVERGENCE WINDOWS ---")
    print(f"{'window':>14}  {'P_sa':>7}  {'P_pa':>7}  {'V_lv':>7}  {'V_rv':>7}  "
        f"{'HR':>5}  {'Qa':>7}  {'Qp':>7}  {'Q_vsd':>8}  {'R_rem':>7}")

    t = data["t"]
    t_end = float(t[-1])
    n_windows = 4
    step = t_end / n_windows
    for i in range(n_windows):
        t_lo = i * step
        t_hi = (i + 1) * step
        m = (t >= t_lo) & (t <= t_hi)
        print(f"[{t_lo:6.0f}-{t_hi:6.0f}]  "
            f"{_window_mean(data, 'P_sa', m):7.2f}  "
            f"{_window_mean(data, 'P_pa', m):7.2f}  "
            f"{_window_mean(data, 'V_lv', m):7.1f}  "
            f"{_window_mean(data, 'V_rv', m):7.1f}  "
            f"{_window_mean(data, 'HR', m):5.1f}  "
            f"{_window_mean(data, 'Q_aortic', m):7.2f}  "
            f"{_window_mean(data, 'Q_pulmonary', m):7.2f}  "
            f"{_window_mean(data, 'Q_vsd', m):+8.2f}  "
            f"{_window_mean(data, 'R_remodel', m):7.3f}")


def _print_lungs_steady(data: dict, label: str) -> None:
    t = data["t"]
    HR_mean = _window_mean(data, "HR", t > t[-1] * 0.5)
    T = 60.0 / max(HR_mean, 1e-6)
    win = t > (t[-1] - 10.0 * T)

    print(f"\n--- {label}: LUNGS / PVR ---")
    for key, name in (
        ("R_remodel",     "R_remodel (mnozitel)"),
        ("f_recruit",     "f_recruit (острый)"),
        ("f_recruit_eff", "f_recruit_eff (с подавлением)"),
        ("R_target",      "R_target (цель)"),
    ):
        v = _window_mean(data, key, win)
        print(f"  {name:32s} = {v:7.3f}")

    mode_v = _window_mean(data, "mode_remodeled", win)
    print(f"  {'mode':32s} = "
          f"{'remodeled' if mode_v > 0.5 else 'healthy'}")


def _print_goal1_check(data: dict, label: str, model=None, sol=None) -> None:
    """
    Сводный тест Цели 1: переворот градиента P_rv > P_lv в систолу
    → устойчивый R→L.
    """
    print(f"\n--- {label}: ЦЕЛЬ 1 — ПЕРЕВОРОТ ГРАДИЕНТА ---")

    t = data["t"]
    HR_mean = _window_mean(data, "HR", t > t[-1] * 0.5)
    T = 60.0 / max(HR_mean, 1e-6) if np.isfinite(HR_mean) else 0.86
    win = t > (t[-1] - 10.0 * T)

    # --- 1. R_remodel в steady ---
    R_rem = _window_mean(data, "R_remodel", win)

    # --- 2. P_pa в steady ---
    P_pa = _window_mean(data, "P_pa", win)

    # --- 3. Систолический пик P_lv vs P_rv за последний цикл ---
    cycle = t > (t[-1] - T)
    P_lv_c = data["P_lv"][cycle]
    P_rv_c = data["P_rv"][cycle]
    i_peak_lv = int(np.argmax(P_lv_c))
    i_peak_rv = int(np.argmax(P_rv_c))
    P_lv_max = float(P_lv_c[i_peak_lv])
    P_rv_max = float(P_rv_c[i_peak_rv])
    dP = P_lv_max - P_rv_max

    # --- 4. Q_vsd среднее за цикл ---
    Q_vsd = _window_mean(data, "Q_vsd", win)

    # --- 6. Печать критериев ---
    def check(cond, name, value, target):
        mark = "✓" if cond else "✗"
        print(f"  [{mark}] {name:28s} = {value:9.3f}   (критерий: {target})")

    check(R_rem > _G1_R_REM_MIN,
          "R_remodel (steady)",
          R_rem, f"> {_G1_R_REM_MIN}")
    check(P_pa > _G1_P_PA_MIN,
          "P_pa (steady, мм рт.ст.)",
          P_pa, f"> {_G1_P_PA_MIN}")
    check(dP < 0.0,
          "P_lv_max − P_rv_max (сист.)",
          dP, "< 0")
    check(Q_vsd < 0.0,
          "Q_vsd (среднее, мл/с)",
          Q_vsd, "< 0")

    # --- 5. Пульмональная бароактивация ПЖ ---
    # compute_outputs пробрасывает baro_activation_rv из
    # baroreflex_out['baro_activation_rv']; при P_pa > P_pa_set
    # (лёгочная гипертензия) сигнал > 1.0, при норме ≈ 1.0.
    baro_rv = _window_mean(data, "baro_activation_rv", win)
    if np.isfinite(baro_rv):
        check(baro_rv > _G1_BARO_MIN,
              "baro_activation_rv",
              baro_rv, f"> {_G1_BARO_MIN}")
    else:
        print(f"  [—] {'baro_activation_rv':28s} = "
              f"{'N/A':>9s}   (ключ отсутствует в outputs "
              f"или содержит NaN)")  

    # --- Проверка периодичности: y_end vs y(t_end − T) ---
    if sol is not None and len(sol.t) >= 2:
        HR_end = float(model.baroreflex.get_outputs(
            sol.y[model.idx['baroreflex'], -1])['HR'])
        T_end = 60.0 / max(HR_end, 1e-6)
        # Число точек, покрывающих один цикл (равномерная сетка t_eval).
        n_cycle = max(int(round(T_end / (sol.t[-1] - sol.t[-2]))), 1)
        if n_cycle < sol.t.size:
            dy = np.linalg.norm(sol.y[:, -1] - sol.y[:, -1 - n_cycle])
            print(f"  [{'✓' if dy < 1.0 else '✗'}] "
                  f"Периодичность ‖y(t_end) − y(t_end−T)‖ = {dy:.3f}")

    # --- Итоговый вердикт ---
    # baro_activation_rv проверяем только при активном ремоделировании,
    # т.к. без лёгочной гипертензии P_pa ≈ P_pa_set и сигнал ≡ 1.0
    if R_rem <= 2.0:
        all_ok = False
    else:
        all_ok = (R_rem  > _G1_R_REM_MIN
                  and P_pa   > _G1_P_PA_MIN
                  and dP     < 0.0
                  and Q_vsd  < 0.0
                  and baro_rv > _G1_BARO_MIN)

    print()
    if all_ok:
        print(f"  ►►► ЦЕЛЬ 1 ДОСТИГНУТА: устойчивый R→L шунт ◄◄◄")
    elif R_rem > _G1_R_REM_MIN and P_pa > _G1_P_PA_MIN and dP < 30.0:
        print(f"  ► Близко: PVR достаточный, но градиент ещё не перевёрнут")
    elif R_rem < _G1_R_REM_MIN:
        print(f"  ► Ремоделирование не дошло — ...")
    else:
        print(f"  ► Цель 1 пока не достигнута")


def _volume_breakdown(model: WholeBodyModel, y: np.ndarray) -> dict:
    """
    Разбивает общий объём крови на компартменты.
    Все объёмы в мл. Считается из сырого состояния y.
    """
    sl = model.idx

    # --- Heart: [V_la, V_lv, V_ra, V_rv] ---
    V_heart_state = y[sl['heart']]
    V_la = max(float(V_heart_state[0]), 0.0)
    V_lv = max(float(V_heart_state[1]), 0.0)
    V_ra = max(float(V_heart_state[2]), 0.0)
    V_rv = max(float(V_heart_state[3]), 0.0)

    # --- Lungs: [P_prox, P_dist, R_remodel] ---
    V_lungs_state = y[sl['lungs']]
    P_prox = max(float(V_lungs_state[0]), 0.0)
    P_dist = max(float(V_lungs_state[1]), 0.0)
    V_lungs = model.lungs.C1 * P_prox + model.lungs.C2 * P_dist

    # --- Systemic veins (Windkessel mode='V') ---
    V_sv = max(float(y[sl['sys_ven']][0]), 0.0)

    # --- Jugular vein (state[0] — объём) ---
    V_jv = max(float(y[sl['jugular_vein']][0]), 0.0)

    # --- Systemic arteries (Windkessel mode='P') ---
    P_sa = max(float(y[sl['sys_art']][0]), 0.0)
    V_sys_art = model.sys_art.C * P_sa

    # --- Pulmonary veins ---
    P_pv = max(float(y[sl['pul_ven']][0]), 0.0)
    V_pul_ven = model.pul_ven.C * P_pv

    # --- Liver: [P_hv, C_bil, C_amm, C_alb, reserve, P_portal] ---
    V_liver_state = y[sl['liver']]
    P_hv = max(float(V_liver_state[0]), 0.0)
    P_portal = max(float(V_liver_state[5]), 0.0)
    V_liver = model.liver.C * P_hv + model.liver.C_portal * P_portal

    # --- GITract: [P_art, P_cap] ---
    V_gitract_state = y[sl['gitract']]
    P_art = max(float(V_gitract_state[0]), 0.0)
    P_cap = max(float(V_gitract_state[1]), 0.0)
    V_gitract = model.gitract.C_art * P_art + model.gitract.C_cap * P_cap

    # --- Brain: [P_br, ...] ---
    P_br = max(float(y[sl['brain']][0]), 0.0)
    V_brain = model.brain.C * P_br

    return {
        'V_sv (systemic veins)':             V_sv,
        'V_jv (jugular vein)':               V_jv,
        'V_heart (LA+LV+RA+RV)':             V_la + V_lv + V_ra + V_rv,
        'V_lungs (C1*P_prox + C2*P_dist)':   V_lungs,
        'V_sys_art (C*P_sa)':                V_sys_art,
        'V_pul_ven (C*P_pv)':                V_pul_ven,
        'V_liver (C*P_hv + Cp*P_portal)':    V_liver,
        'V_gitract (Ca*P_art + Cc*P_cap)':   V_gitract,
        'V_brain (C*P_br)':                  V_brain,
    }


def _print_volume_breakdown(model: WholeBodyModel, sol, label: str,
                            n_cycles: float = 10.0) -> None:
    """
    Разбивка физического объёма по 9 компартментам; сумма = V_blood_total
    """
    # --- Определяем окно последних ~n_cycles циклов ---
    HR_end = float(model.baroreflex.get_outputs(
        sol.y[model.idx['baroreflex'], -1])['HR'])
    T = 60.0 / max(HR_end, 1e-6)
    t_end = sol.t[-1]
    mask = sol.t > (t_end - n_cycles * T)
    idxs = np.where(mask)[0]
    if idxs.size < 2:
        idxs = np.arange(max(0, sol.t.size - 100), sol.t.size)

    # --- Усредняем разбивку ---
    sums = None
    keys = None
    for i in idxs:
        vols = _volume_breakdown(model, sol.y[:, i])
        if keys is None:
            keys = list(vols.keys())
            sums = {k: 0.0 for k in keys}
        for k, v in vols.items():
            sums[k] += v
    means = {k: sums[k] / len(idxs) for k in keys}

    # --- Печатаем ---
    print(f"\n--- {label}: VOLUME COMPARTMENTS "
          f"(mean over last {n_cycles:.0f} cycles = {len(idxs)} pts) ---")
    for k in keys:
        print(f"  {k:42s} = {means[k]:9.2f} мл")

    V_total = sum(means.values())
    print(f"  {'-' * 58}")
    print(f"  {'V_blood_total (SUM всех компартментов)':42s} = "
        f"{V_total:9.2f} мл")

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
    if abs(balance_lv) > _BAL_TOL or abs(balance_rv) > _BAL_TOL:
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
    issues = []
    if not (_P_SA_LO < P_sa < _P_SA_HI):
        issues.append(f"P_sa={P_sa:.1f} вне [{_P_SA_LO}, {_P_SA_HI}]")
    if not (_P_PA_LO < P_pa < _P_PA_HI):
        issues.append(f"P_pa={P_pa:.1f} вне [{_P_PA_LO}, {_P_PA_HI}]")
    if Qa < _QA_MIN:
        issues.append(f"Q_aortic={Qa:.1f} < {_QA_MIN} (системный коллапс)")
    if Qp_Qs > _QPQ_S_HI:
        issues.append(f"Qp/Qs={Qp_Qs:.2f} > {_QPQ_S_HI} (экстремальный шунт)")
    if Qp_Qs < _QPQ_S_LO:
        issues.append(f"Qp/Qs={Qp_Qs:.2f} < {_QPQ_S_LO} (нефизиологичная инверсия)")
    if EDV_LV < _EDV_LV_MIN:
        issues.append(f"EDV_LV={EDV_LV:.1f} < {_EDV_LV_MIN} (недонаполнение ЛЖ)")
    if issues:
        print("  ⚠ ФИЗИОЛОГИЧЕСКИЕ ПРЕДУПРЕЖДЕНИЯ:")
        for w in issues:
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

def run_scenario(name: str, params: dict) -> None:
    """
    Параметры
    ---------
    name   : короткое имя variant ('healthy', 'vsd_r5', ...)
    params : merged-конфиг из load_all_patients(base_physiology=...)
    """
    label = params["label"]
    print("\n" + "=" * 78)
    print(f"  SCENARIO: {name}  —  {label}")
    print("=" * 78)
    print(f"  timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}")

    # --- Построение модели ---
    try:
        model = build_model_from_params(params)
    except Exception as e:
        print(f"  ✗ build_model FAILED: {e}")
        return

    # --- Симуляционные константы из YAML ---
    sc          = extract_simulation_config(params)
    t_calib_eff = sc['t_calib']
    rtol        = sc['rtol']
    atol        = sc['atol']
    max_step    = sc['max_step']
    method      = sc['method']
    t_span      = sc['t_span']
    n_pts       = sc['n_samples_t']

    print(f"\n  Калибровка t_calib={t_calib_eff:.0f} с...")
    try:
        y0 = model.calibrate_initial_state(t_calib=t_calib_eff)
    except Exception as e:
        print(f"  ✗ calibrate_initial_state FAILED: {e}")
        return

    heart_slc = model.idx['heart']
    P_sa_0 = y0[model.idx['sys_art']][0]
    print(f"  y0: heart={y0[heart_slc]}  P_sa={P_sa_0:.2f}")

    print(f"\n  Симуляция {t_span[0]:.0f}..{t_span[1]:.0f} с, {method} "
          f"(с RHS-счётчиком)")
    t_eval = np.linspace(t_span[0], t_span[1], n_pts)
    _t_sim_start = time.perf_counter()
    try:
        sol, rhs_counter = simulate_with_counter(
            model, t_span, t_eval=t_eval, y0=y0,
            bin_size=max((t_span[1] - t_span[0]) / 2000.0, 0.1),
            method=method, rtol=rtol, atol=atol, max_step=max_step,
        )
    except Exception as e:
        print(f"  ✗ simulate FAILED: {e}")
        return
    _wall = time.perf_counter() - _t_sim_start

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
    # --- T_cycle из конечной точки ---
    try:
        HR_end = float(model.baroreflex.get_outputs(
            sol.y[model.idx['baroreflex'], -1])['HR'])
        T_cycle = 60.0 / max(HR_end, 1e-6)
    except Exception:
        T_cycle = None

    _print_solver_cost(
        sol, rhs_counter, t_span[1], label,
        T_cycle_hint=T_cycle, wall_time=_wall,
        n_blocks=24, sparkline_width=80,
    )

    _print_one_cycle(data, label)
    _print_steady(data, label, model=model, sol=sol)
    _print_volume_breakdown(model, sol, label)
    _print_lungs_steady(data, label)
    _print_goal1_check(data, label, model=model, sol=sol)


# =====================================================================
# main
# =====================================================================

def _main_impl(variant: str) -> None:
    """Основная логика main(). Вызывается внутри Tee-контекста."""
    t_start = datetime.now()
    print("=" * 78)
    print(f"  DEBUG WholeBodyModel — variant: {variant}")
    print(f"  Старт: {t_start:%Y-%m-%d %H:%M:%S}")
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
    print(f"  Финиш: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Длительность: {datetime.now() - t_start}")
    print("=" * 78)


def main(variant: str,
         log_path: Path | None = None,
         no_log: bool = False) -> None:
    """
    Обёртка main(): Tee-логгер в указанный файл.

    Если log_path=None и no_log=False — авто-путь
        results/debug_whole_body_<variant>_<timestamp>.log
    """
    if no_log or log_path is None:
        _main_impl(variant)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f"[log] Script    : {Path(sys.argv[0]).resolve().name}")
            print(f"[log] Запись в: {log_path}")
            _main_impl(variant)
            print(f"[log] Полный лог: {log_path}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Isolated diagnostic of WholeBodyModel."
    )
    parser.add_argument(
        "--variant", "-V", type=str, default="vsd_r5",
        help="Алиас сценария из config/patient_*.yaml или 'all'. "
             "Доступные см. в начале лога. (default: vsd_r5)",
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Путь к лог-файлу. По умолчанию — "
             "results/debug_whole_body_<variant>_<timestamp>.log",
    )
    parser.add_argument(
        "--log-flat", action="store_true",
        help="Имя лога без timestamp: "
             "results/debug_whole_body_<variant>.log (перезапись).",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Отключить запись в файл (только консоль).",
    )
    args = parser.parse_args()

    log_path = _resolve_log_path(
        variant=args.variant,
        explicit=args.log,
        flat=args.log_flat,
        no_log=args.no_log,
    )
    main(variant=args.variant, log_path=log_path, no_log=args.no_log)