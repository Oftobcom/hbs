#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_simulation_parallel.py — параллельная версия run_simulation.py

Логирование:
    По умолчанию весь вывод (родитель + рабочие процессы loky) дублируется в
        <script_dir>/results/run_simulation_parallel_<YYYYmmdd_HHMMSS>.log
    Отключить:      --no-log
    Свой путь:      --log path/to/file.log
    Без timestamp:  --log-flat   (results/run_simulation_parallel.log, перезапись)
"""

import matplotlib
matplotlib.use('Agg')  # важно для joblib/loky — не форкать GUI backend

import argparse
import sys
import warnings
import time
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from whole_body import WholeBodyModel
from utils import (subsample, steady_mask,
                   clean_nans, steady_mean, clinical_qp_qs_series,
                   format_mean_std, qp_qs_steady, auto_ylim,
                   STEADY_FRAC_DEFAULT)
from physio_config import load_all_patients, load_physiology
warnings.filterwarnings('ignore')
from sim_builder import build_model_from_params, extract_simulation_config

# =====================================================================
# Tee-логгер: дублирование вывода в консоль и в файл
# =====================================================================

class Tee:
    """
    Перенаправляет вывод одновременно в несколько потоков.

    • write() — буферизация по строкам: flush только при '\\n'.
    • flush()  — явный сброс всех потоков.
    • fileno() — проксирует fileno первого потока.
    Все операции обёрнуты в try/except, чтобы не валить задачи при
    закрытии одного из потоков.
    """
    def __init__(self, *streams):
        if not streams:
            raise ValueError("Tee: нужен хотя бы один поток")
        self.streams = streams
        self.encoding = getattr(streams[0], 'encoding', 'utf-8')

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        if '\n' in data:
            self.flush()

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def fileno(self):
        return self.streams[0].fileno()


# =====================================================================
# Логирование в рабочем процессе loky
# =====================================================================

_WORKER_LOG_FILE = None


def _init_worker_logging(log_path: str) -> None:
    """
    Вызывается joblib один раз на процесс-воркер.
    Устанавливает Tee поверх текущих stdout/stderr, чтобы весь вывод
    рабочего процесса дублировался в общий лог-файл.

    Идемпотентна: повторный вызов в том же процессе — no-op.
    """
    global _WORKER_LOG_FILE
    if not log_path:
        return
    if isinstance(sys.stdout, Tee):
        return
    try:
        _WORKER_LOG_FILE = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = Tee(sys.stdout, _WORKER_LOG_FILE)
        sys.stderr = Tee(sys.stderr, _WORKER_LOG_FILE)
    except Exception as e:
        # Не валим задачи, если лог не открылся.
        print(f"[WARN] worker logging disabled: {e}", file=sys.__stdout__)


def _resolve_log_path(explicit, flat: bool, no_log: bool):
    """
    Определяет путь лог-файла.

    Приоритеты:
      no_log=True      → None (логирование отключено)
      explicit != None → этот путь
      flat=True        → results/run_simulation_parallel.log (перезапись)
      иначе            → results/run_simulation_parallel_<timestamp>.log
    """
    if no_log:
        return None
    if explicit:
        p = Path(explicit)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if flat:
        return out_dir / "run_simulation_parallel.log"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"run_simulation_parallel_{ts}.log"


# =====================================================================
# Глобальные константы и пациенты
# =====================================================================

_PHYSIOLOGY = load_physiology()
_PATIENTS = load_all_patients(base_physiology=_PHYSIOLOGY)
COLORS = {p['label']: p['color'] for p in _PATIENTS.values()}
SCENARIO_ORDER = [p['label'] for p in sorted(_PATIENTS.values(),
                                              key=lambda c: int(c['order']))]

N_PLOT_POINTS = 4000
N_PLOT_POINTS_DETAIL = 1200


# =====================================================================
# Симуляция одного сценария
# =====================================================================

def simulate_scenario(label: str, params: dict,
                      t_span=None, t_calib=None):
    """
    Одна симуляция сценария.

    Параметры
    ---------
    label   : человекочитаемое имя сценария ('Здоровый', 'Малый ДМЖП (R=5.0)', ...)
    params  : merged-словарь из load_all_patients(base_physiology=_PHYSIOLOGY).
              Содержит секции physiology.yaml (heart, lungs, baroreflex,
              blood, peripheral, liver, kidney, brain, gitract,
              gas_exchange, systemic, jugular_vein, simulation) плюс
              top-level overrides из patient_*.yaml (vsd_resistance,
              flow_dependent_lungs, pressure_remodel, HR_base,
              E_max_rv, E_max_lv, EDV_rv, k_hr, k_inotropy, ...).
    t_span  : (t0, t1), переопределяет params['simulation']['t_span'].
    t_calib : длительность калибровки, переопределяет
              params['simulation']['t_calib'].

    Возвращает
    ----------
    dict с массивами выходов + 't' + 'Qp_Qs_steady'.
    """
    model = build_model_from_params(params)

    # --- Адаптивный t_calib: быстрый для healthy без ремоделирования ---
    sc = extract_simulation_config(params)
    t_span      = tuple(t_span) if t_span is not None else sc['t_span']
    t_calib_eff = float(t_calib) if t_calib is not None else sc['t_calib']
    n_eval      = sc['n_samples_t']
    max_step    = sc['max_step']
    rtol        = sc['rtol']
    atol        = sc['atol']
    method      = sc['method']
    steady_frac = sc['steady_frac']
    
    print(f"  [PID {os.getpid()}] [{label}] "
          f"Калибровка t_calib={t_calib_eff:.0f}с...", flush=True)
    with warnings.catch_warnings(record=True) as calib_warns:
        warnings.simplefilter("always", category=UserWarning)
        y0 = model.calibrate_initial_state(t_calib=t_calib_eff)
    if os.environ.get("HBS_STRICT_CALIB") == "1" and calib_warns:
        _bad = [w for w in calib_warns if "calibrate:" in str(w.message)]
        if _bad:
            raise RuntimeError(
                f"[{label}] calibration failed: {_bad[0].message}"
            )
    for w in calib_warns:
        if "calibrate:" in str(w.message):
            print(f"  [PID {os.getpid()}] [{label}] "
                  f"⚠ CALIBRATION: {w.message}", flush=True)
            print(f"  [PID {os.getpid()}] [{label}] "
                  f"⚠ y0 NOT relaxed — CHECK ниже отражает analytic y0, "
                  f"не физиологическое равновесие.", flush=True)

    # --- Диагностика начального состояния ---
    cycle = model.cycle_averaged_flows(0.0, y0, n_pts=24)
    out0  = model.compute_outputs(0.0, y0)
    print(f"  [PID {os.getpid()}] [{label}] CHECK "
          f"P_sa={out0['P_sa']:.1f} P_pa={out0['P_pa']:.1f} "
          f"P_sv={out0['P_sv']:.1f} HR={out0['HR']:.1f} "
          f"V_blood={out0['V_blood']:.0f} "
          f"Qp={cycle['Qp_cycle_mean']:.1f} Qs={cycle['Qs_cycle_mean']:.1f} "
          f"Qp/Qs={cycle['Qp_Qs_cycle']:.2f} "
          f"Q_vsd={cycle['Q_vsd_cycle_mean']:+.1f} "
          f"balance={cycle['mass_balance_error']:+.2f} "
          f"SaO2={cycle['SaO2_cycle_mean']*100:.1f}% "
          f"R2L={cycle['shunt_fraction_R2L_mean']*100:.1f}% "
          f"baro_vaso={out0['baro_vasomotor']:.2f} "
          f"baro_ino={out0['baro_inotropy']:.2f} "
          f"suppress={out0['suppress']:.2f}", flush=True)

    # --- Симуляция ---
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    print(f"  [PID {os.getpid()}] [{label}] "
          f"Симуляция {t_span[0]:.0f}..{t_span[1]:.0f}с "
          f"{method} max_step={max_step:.3f} {n_eval} точек...", flush=True)
    try:
        sol = model.simulate(
            t_span, t_eval,
            rtol=rtol, atol=atol, max_step=max_step,
        )
    except Exception as e:
        print(f"  [PID {os.getpid()}] [{label}] ⚠ Integration Method FAILED: {e}",
              flush=True)
        return None
    print(f"  [PID {os.getpid()}] [{label}] готово {sol.t.size} точек, "
          f"{sol.nfev} RHS, "
          f"cache_hits={getattr(model, '_flow_cache_hits', 0)}", flush=True)

    # --- Установившееся окно (границы — из sim_cfg['steady_frac']) ---
    if sol.t.size > 0:
        mask_steady = sol.t > steady_frac * sol.t[-1]
        if np.sum(mask_steady) < 100:
            mask_steady = np.ones_like(sol.t, dtype=bool)
            mask_steady[: int((1.0 - steady_frac) * len(mask_steady))] = False
    else:
        mask_steady = np.array([], dtype=bool)

    idx_steady = np.where(mask_steady)[0]
    if idx_steady.size == 0:
        raise RuntimeError(
            f"[{label}] Нет точек в установившемся окне "
            f"(steady_frac={steady_frac}, "
            f"t_end={sol.t[-1] if sol.t.size else 'N/A'})"
        )

    outputs = [model.compute_outputs(sol.t[i], sol.y[:, i])
               for i in idx_steady]
    keys = list(outputs[0].keys())
    data = {k: np.fromiter((o[k] for o in outputs),
                           dtype=float, count=len(outputs))
            for k in keys}
    data['t'] = sol.t[idx_steady]
    data = clean_nans(data)
    qpq = qp_qs_steady(data)
    data['Qp_Qs_steady'] = np.array([qpq if qpq is not None else np.nan])
    return data


# =====================================================================
# Плоттеры (без изменений по логике)
# =====================================================================

def plot_enhanced_comparison(results_dict):
    """Сводная панель 3x3 по сценариям."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    ys_axis = []
    for name, (data, _, _) in results_dict.items():
        t, ps = subsample(data, 'P_sa', N_PLOT_POINTS)
        _, pp = subsample(data, 'P_pa', N_PLOT_POINTS)
        m = np.isfinite(ps) & np.isfinite(pp)
        if not np.any(m):
            continue
        c = COLORS.get(name, 'gray')
        ax.plot(t[m], ps[m], color=c, lw=1.5, label=f"{name} (P_sa)")
        ax.plot(t[m], pp[m], color=c, lw=1.2, ls='--', alpha=0.7)
        ys_axis.append(ps[m])
        ys_axis.append(pp[m])
    ax.set_ylabel('Давление (мм рт. ст.)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Системное (—) и лёгочное (- -) давление')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    if ys_axis:
        auto_ylim(ax, np.concatenate(ys_axis))

    ax = fig.add_subplot(gs[0, 1])
    for name, (data, _, _) in results_dict.items():
        t, qa = subsample(data, 'Q_aortic')
        _, qp = subsample(data, 'Q_pulmonary')
        m = np.isfinite(qa) & np.isfinite(qp)
        if not np.any(m):
            continue
        c = COLORS.get(name, 'gray')
        ax.plot(t[m], qa[m], color=c, lw=1.5, label=f"{name} (Qs)")
        ax.plot(t[m], qp[m], color=c, lw=1.2, ls='--', alpha=0.7)
    ax.set_ylabel('Кровоток (мл/с)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Системный (—) и лёгочный (- -) кровоток')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 250)

    ax = fig.add_subplot(gs[0, 2])
    for name, (data, _, _) in results_dict.items():
        ratio = clinical_qp_qs_series(data, window=501, polyorder=3)
        t = data['t']
        n = min(len(t), len(ratio))
        m = np.isfinite(ratio[:n])
        if not np.any(m):
            continue
        qp_qs_val = qp_qs_steady(data)
        qp_qs_str = f"{qp_qs_val:.2f}" if qp_qs_val is not None else "N/A"
        ax.plot(t[:n][m], ratio[:n][m], color=COLORS.get(name, 'gray'),
                lw=2, label=f"{name} (Qp/Qs={qp_qs_str})")
        if qp_qs_val is not None:
            ax.axhline(qp_qs_val, color=COLORS.get(name, 'gray'),
                       ls='--', alpha=0.4, lw=1)
    ax.set_ylabel('Qp/Qs')
    ax.set_xlabel('Время (с)')
    ax.set_title('Соотношение лёгочного и системного кровотока')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='black', ls='--', alpha=0.5)
    ax.axhline(y=1.5, color='orange', ls=':', alpha=0.5)
    ax.axhline(y=2.0, color='red', ls=':', alpha=0.5)
    ax.set_ylim(0.5, 5.0)

    ax = fig.add_subplot(gs[1, 0])
    for name, (data, _, _) in results_dict.items():
        t, q_vsd = subsample(data, 'Q_vsd')
        m = np.isfinite(q_vsd)
        if not np.any(m):
            continue
        ax.plot(t[m], q_vsd[m], color=COLORS.get(name, 'gray'),
                lw=1.5, label=name)
    ax.set_ylabel('Шунт (мл/с)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Шунт через ДМЖП')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', ls='--', alpha=0.5)

    ax = fig.add_subplot(gs[1, 1])
    ys_axis = []
    for name, (data, _, _) in results_dict.items():
        t, v_lv = subsample(data, 'V_lv')
        _, v_rv = subsample(data, 'V_rv')
        m = np.isfinite(v_lv) & np.isfinite(v_rv)
        if not np.any(m):
            continue
        c = COLORS.get(name, 'gray')
        ax.plot(t[m], v_lv[m], color=c, lw=1.2, alpha=0.8)
        ax.plot(t[m], v_rv[m], color=c, lw=1.2, ls='--', alpha=0.8)
        ys_axis.append(v_lv[m])
        ys_axis.append(v_rv[m])
    ax.set_ylabel('Объём (мл)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Объёмы желудочков (— ЛЖ, - - ПЖ)')
    ax.grid(True, alpha=0.3)
    if ys_axis:
        auto_ylim(ax, np.concatenate(ys_axis))

    ax = fig.add_subplot(gs[1, 2])
    for name, (data, _, _) in results_dict.items():
        t, v = subsample(data, 'V_blood')
        m = np.isfinite(v)
        if not np.any(m):
            continue
        ax.plot(t[m], v[m], color=COLORS.get(name, 'gray'),
                lw=1.5, label=name)
    ax.set_ylabel('Объём крови (мл)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Волемический статус')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 0])
    for name, (data, _, _) in results_dict.items():
        t, q = subsample(data, 'Q_brain')
        m = np.isfinite(q)
        if not np.any(m):
            continue
        ax.plot(t[m], q[m], color=COLORS.get(name, 'gray'),
                lw=1.5, label=name)
    ax.set_ylabel('Мозговой кровоток (мл/с)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Церебральная гемодинамика')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    for name, (data, _, _) in results_dict.items():
        t, s = subsample(data, 'SaO2')
        m = np.isfinite(s)
        if not np.any(m):
            continue
        ax.plot(t[m], s[m] * 100, color=COLORS.get(name, 'gray'),
                lw=1.5, label=name)
    ax.axhline(y=90, color='orange', ls=':', alpha=0.5, label='SaO2 = 90%')
    ax.set_ylabel('SaO₂ (%)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Артериальная сатурация O₂')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(60, 100)

    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    table_data = [['Показатель'] + [s[:15] for s in SCENARIO_ORDER]]
    summary_metrics = [
        ('P_sa',      'P_sa, мм Hg', '{:.0f}', 'mean'),
        ('P_pa',      'P_pa, мм Hg', '{:.0f}', 'mean'),
        ('__qp_qs__', 'Qp/Qs',       '{:.2f}', 'qp_qs'),
        ('SaO2',      'SaO₂, %',     '{:.1f}', 'mean_scaled'),
        ('V_rv',      'V_rv, мл',    '{:.0f}', 'mean'),
        ('V_blood',   'V_blood, мл', '{:.0f}', 'mean'),
    ]
    for metric, label, fmt, mode in summary_metrics:
        row = [label]
        for name in SCENARIO_ORDER:
            if name not in results_dict:
                row.append('N/A')
                continue
            data, _, _ = results_dict[name]
            if mode == 'qp_qs':
                v = qp_qs_steady(data)
            elif mode == 'mean_scaled':
                v = steady_mean(data, metric, scale=100.0)
            else:
                v = steady_mean(data, metric)
            row.append(fmt.format(v) if v is not None else 'N/A')
        table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.24] + [0.19] * len(SCENARIO_ORDER))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#f0f0f0')

    ax.set_title('Сводка установившихся значений',
                 fontsize=10, fontweight='bold')

    fig.suptitle('Гемодинамика: Здоровый vs ДМЖП (малый / большой / Эйзенменгер)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hemodynamics_comparison_enhanced.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_shunt_effect_analysis(results_dict):
    """Анализ влияния размера шунта на гемодинамику."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Анализ влияния размера ДМЖП', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    for name, (data, _, _) in results_dict.items():
        ratio = clinical_qp_qs_series(data, window=501, polyorder=3)
        t = data['t']
        n = min(len(t), len(ratio))
        m = np.isfinite(ratio[:n])
        if not np.any(m):
            continue
        qp_qs_val = qp_qs_steady(data)
        qp_qs_str = f"{qp_qs_val:.2f}" if qp_qs_val is not None else "N/A"
        ax.plot(t[:n][m], ratio[:n][m], color=COLORS.get(name, 'gray'),
                lw=2, label=f"{name} (Qp/Qs={qp_qs_str})")
        if qp_qs_val is not None:
            ax.axhline(qp_qs_val, color=COLORS.get(name, 'gray'),
                       ls='--', alpha=0.4, lw=1)
    ax.axhline(y=1.0, color='black', ls='--', alpha=0.5)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Qp/Qs')
    ax.set_title('Динамика Qp/Qs (скользящее среднее)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 5.0)

    ax = axes[0, 1]
    x = np.arange(2)
    width = 0.20
    for i, name in enumerate(SCENARIO_ORDER):
        if name not in results_dict:
            continue
        data, _, _ = results_dict[name]
        m = steady_mask(data)
        if 'P_sa' not in data or 'P_pa' not in data:
            continue
        ps = np.mean(data['P_sa'][m & np.isfinite(data['P_sa'])])
        pp = np.mean(data['P_pa'][m & np.isfinite(data['P_pa'])])
        offset = (i - 1.5) * width
        ax.bar(x[0] + offset, ps, width, color=COLORS.get(name, 'gray'),
               alpha=0.75, edgecolor='black', label=name)
        ax.bar(x[1] + offset, pp, width, color=COLORS.get(name, 'gray'),
               alpha=0.40, edgecolor='black', hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels(['Системное', 'Лёгочное'])
    ax.set_ylabel('Давление (мм рт. ст.)')
    ax.set_title('Сравнение давлений')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[0, 2]
    xs, ys = [], []
    for name, (data, _, _) in results_dict.items():
        s = steady_mean(data, 'Q_vsd')
        q = qp_qs_steady(data)
        if s is None or q is None:
            continue
        if abs(s) > 1.0:
            xs.append(s); ys.append(q)
            ax.scatter(s, q, s=120, c=COLORS.get(name, 'gray'),
                       edgecolor='black', linewidth=1.5, label=name)
    if len(xs) > 1:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(xs), max(xs), 50)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5,
                label=f'Qp/Qs ≈ {z[0]:.3f}·Q_vsd + {z[1]:.2f}')
    ax.axvline(x=0, color='black', ls='--', alpha=0.5)
    ax.set_xlabel('Средний шунт (мл/с)')
    ax.set_ylabel('Qp/Qs')
    ax.set_title('Корреляция: шунт → Qp/Qs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for name, (data, _, _) in results_dict.items():
        t, v = subsample(data, 'V_blood', N_PLOT_POINTS_DETAIL)
        m = np.isfinite(v)
        if np.any(m):
            ax.plot(t[m], v[m], color=COLORS.get(name, 'gray'),
                    lw=2, label=name)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Объём крови (мл)')
    ax.set_title('Объём крови')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ymax = 0.0
    for name, (data, _, _) in results_dict.items():
        if 'shunt_fraction_R2L' not in data:
            continue
        t, sf = subsample(data, 'shunt_fraction_R2L', N_PLOT_POINTS_DETAIL)
        m = np.isfinite(sf)
        if not np.any(m):
            continue
        sf_pct = sf[m] * 100
        ax.plot(t[m], sf_pct, color=COLORS.get(name, 'gray'),
                lw=2, label=name)
        ymax = max(ymax, float(np.max(sf_pct)))
    ax.axhline(y=10, color='orange', ls=':', alpha=0.5, label='10% R→L')
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('R→L шунт, %')
    ax.set_title('Фракция R→L шунта')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(10.0, ymax * 1.2))

    fig.delaxes(axes[1, 2])
    ax_polar = fig.add_subplot(2, 3, 6, projection='polar')
    cats = ['P_sa', 'Q_aortic', 'V_blood', 'GFR', 'SaO2', 'Q_brain']
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]

    healthy = {}
    if 'Здоровый' in results_dict:
        hd, _, _ = results_dict['Здоровый']
        mh = steady_mask(hd)
        for c in cats:
            if c in hd:
                m = mh & np.isfinite(hd[c])
                if np.any(m):
                    healthy[c] = float(np.mean(hd[c][m]))

    for name, (data, _, _) in results_dict.items():
        vals = []
        ml = steady_mask(data)
        for c in cats:
            if c not in data:
                vals.append(0.0); continue
            m = ml & np.isfinite(data[c])
            if not np.any(m):
                vals.append(0.0); continue
            v = float(np.mean(data[c][m]))
            if name != 'Здоровый' and c in healthy and healthy[c] > 0:
                v = v / healthy[c]
            vals.append(v)
        vals += vals[:1]
        ax_polar.plot(angles, vals, 'o-', lw=2,
                      color=COLORS.get(name, 'gray'), label=name, markersize=5)
        ax_polar.fill(angles, vals, alpha=0.10, color=COLORS.get(name, 'gray'))
    ax_polar.set_xticks(angles[:-1])
    ax_polar.set_xticklabels(cats, fontsize=8)
    ax_polar.set_title('Нормированные показатели (здоровый = 1)', fontsize=10)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=7)
    ax_polar.set_ylim(0, 1.6)

    plt.tight_layout()
    plt.savefig('shunt_effect_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# Детальный отчёт
# =====================================================================

def print_detailed_report(results_dict):
    print("\n" + "=" * 100)
    print("ДЕТАЛЬНЫЙ ОТЧЁТ")
    print("=" * 100)
    print("Формат: mean ± std по установившемуся окну "
          f"(t > {STEADY_FRAC_DEFAULT:.2f}·t_end).")
    print("=" * 100)

    for name, (data, _, _) in results_dict.items():
        print(f"\n📊 {name}")
        print("-" * 50)
        print(f"  • Системное АД (P_sa)              : {format_mean_std(data, 'P_sa')} мм рт. ст.")
        print(f"  • Лёгочное АД (P_pa)               : {format_mean_std(data, 'P_pa')} мм рт. ст.")
        print(f"  • Системный выброс (Q_aortic)      : {format_mean_std(data, 'Q_aortic')} мл/с")
        print(f"  • Лёгочный кровоток (Q_pulmonary)  : {format_mean_std(data, 'Q_pulmonary')} мл/с")
        qp_qs = qp_qs_steady(data)
        if qp_qs is not None:
            print(f"  • Соотношение Qp/Qs                : {qp_qs:.2f}")

        _vsd = format_mean_std(data, 'Q_vsd', fmt='{:+.2f}')
        if _vsd != "N/A":
            _vsd_val = float(_vsd.split(' ± ')[0])
            if abs(_vsd_val) > 1.0:
                direction = ("L→R (лево-правый)" if _vsd_val > 0
                             else "R→L (право-левый)")
                print(f"  • Шунт через ДМЖП (Q_vsd)          : {_vsd} мл/с  — {direction}")

        n_tail = max(200, len(data['SaO2']) // 6)
        sao2_steady = float(np.mean(data['SaO2'][-n_tail:])) * 100

        print(f"  • Объём ЛЖ (V_lv)                  : {format_mean_std(data, 'V_lv')} мл")
        print(f"  • Объём ПЖ (V_rv)                  : {format_mean_std(data, 'V_rv')} мл")
        print(f"  • Объём крови (V_blood)            : {format_mean_std(data, 'V_blood', fmt='{:.0f}')} мл")
        print(f"  • Сатурация O₂ (SaO2)              : {sao2_steady:.1f} %")
        print(f"  • ЧСС (HR)                         : {format_mean_std(data, 'HR')} уд/мин")
        print(f"  • СКФ (GFR)                        : {format_mean_std(data, 'GFR', fmt='{:.2f}')} мл/с")
        print(f"  • Доставка O₂ мозгу (DO₂_br)       : "
              f"Q_br={format_mean_std(data, 'Q_brain', fmt='{:.2f}')} × "
              f"C_a_O₂={format_mean_std(data, 'C_a_O2', fmt='{:.3f}')}")
        print(f"  • Церебральный O₂-баланс          : "
              f"CMRO₂={format_mean_std(data, 'VO2_brain', fmt='{:.3f}')}")
        print(f"  • Потребление O₂ периферией        : "
              f"{format_mean_std(data, 'O2_consumption_periph', fmt='{:.3f}')} мл O₂/с")
        print(f"  • Поглощение O₂ лёгкими            : "
              f"{format_mean_std(data, 'O2_uptake', fmt='{:.3f}')} мл O₂/с")

        q_vsd_mean = float(np.mean(data['Q_vsd']))
        qs_mean    = float(np.mean(data['Q_aortic']))
        f_r2l = max(-q_vsd_mean, 0.0) / max(qs_mean, 1e-6)
        print(f"  • Доля R→L шунта                   : {f_r2l*100:.1f} %")

# =====================================================================
# Обёртка для joblib
# =====================================================================

def simulate_one_scenario(name, params):
    t0 = time.perf_counter()
    data = simulate_scenario(label=name, params=params)
    dt = time.perf_counter() - t0
    filename = f"vsd_results_{params['id']}.npz"
    np.savez(filename, **data,
             label=np.array(name),
             id=np.array(params['id']),
             description=np.array(params.get('description', '')))
    print(f"  [PID {os.getpid()}] 💾 {filename} ({dt:.1f}с)", flush=True)
    return name, data, params['color'], dt, filename


# =====================================================================
# Основная логика (вызывается внутри Tee-контекста)
# =====================================================================

def _main_impl(parallel: bool, n_jobs: int, log_path: str | None) -> None:
    t_start = time.perf_counter()
    dt_start = datetime.now()
    print("=" * 80)
    print(f"Запуск: {dt_start:%Y-%m-%d %H:%M:%S}")
    print("СРАВНЕНИЕ ГЕМОДИНАМИКИ: ЗДОРОВЫЙ vs ДМЖП — v4 PARALLEL")
    print("Per-scenario simulation config (from YAML):")
    for label, p in _PATIENTS.items():
        sc = extract_simulation_config(p)
        print(f"  {label:40s} t_span={sc['t_span']}  "
            f"n={sc['n_samples_t']:6d}  t_calib={sc['t_calib']:.0f}s")
    print("=" * 80)
    scenarios = _PATIENTS
    print(f"Загружено пациентов: {list(scenarios.keys())}")

    if parallel:
        if n_jobs <= 0:
            n_jobs = min(os.cpu_count() or 1, len(scenarios))
        else:
            n_jobs = min(n_jobs, len(scenarios), os.cpu_count() or 1)

        # --- Готовим initargs для воркеров ---
        if log_path:
            initializer = _init_worker_logging
            initargs = (log_path,)
            print(f"\n🚀 Запуск {len(scenarios)} симуляций параллельно "
                  f"n_jobs={n_jobs} (loky), лог: {log_path}")
        else:
            initializer = None
            initargs = ()
            print(f"\n🚀 Запуск {len(scenarios)} симуляций параллельно "
                  f"n_jobs={n_jobs} (loky), лог отключён")

        results_list = Parallel(
            n_jobs=n_jobs,
            backend='loky',
            verbose=10,
            initializer=initializer,
            initargs=initargs,
        )(
            delayed(simulate_one_scenario)(name, params)
            for name, params in scenarios.items()
        )
    else:
        print("\n🚀 Запуск последовательно...")
        results_list = [simulate_one_scenario(name, params)
                        for name, params in scenarios.items()]

    results = {}
    per_scenario_times = {}
    results_list = [r for r in results_list if r is not None and r[1] is not None]
    for name, data, color, dt, fname in results_list:
        results[name] = (data, color, name)
        per_scenario_times[name] = dt

    print("\n📊 Генерация визуализаций (последовательно)...")
    plot_enhanced_comparison(results)
    plot_shunt_effect_analysis(results)
    print_detailed_report(results)

    print("\n✅ Готово. Файлы:")
    print("   - hemodynamics_comparison_enhanced.png")
    print("   - shunt_effect_analysis.png")
    print("   - vsd_results_*.npz")
    print("\n⏱  Время по сценариям:")
    for nm, dt in per_scenario_times.items():
        print(f"   • {nm:<40s} {dt:>7.1f}с")
    t_total = time.perf_counter() - t_start
    print(f"\n⏱  Общее wall time: {t_total:.1f}с ({t_total/60:.1f} мин)")
    if parallel:
        print(f"   Сумма CPU времени: {sum(per_scenario_times.values()):.1f}с")
        print(f"   Ускорение vs последовательно: "
              f"{sum(per_scenario_times.values())/max(t_total, 1):.2f}x")
    dt_end = datetime.now()
    print(f"\n🏁 Завершение: {dt_end:%Y-%m-%d %H:%M:%S}")
    print(f"   Общее время работы: {dt_end - dt_start}")


# =====================================================================
# main — обёртка с Tee-логгером
# =====================================================================

def main(parallel: bool = True,
         n_jobs: int = 0,
         log_path: Path | None = None,
         no_log: bool = False) -> None:

    # --- Решаем путь ---
    if log_path is None and not no_log:
        log_path = _resolve_log_path(None, flat=False, no_log=False)

    if no_log or log_path is None:
        _main_impl(parallel=parallel, n_jobs=n_jobs, log_path=None)
        return

    # --- Truncate + append (кросс-платформенно для loky-workers) ---
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "w", encoding="utf-8"):
            pass
    except OSError as e:
        print(f"[WARN] Не удалось создать лог-файл {log_path}: {e}. "
              f"Продолжаем без лога.", file=sys.stderr)
        _main_impl(parallel=parallel, n_jobs=n_jobs, log_path=None)
        return

    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f"[log] Script    : {Path(sys.argv[0]).resolve().name}")
            print(f"[log] Log file  : {log_path.resolve()}")
            print(f"[log] Started   : {datetime.now():%Y-%m-%d %H:%M:%S}")
            _main_impl(
                parallel=parallel,
                n_jobs=n_jobs,
                log_path=str(log_path.resolve()),
            )
            print(f"[log] Finished  : {datetime.now():%Y-%m-%d %H:%M:%S}")
            print(f"[log] Полный лог: {log_path.resolve()}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Сравнение гемодинамики: Здоровый vs ДМЖП — v4 PARALLEL"
    )
    parser.add_argument('--no-parallel', action='store_true',
                        help='отключить параллель')
    parser.add_argument('--n-jobs', type=int, default=0,
                        help='число процессов; 0=auto = min(cpu_count, n_scenarios)')
    parser.add_argument('--log', type=str, default=None,
                        help='Путь к лог-файлу. По умолчанию — '
                             'results/run_simulation_parallel_<timestamp>.log')
    parser.add_argument('--log-flat', action='store_true',
                        help='Имя лога без timestamp: '
                             'results/run_simulation_parallel.log (перезапись).')
    parser.add_argument('--no-log', action='store_true',
                        help='Отключить запись в файл (только консоль).')
    args = parser.parse_args()

    log_path = _resolve_log_path(
        explicit=args.log,
        flat=args.log_flat,
        no_log=args.no_log,
    )
    main(
        parallel=not args.no_parallel,
        n_jobs=args.n_jobs,
        log_path=log_path,
        no_log=args.no_log,
    )