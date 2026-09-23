#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_simulation_parallel.py — параллельная версия run_simulation.py

"""

import matplotlib
matplotlib.use('Agg')  # важно для joblib/loky — не форкать GUI backend
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import os
from datetime import datetime
from joblib import Parallel, delayed
from whole_body import WholeBodyModel
from utils import (safe_savgol_filter, subsample, steady_mask,
                   clean_nans, steady_mean, steady_mean_std,
                   format_mean_std, qp_qs_steady, auto_ylim,
                   SUBSAMPLE_N_TARGET, STEADY_FRAC_DEFAULT)
from physio_config import load_all_patients

warnings.filterwarnings('ignore')

_PATIENTS = load_all_patients()
COLORS = {p['label']: p['color'] for p in _PATIENTS.values()}
SCENARIO_ORDER = [p['label'] for p in sorted(_PATIENTS.values(), key=lambda c: int(c['order']))]

# =====================================================================
# Константы v4
# =====================================================================
T_END   = 800.0
T_CALIB = 800.0
T_CALIB_HEALTHY = 400.0
N_EVAL  = 4000   # v3: было 10000 -> 4000 (для отчетов хватает)
DT_EVAL = T_END / (N_EVAL - 1)
MAX_STEP = 0.10  # v3: было 0.07 -> 0.10 (-30% RHS вызовов)

TOTAL_VO2_BASE  = 4.2
PERIPH_VO2_BASE = 1.5
GAS_EX_VO2_BASE = TOTAL_VO2_BASE - PERIPH_VO2_BASE

N_PLOT_POINTS = 4000
N_PLOT_POINTS_DETAIL = 1200
STEADY_FRAC = 0.75

# =====================================================================
# Симуляция одного сценария (оптимизированная)
# =====================================================================
def simulate_scenario(vsd_resistance,
                      flow_dependent_lungs,
                      label,
                      pressure_remodel=False,
                      pressure_sensitivity=0.06,
                      R_remodel_max=5.0,
                      tau_remodel=200.0,
                      HR_base=None,
                      E_max_rv_override=None,
                      flow_sensitivity=0.15,
                      t_span=(0.0, T_END),
                      t_calib=T_CALIB):

    initial_conc = {
        'tox': 0.0, 'bilirubin': 0.5, 'ammonia': 0.3,
        'albumin': 4.5, 'glucose': 5.0, 'oxygen': 0.15,
        'co2': 0.52, 'lactate': 0.10,
    }
    blood_params = {'initial_concentrations': initial_conc, 'V0': 5800.0}

    if HR_base is None:
        HR_base = 75 if vsd_resistance != np.inf else 70
    heart_params = {'hr': HR_base, 'R_vsd': vsd_resistance}
    if E_max_rv_override is not None:
        heart_params['E_max_rv'] = E_max_rv_override

    model = WholeBodyModel(
        baroreflex_params={'P_set': 80.0, 'HR_base': HR_base},
        blood_params=blood_params,
        flow_dependent_lungs=flow_dependent_lungs,
        lungs_params={
            'flow_sensitivity': flow_sensitivity,
            'pressure_remodel': pressure_remodel,
            'P_pa_threshold': 25.0,
            'pressure_sensitivity': pressure_sensitivity,
            'R_remodel_max': R_remodel_max,
            'tau_remodel': tau_remodel,
        },
        heart_params=heart_params,
        peripheral_params={'VO2_base': PERIPH_VO2_BASE},
        gas_exchange_params={'VO2_base': GAS_EX_VO2_BASE},
        R_sys_peripheral=None,
        target_MAP=85.0, target_CO=83.0,
    )

    # Адаптивный t_calib
    t_calib_eff = t_calib
    if not pressure_remodel and t_calib >= 600:
        if t_calib == T_CALIB:
            t_calib_eff = T_CALIB_HEALTHY

    print(f"  [PID {os.getpid()}] [{label}] Калибровка t_calib={t_calib_eff:.0f}с...", flush=True)
    y0 = model.calibrate_initial_state(t_calib=t_calib_eff)

    out0 = model.compute_outputs(0.0, y0)
    print(f"  [PID {os.getpid()}] [{label}] CHECK P_sa={out0['P_sa']:.1f} P_sv={out0['P_sv']:.1f} HR={out0['HR']:.1f} V_blood={out0['V_blood']:.0f}", flush=True)

    n_pts = N_EVAL
    t_eval = np.linspace(t_span[0], t_span[1], n_pts)
    print(f"  [PID {os.getpid()}] [{label}] Симуляция 0..{t_span[1]:.0f}с LSODA max_step={MAX_STEP} {n_pts} точек...", flush=True)
    sol = model.simulate(
        t_span, t_eval, y0=y0,
        method='LSODA',
        rtol=1e-4, atol=1e-5,
        max_step=MAX_STEP,
    )
    print(f"  [PID {os.getpid()}] [{label}] готово {sol.t.size} точек, {sol.nfev} RHS, cache_hits={getattr(model, '_flow_cache_hits', 0)}", flush=True)

    # --- ОПТИМИЗАЦИЯ 1: только установившееся окно ---
    if sol.t.size > 0:
        mask_steady = sol.t > STEADY_FRAC * sol.t[-1]
        if np.sum(mask_steady) < 100:
            mask_steady = np.ones_like(sol.t, dtype=bool)
            mask_steady[: int((1-STEADY_FRAC)*len(mask_steady))] = False
    else:
        mask_steady = np.array([], dtype=bool)
    idx_steady = np.where(mask_steady)[0]

    outputs = [model.compute_outputs(sol.t[i], sol.y[:, i]) for i in idx_steady]
    if len(outputs) == 0:
        raise RuntimeError("Нет точек в установившемся окне")
    data = {key: np.array([out[key] for out in outputs]) for key in outputs[0].keys()}
    data['t'] = sol.t[idx_steady]
    data['_t_full'] = sol.t  # для графиков если нужно
    data = clean_nans(data)
    return data


def plot_enhanced_comparison(results_dict):
    """Сводная панель 3x3 по 4 сценариям."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)

    # 1. Давления P_sa / P_pa
    ax = fig.add_subplot(gs[0, 0])
    ys_axis = []                       # <-- накопитель всех y-значений
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
        all_y = np.concatenate(ys_axis)
        auto_ylim(ax, all_y)

    # 2. Кровотоки Q_aortic / Q_pulmonary
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

    # 3. Qp/Qs
    ax = fig.add_subplot(gs[0, 2])
    for name, (data, _, _) in results_dict.items():
        t, qp_qs = subsample(data, 'Qp_Qs')
        m = np.isfinite(qp_qs)
        if not np.any(m):
            continue
        vals = qp_qs[m]
        if len(vals) > 100:
            vals = safe_savgol_filter(vals, 101, 3)
        ax.plot(t[m], vals, color=COLORS.get(name, 'gray'), lw=2, label=name)
    ax.set_ylabel('Qp/Qs')
    ax.set_xlabel('Время (с)')
    ax.set_title('Соотношение лёгочного и системного кровотока')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='black', ls='--', alpha=0.5)
    ax.axhline(y=1.5, color='orange', ls=':', alpha=0.5)
    ax.axhline(y=2.0, color='red', ls=':', alpha=0.5)
    ax.set_ylim(0.5, 5.0)

    # 4. Шунт через ДМЖП
    ax = fig.add_subplot(gs[1, 0])
    for name, (data, _, _) in results_dict.items():
        t, q_vsd = subsample(data, 'Q_vsd')
        m = np.isfinite(q_vsd)
        if not np.any(m):
            continue
        ax.plot(t[m], q_vsd[m], color=COLORS.get(name, 'gray'), lw=1.5, label=name)
    ax.set_ylabel('Шунт (мл/с)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Шунт через ДМЖП')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', ls='--', alpha=0.5)

    # 5. Объёмы желудочков
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
        all_y = np.concatenate(ys_axis)
        auto_ylim(ax, all_y)

    # 6. Объём крови
    ax = fig.add_subplot(gs[1, 2])
    for name, (data, _, _) in results_dict.items():
        t, v = subsample(data, 'V_blood')
        m = np.isfinite(v)
        if not np.any(m):
            continue
        ax.plot(t[m], v[m], color=COLORS.get(name, 'gray'), lw=1.5, label=name)
    ax.set_ylabel('Объём крови (мл)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Волемический статус')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)

    # 7. Мозговой кровоток
    ax = fig.add_subplot(gs[2, 0])
    for name, (data, _, _) in results_dict.items():
        t, q = subsample(data, 'Q_brain')
        m = np.isfinite(q)
        if not np.any(m):
            continue
        ax.plot(t[m], q[m], color=COLORS.get(name, 'gray'), lw=1.5, label=name)
    ax.set_ylabel('Мозговой кровоток (мл/с)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Церебральная гемодинамика')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)

    # 8. SaO2
    ax = fig.add_subplot(gs[2, 1])
    for name, (data, _, _) in results_dict.items():
        t, s = subsample(data, 'SaO2')
        m = np.isfinite(s)
        if not np.any(m):
            continue
        ax.plot(t[m], s[m] * 100, color=COLORS.get(name, 'gray'), lw=1.5, label=name)
    ax.axhline(y=90, color='orange', ls=':', alpha=0.5, label='SaO2 = 90%')
    ax.set_ylabel('SaO₂ (%)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Артериальная сатурация O₂')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(60, 100)

    # 9. Сводная таблица
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    table_data = [['Показатель'] + [s[:15] for s in SCENARIO_ORDER]]
    # metric, label, fmt, mode
    # mode: 'mean'         — среднее по steady-окну
    #       'mean_scaled'  — среднее × 100 (для SaO2)
    #       'qp_qs'        — Qp/Qs из средних потоков
    summary_metrics = [
        ('P_sa',   'P_sa, мм Hg',   '{:.0f}',   'mean'),
        ('P_pa',   'P_pa, мм Hg',   '{:.0f}',   'mean'),
        ('Qp_Qs',  'Qp/Qs',         '{:.2f}',   'qp_qs'),
        ('SaO2',   'SaO₂, %',       '{:.1f}',   'mean_scaled'),
        ('V_rv',   'V_rv, мл',      '{:.0f}',   'mean'),
        ('V_blood','V_blood, мл',   '{:.0f}',   'mean'),
    ]

    for metric, label, fmt, mode in summary_metrics:
        row = [label]
        for name in SCENARIO_ORDER:
            if name not in results_dict:
                row.append('N/A'); continue
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

    ax.set_title('Сводка установившихся значений', fontsize=10, fontweight='bold')

    fig.suptitle('Гемодинамика: Здоровый vs ДМЖП (малый / большой / Эйзенменгер)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hemodynamics_comparison_enhanced.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_shunt_effect_analysis(results_dict):
    """Анализ влияния размера шунта на гемодинамику."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Анализ влияния размера ДМЖП', fontsize=14, fontweight='bold')

    # 1. Динамика Qp/Qs
    ax = axes[0, 0]
    for name, (data, _, _) in results_dict.items():
        t, q = subsample(data, 'Qp_Qs', N_PLOT_POINTS_DETAIL)
        m = np.isfinite(q)
        if np.any(m):
            ax.plot(t[m], q[m], color=COLORS.get(name, 'gray'), lw=2, label=name)
    ax.axhline(y=1.0, color='black', ls='--', alpha=0.5)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Qp/Qs')
    ax.set_title('Динамика Qp/Qs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Давления: bar chart
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

    # 3. Корреляция шунт → Qp/Qs
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

    # 4. Объём крови
    ax = axes[1, 0]
    for name, (data, _, _) in results_dict.items():
        t, v = subsample(data, 'V_blood', N_PLOT_POINTS_DETAIL)
        m = np.isfinite(v)
        if np.any(m):
            ax.plot(t[m], v[m], color=COLORS.get(name, 'gray'), lw=2, label=name)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Объём крови (мл)')
    ax.set_title('Объём крови')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Фракция R→L
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
        ax.plot(t[m], sf_pct, color=COLORS.get(name, 'gray'), lw=2, label=name)
        ymax = max(ymax, float(np.max(sf_pct)))
    ax.axhline(y=10, color='orange', ls=':', alpha=0.5, label='10% R→L')
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('R→L шунт, %')
    ax.set_title('Фракция R→L шунта')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(10.0, ymax * 1.2))

    # 6. Радарная диаграмма
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
    plt.show()
    plt.close(fig)


# =====================================================================
# Детальный отчёт в консоль
# =====================================================================

def print_detailed_report(results_dict):
    print("\n" + "=" * 100)
    print("ДЕТАЛЬНЫЙ ОТЧЁТ")
    print("=" * 100)
    print("Формат: mean ± std по установившемуся окну "
          f"(t > {STEADY_FRAC_DEFAULT:.2f}·t_end).")
    print("  – mean — среднее значение метрики за окно;")
    print("  – std  — стандартное отклонение за то же окно "
          "(пульсовые колебания + остаточный дрейф).")
    print("Единицы указаны рядом с каждой строкой; "
          "SaO2 и shunt_fraction_R2L приведены в %.")
    print("=" * 100)

    for name, (data, _, _) in results_dict.items():
        print(f"\n📊 {name}")
        print("-" * 50)


        # ------------------------------------------------------------------
        # Детальный отчёт по установившемуся режиму
        # ------------------------------------------------------------------
        print(f"  • Системное АД (P_sa)              : {format_mean_std(data, 'P_sa')} мм рт. ст.  — давление в аорте")
        print(f"  • Лёгочное АД (P_pa)               : {format_mean_std(data, 'P_pa')} мм рт. ст.  — давление в лёгочной артерии")
        print(f"  • Системный выброс (Q_aortic)      : {format_mean_std(data, 'Q_aortic')} мл/с  — кровоток через аорту (Qs)")
        print(f"  • Лёгочный кровоток (Q_pulmonary)  : {format_mean_std(data, 'Q_pulmonary')} мл/с  — кровоток через лёгкие (Qp)")
        qp_qs = qp_qs_steady(data)
        if qp_qs is not None:
            print(f"  • Соотношение Qp/Qs                : {qp_qs:.2f}  — норма ≈ 1.0; >2.0 большой L→R, <1.0 R→L")

        # Направление шунта — печатаем отдельной строкой, если он значим
        _vsd = format_mean_std(data, 'Q_vsd', fmt='{:+.2f}')
        if _vsd != "N/A":
            _vsd_val = float(_vsd.split(' ± ')[0])
            if abs(_vsd_val) > 1.0:
                direction = "L→R (лево-правый)" if _vsd_val > 0 else "R→L (право-левый)"
                print(f"  • Шунт через ДМЖП (Q_vsd)          : {_vsd} мл/с  — {direction}")
            else:
                print(f"  • Шунт через ДМЖП (Q_vsd)          : {_vsd} мл/с  — гемодинамически незначим")

        print(f"  • Объём ЛЖ (V_lv)                  : {format_mean_std(data, 'V_lv')} мл  — конечно-диастолический объём левого желудочка")
        print(f"  • Объём ПЖ (V_rv)                  : {format_mean_std(data, 'V_rv')} мл  — объём правого желудочка (растёт при L→R)")
        print(f"  • Объём крови (V_blood)            : {format_mean_std(data, 'V_blood', fmt='{:.0f}')} мл  — общий циркулирующий объём")
        print(f"  • Сатурация O₂ (SaO2)              : {format_mean_std(data, 'SaO2', scale=100.0, fmt='{:.1f}')} %  — <90% = гипоксемия (R→L)")
        print(f"  • ЧСС (HR)                         : {format_mean_std(data, 'HR')} уд/мин  — текущая частота сердечных сокращений")
        print(f"  • СКФ (GFR)                        : {format_mean_std(data, 'GFR', fmt='{:.2f}')} мл/с  — скорость клубочковой фильтрации почек")
        print(f"  • Потребление O₂ мозгом            : {format_mean_std(data, 'O2_consumption', fmt='{:.3f}')} мл O₂/с  — утилизация O₂ церебральной тканью")
        # --- Церебральный O₂-баланс: градиент здоровый ≈ компенс. > декомпенс. ---
        _sao2_v = format_mean_std(data, 'SaO2', scale=100.0, fmt='{:.1f}')
        _cao2_v = format_mean_std(data, 'C_a_O2', fmt='{:.3f}')
        _o2_v   = format_mean_std(data, 'O2_consumption', fmt='{:.3f}')
        _qbr_v  = format_mean_std(data, 'Q_brain', fmt='{:.2f}')
        print(f"  • Церебральный O₂-баланс          : "
            f"SaO₂={_sao2_v} %  |  C_a_O₂={_cao2_v} мл/мл  |  "
            f"Q_br={_qbr_v} мл/с  |  CMRO₂={_o2_v} мл/с")        
        print(f"  • Потребление O₂ периферией        : {format_mean_std(data, 'O2_consumption_periph', fmt='{:.3f}')} мл O₂/с  — утилизация O₂ периферической тканью")
        print(f"  • Поглощение O₂ лёгкими            : {format_mean_std(data, 'O2_uptake', fmt='{:.3f}')} мл O₂/с  — поглощение O₂ лёгкими")

        # Доля R→L шунта в системном выбросе — печатаем только если есть
        _sh = format_mean_std(data, 'shunt_fraction_R2L', scale=100.0, fmt='{:.1f}')
        if _sh != "N/A":
            _sh_val = float(_sh.split(' ± ')[0])
            if _sh_val > 1.0:
                print(f"  • Доля R→L шунта                   : {_sh} %  — часть венозной крови идёт в аорту, минуя лёгкие")

# =====================================================================
# Обертка для joblib (должна быть top-level для pickle)
# =====================================================================
def simulate_one_scenario(name, params):
    t0 = time.perf_counter()
    data = simulate_scenario(
        vsd_resistance=params['vsd_resistance'],
        flow_dependent_lungs=params['flow_dependent_lungs'],
        label=name,
        pressure_remodel=params['pressure_remodel'],
        pressure_sensitivity=params.get('pressure_sensitivity', 0.06),
        R_remodel_max=params.get('R_remodel_max', 5.0),
        tau_remodel=params.get('tau_remodel', 200.0),
        HR_base=params.get('HR_base', None),
        E_max_rv_override=params.get('E_max_rv', None),
        flow_sensitivity=params.get('flow_sensitivity', 0.15),
    )
    dt = time.perf_counter() - t0
    filename = f"vsd_results_{params['id']}.npz"
    np.savez(filename, **data,
            label=np.array(name),
            id=np.array(params['id']),
            description=np.array(params.get('description','')))
    print(f"  [PID {os.getpid()}] 💾 {filename} ({dt:.1f}с)", flush=True)
    return name, data, params['color'], dt, filename

# =====================================================================
# main
# =====================================================================

def main(parallel=True, n_jobs=5):
    t_start = time.perf_counter()
    dt_start = datetime.now()
    print("="*80)
    print(f"Запуск: {dt_start:%Y-%m-%d %H:%M:%S}")
    print("СРАВНЕНИЕ ГЕМОДИНАМИКИ: ЗДОРОВЫЙ vs ДМЖП — v4 PARALLEL")
    print(f"N_EVAL={N_EVAL}, MAX_STEP={MAX_STEP}, STEADY_FRAC={STEADY_FRAC}, t_calib healthy={T_CALIB_HEALTHY}")
    print("="*80)
    scenarios = _PATIENTS
    print(f"Загружено пациентов: {list(scenarios.keys())}")

    if parallel:
        n_jobs = min(n_jobs, len(scenarios), os.cpu_count() or 1)
        print(f"\n🚀 Запуск {len(scenarios)} симуляций параллельно n_jobs={n_jobs} (loky)...")
        results_list = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
            delayed(simulate_one_scenario)(name, params) for name, params in scenarios.items()
        )
    else:
        print("\n🚀 Запуск последовательно...")
        results_list = [simulate_one_scenario(name, params) for name, params in scenarios.items()]

    results = {}
    per_scenario_times = {}
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
        print(f"   Ускорение vs последовательно: {sum(per_scenario_times.values())/max(t_total,1):.2f}x")
    dt_end = datetime.now()
    print(f"\n🏁 Завершение: {dt_end:%Y-%m-%d %H:%M:%S}")
    print(f"   Общее время работы: {dt_end - dt_start}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-parallel', action='store_true', help='отключить параллель')
    parser.add_argument('--n-jobs', type=int, default=5, help='число процессов')
    args = parser.parse_args()
    main(parallel=not args.no_parallel, n_jobs=args.n_jobs)
