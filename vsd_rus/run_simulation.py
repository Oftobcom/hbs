#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_simulation.py
Сравнение гемодинамики: здоровый vs ДМЖП (малый / большой / Эйзенменгер).

Совместим с текущей архитектурой whole_body.WholeBodyModel:
  - Mass-balance B: sys_ven в V-mode (target_fraction=0.5, tau=300 s)
  - Мягкий барорефлекс (P_set=80, gain=0.002, k_inotropy=0.5)
  - heart с мягкими клапанами (R_mitral=0.03, R_venous=0.05)
  - peripheral с мягкой ауторегуляцией (k_O2=0.5, k_P=0.002)
  - V0_blood = 5800 мл (дефолт whole_body)

Сценарии:
  - Здоровый:       R_vsd = inf
  - Малый ДМЖП:     R_vsd = 5.0  (d_vsd ≈ 4 мм)
  - Большой ДМЖП:   R_vsd = 1.0  (d_vsd ≈ 6 мм)
  - Эйзенменгер:    R_vsd = 0.7  + ремоделирование лёгких
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from whole_body import WholeBodyModel


# =====================================================================
# Константы и палитра
# =====================================================================

COLORS = {
    'Здоровый': '#2ecc71',
    'Малый ДМЖП (R=5.0)': '#f39c12',
    'Большой ДМЖП (R=1.0)': '#e74c3c',
    'Эйзенменгер (R=0.7)': '#8e44ad',
}
LABELS = {k: k for k in COLORS}

SCENARIO_ORDER = list(COLORS.keys())

# Длительность симуляции и калибровки.
# tau_remodel = 200 с (Эйзенменгер), tau_target = 300 с (mass-balance B),
# поэтому t_end = 800 с даёт 2-3 времени релаксации.
T_END = 800.0
T_CALIB = 150.0
DT_EVAL = 0.05
# N_EVAL = int(T_END / DT_EVAL)  # 16 000 точек
N_EVAL = 8000
TAIL_FRAC = 0.75  # последние 25% симуляции для среднего


# =====================================================================
# Утилиты
# =====================================================================

def safe_savgol_filter(data, window_length, polyorder):
    """Безопасная версия savgol_filter, обрабатывающая NaN и краевые случаи."""
    from scipy.signal import savgol_filter

    data_clean = np.where(np.isfinite(data), data, np.nan)
    if np.any(np.isnan(data_clean)):
        nan_mask = np.isnan(data_clean)
        valid = np.where(~nan_mask)[0]
        if len(valid) > 1:
            data_clean[nan_mask] = np.interp(
                np.flatnonzero(nan_mask), valid, data_clean[valid]
            )
        else:
            return data

    try:
        if window_length > len(data_clean):
            window_length = (len(data_clean) if len(data_clean) % 2 == 1
                             else len(data_clean) - 1)
        if window_length < 3:
            return data_clean
        if polyorder >= window_length:
            polyorder = window_length - 1
        if polyorder < 1:
            return data_clean
        return savgol_filter(data_clean, window_length, polyorder)
    except Exception:
        return data


def subsample(data, key, n_target=5000):
    """Прореживает временной ряд data[key] до ~n_target точек."""
    step = max(1, len(data['t']) // n_target)
    return data['t'][::step], data[key][::step]


def steady_mask(data, frac=TAIL_FRAC):
    """Маска последних (1 - frac) симуляции."""
    t_start = frac * data['t'][-1]
    return data['t'] >= t_start


def clean_nans(data):
    """Заменяет NaN/Inf в числовых полях линейной интерполяцией."""
    for key in list(data.keys()):
        if not isinstance(data[key], np.ndarray):
            continue
        arr = data[key]
        if arr.ndim != 1 or arr.dtype.kind not in 'fc':
            continue
        bad = ~np.isfinite(arr)
        if not np.any(bad):
            continue
        idx = np.arange(len(arr))
        if np.any(~bad):
            arr[bad] = np.interp(idx[bad], idx[~bad], arr[~bad])
        else:
            data[key] = np.zeros_like(arr)
    return data


# =====================================================================
# Симуляция сценария
# =====================================================================

def simulate_scenario(vsd_resistance,
                      flow_dependent_lungs,
                      label,
                      color,
                      description="",
                      pressure_remodel=False,
                      pressure_sensitivity=0.06,
                      t_span=(0.0, T_END),
                      t_calib=T_CALIB):
    """
    Запускает симуляцию одного сценария.

    Возвращает (data, color, label), где data — dict с временными рядами.
    """
    # --- Initial concentrations (включая лактат, добавленный whole_body) ---
    initial_conc = {
        'tox': 0.0,
        'bilirubin': 0.5,
        'ammonia': 0.3,
        'albumin': 4.5,
        'glucose': 5.0,
        'oxygen': 0.15,
        'co2': 0.52,
        'lactate': 0.10,
    }

    blood_params = {
        'initial_concentrations': initial_conc,
        'V0': 5800.0,                 # совпадает с дефолтом whole_body
    }

    # HR_base = 70 везде; для VSD-сценариев можно чуть выше,
    # но барорефлекс всё равно подстроит HR.
    heart_params = {'hr': 70}
    if vsd_resistance != np.inf:
        heart_params['hr'] = 75

    # --- Сборка модели. Не переопределяем C_sys_art / C_pul_ven / P_*0 ---
    # (используем дефолты whole_body: C_sys_art=1.5, C_pul_ven=15,
    #  P_sa0=85, P_sv0=12, P_pv0=12).
    model = WholeBodyModel(
        baroreflex_params={
            'P_set':   80.0,
            'HR_base': 75 if vsd_resistance != np.inf else 70,
            'gain':    0.002,           # мягкий, дефолт
            'tau':     2.0,
            'k_inotropy': 0.5,
        },
        blood_params=blood_params,
        vsd_resistance=vsd_resistance,
        flow_dependent_lungs=flow_dependent_lungs,
        lungs_params={
            'flow_dependent_resistance': flow_dependent_lungs,
            'flow_sensitivity':   0.15,
            'pressure_remodel':   pressure_remodel,
            'P_pa_threshold':     25.0,
            'pressure_sensitivity': pressure_sensitivity,
            'R_remodel_max':      5.0,
            'tau_remodel':        200.0,
        },
        heart_params=heart_params,
        R_sys_peripheral=None,      # калибровка из target_MAP / target_CO
        target_MAP=85.0, target_CO=83.0,
        # C_sys_art, C_sys_ven, C_pul_ven, P_sa0, P_sv0, P_pv0 — НЕ передаём,
        # чтобы использовать дефолты whole_body.
    )

    # --- Калибровка ---
    print(f"  Калибровка (t_calib={t_calib:.0f} с)...", end=" ", flush=True)
    y0 = model.calibrate_initial_state(t_calib=t_calib)
    print("готово")

    # --- Диагностика после калибровки ---
    out0 = model.compute_outputs(0.0, y0)
    print(f"  [CHECK] после калибровки: "
          f"P_sa={out0['P_sa']:.1f}  P_sv={out0['P_sv']:.1f}  "
          f"HR={out0['HR']:.1f}  V_blood={out0['V_blood']:.0f}")

    # --- Основная симуляция ---
    t_eval = np.linspace(t_span[0], t_span[1], N_EVAL)
    print(f"  Симуляция {label} (0..{t_span[1]:.0f} с, LSODA)...",
          end=" ", flush=True)
    sol = model.simulate(
        t_span, t_eval, y0=y0,
        method='LSODA',
        rtol=1e-5, atol=1e-7,
        max_step=0.05,
    )
    print(f"завершена за {len(sol.t)} шагов")

    # --- Сборка выходов ---
    outputs = [model.compute_outputs(sol.t[i], sol.y[:, i])
               for i in range(len(sol.t))]
    data = {key: np.array([out[key] for out in outputs])
            for key in outputs[0].keys()}
    data['t'] = sol.t
    data['description'] = description

    # --- Очистка NaN/Inf ---
    data = clean_nans(data)

    return data, color, label


# =====================================================================
# Визуализация: 4 сценария на одной сетке
# =====================================================================

def _auto_ylim(ax, arr_lo, arr_hi, pad=0.05):
    """Подгоняет ylim по диапазону, если он конечен и ненулевой."""
    lo = np.nanmin(arr_lo) if np.size(arr_lo) else 0.0
    hi = np.nanmax(arr_hi) if np.size(arr_hi) else 1.0
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return
    span = hi - lo
    ax.set_ylim(lo - pad * span, hi + pad * span)


def plot_enhanced_comparison(results_dict):
    """Сводная панель 3x3 по 4 сценариям."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)

    # 1. Давления P_sa / P_pa
    ax = fig.add_subplot(gs[0, 0])
    for name, (data, _, _) in results_dict.items():
        t, ps = subsample(data, 'P_sa')
        _, pp = subsample(data, 'P_pa')
        m = np.isfinite(ps) & np.isfinite(pp)
        if not np.any(m):
            continue
        c = COLORS.get(name, 'gray')
        ax.plot(t[m], ps[m], color=c, lw=1.5, label=f"{name} (P_sa)")
        ax.plot(t[m], pp[m], color=c, lw=1.2, ls='--', alpha=0.7)
    ax.set_ylabel('Давление (мм рт. ст.)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Системное (—) и лёгочное (- -) давление')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 200)

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
    for name, (data, _, _) in results_dict.items():
        t, v_lv = subsample(data, 'V_lv')
        _, v_rv = subsample(data, 'V_rv')
        m = np.isfinite(v_lv) & np.isfinite(v_rv)
        if not np.any(m):
            continue
        c = COLORS.get(name, 'gray')
        ax.plot(t[m], v_lv[m], color=c, lw=1.2, alpha=0.8)
        ax.plot(t[m], v_rv[m], color=c, lw=1.2, ls='--', alpha=0.8)
    ax.set_ylabel('Объём (мл)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Объёмы желудочков (— ЛЖ, - - ПЖ)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 250)

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
    summary_metrics = [
        ('P_sa',   'P_sa, мм Hg',   '{:.0f}'),
        ('P_pa',   'P_pa, мм Hg',   '{:.0f}'),
        ('Qp_Qs',  'Qp/Qs',         '{:.2f}'),
        ('SaO2',   'SaO₂, %',       '{:.1f}'),
        ('V_rv',   'V_rv, мл',      '{:.0f}'),
        ('V_blood','V_blood, мл',   '{:.0f}'),
    ]

    for metric, label, fmt in summary_metrics:
        row = [label]
        for name in SCENARIO_ORDER:
            if name not in results_dict:
                row.append('N/A')
                continue
            data, _, _ = results_dict[name]
            if metric not in data:
                row.append('N/A')
                continue
            m = steady_mask(data) & np.isfinite(data[metric])
            vals = data[metric][m]
            if len(vals) == 0:
                row.append('N/A')
                continue
            v = np.mean(vals)
            if metric == 'SaO2':
                v *= 100
            row.append(fmt.format(v))
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
        t, q = subsample(data, 'Qp_Qs', 1500)
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
        m = steady_mask(data) & np.isfinite(data.get('Q_vsd', np.array([]))) \
            & np.isfinite(data.get('Qp_Qs', np.array([])))
        if not np.any(m):
            continue
        s = np.mean(data['Q_vsd'][m])
        q = np.mean(data['Qp_Qs'][m])
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
        t, v = subsample(data, 'V_blood', 800)
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
        t, sf = subsample(data, 'shunt_fraction_R2L', 800)
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

    for name, (data, _, _) in results_dict.items():
        print(f"\n📊 {name}")
        print("-" * 50)

        m = steady_mask(data)

        def _ms(key, scale=1.0, fmt="{:.1f}"):
            if key not in data:
                return "N/A"
            mm = m & np.isfinite(data[key])
            if not np.any(mm):
                return "N/A"
            v = np.mean(data[key][mm]) * scale
            s = np.std(data[key][mm]) * scale
            return fmt.format(v) + f" ± {fmt.format(s)}"

        # print(f"  • Системное АД (P_sa)        : {_ms('P_sa')} мм Hg")
        # print(f"  • Лёгочное АД (P_pa)        : {_ms('P_pa')} мм Hg")
        # print(f"  • Системный выброс (Q_aortic)    : {_ms('Q_aortic')} мл/с")
        # print(f"  • Лёгочный кровоток (Q_pulmonary) : {_ms('Q_pulmonary')} мл/с")
        # print(f"  • Соотношение Qp/Qs       : {_ms('Qp_Qs', fmt='{:.2f}')}")
        # print(f"  • Q_vsd       : {_ms('Q_vsd', fmt='{:+.2f}')} мл/с")
        # print(f"  • V_lv        : {_ms('V_lv')} мл")
        # print(f"  • V_rv        : {_ms('V_rv')} мл")
        # print(f"  • V_blood     : {_ms('V_blood', fmt='{:.0f}')} мл")
        # print(f"  • SaO2        : {_ms('SaO2', scale=100.0, fmt='{:.1f}')} %")
        # print(f"  • HR          : {_ms('HR')} уд/мин")
        # print(f"  • GFR         : {_ms('GFR', fmt='{:.2f}')} мл/с")

        # ------------------------------------------------------------------
        # Детальный отчёт по установившемуся режиму
        # ------------------------------------------------------------------
        print(f"  • Системное АД (P_sa)              : {_ms('P_sa')} мм рт. ст.  — давление в аорте")
        print(f"  • Лёгочное АД (P_pa)               : {_ms('P_pa')} мм рт. ст.  — давление в лёгочной артерии")
        print(f"  • Системный выброс (Q_aortic)      : {_ms('Q_aortic')} мл/с  — кровоток через аорту (Qs)")
        print(f"  • Лёгочный кровоток (Q_pulmonary)  : {_ms('Q_pulmonary')} мл/с  — кровоток через лёгкие (Qp)")
        print(f"  • Соотношение Qp/Qs                : {_ms('Qp_Qs', fmt='{:.2f}')}  — норма ≈ 1.0; >2.0 большой L→R, <1.0 R→L")

        # Направление шунта — печатаем отдельной строкой, если он значим
        _vsd = _ms('Q_vsd', fmt='{:+.2f}')
        if _vsd != "N/A":
            _vsd_val = float(_vsd.split(' ± ')[0])
            if abs(_vsd_val) > 1.0:
                direction = "L→R (лево-правый)" if _vsd_val > 0 else "R→L (право-левый)"
                print(f"  • Шунт через ДМЖП (Q_vsd)          : {_vsd} мл/с  — {direction}")
            else:
                print(f"  • Шунт через ДМЖП (Q_vsd)          : {_vsd} мл/с  — гемодинамически незначим")

        print(f"  • Объём ЛЖ (V_lv)                  : {_ms('V_lv')} мл  — конечно-диастолический объём левого желудочка")
        print(f"  • Объём ПЖ (V_rv)                  : {_ms('V_rv')} мл  — объём правого желудочка (растёт при L→R)")
        print(f"  • Объём крови (V_blood)            : {_ms('V_blood', fmt='{:.0f}')} мл  — общий циркулирующий объём")
        print(f"  • Сатурация O₂ (SaO2)              : {_ms('SaO2', scale=100.0, fmt='{:.1f}')} %  — <90% = гипоксемия (R→L)")
        print(f"  • ЧСС (HR)                         : {_ms('HR')} уд/мин  — текущая частота сердечных сокращений")
        print(f"  • СКФ (GFR)                        : {_ms('GFR', fmt='{:.2f}')} мл/с  — скорость клубочковой фильтрации почек")

        # Доля R→L шунта в системном выбросе — печатаем только если есть
        _sh = _ms('shunt_fraction_R2L', scale=100.0, fmt='{:.1f}')
        if _sh != "N/A":
            _sh_val = float(_sh.split(' ± ')[0])
            if _sh_val > 1.0:
                print(f"  • Доля R→L шунта                   : {_sh} %  — часть венозной крови идёт в аорту, минуя лёгкие")

# =====================================================================
# main
# =====================================================================

def main():
    print("=" * 80)
    print("СРАВНЕНИЕ ГЕМОДИНАМИКИ: ЗДОРОВЫЙ vs ДМЖП")
    print("=" * 80)

    scenarios = {
        'Здоровый': {
            'vsd_resistance': np.inf,
            'flow_dependent_lungs': False,
            'pressure_remodel': False,
            'color': COLORS['Здоровый'],
            'description': 'Здоровое сердце без дефекта',
        },
        'Малый ДМЖП (R=5.0)': {
            'vsd_resistance': 5.0,
            'flow_dependent_lungs': False,
            'pressure_remodel': False,
            'color': COLORS['Малый ДМЖП (R=5.0)'],
            'description': 'Малый дефект (d≈4 мм), незначительный L→R шунт',
        },
        'Большой ДМЖП (R=1.0)': {
            'vsd_resistance': 1.0,
            'flow_dependent_lungs': True,
            'pressure_remodel': False,
            'color': COLORS['Большой ДМЖП (R=1.0)'],
            'description': 'Большой дефект (d≈6 мм), выраженный L→R шунт',
        },
        'Эйзенменгер (R=0.7)': {
            'vsd_resistance': 0.7,
            'flow_dependent_lungs': True,
            'pressure_remodel': True,
            'pressure_sensitivity': 0.06,
            'color': COLORS['Эйзенменгер (R=0.7)'],
            'description': 'Ремоделирование лёгких → R→L шунт',
        },
    }

    print("\n🚀 Запуск симуляций...")
    results = {}

    for name, params in scenarios.items():
        print(f"\n▶ Сценарий: {name}")
        print(f"   {params['description']}")

        data, color, label = simulate_scenario(
            vsd_resistance=params['vsd_resistance'],
            flow_dependent_lungs=params['flow_dependent_lungs'],
            label=name,
            color=params['color'],
            description=params['description'],
            pressure_remodel=params.get('pressure_remodel', False),
            pressure_sensitivity=params.get('pressure_sensitivity', 0.06),
        )
        results[name] = (data, color, label)

        filename = f"vsd_results_{name.replace(' ', '_').replace('(', '').replace(')', '')}.npz"
        np.savez(filename, **data)
        print(f"   💾 {filename}")

    print("\n📊 Генерация визуализаций...")
    plot_enhanced_comparison(results)
    plot_shunt_effect_analysis(results)
    print_detailed_report(results)

    print("\n✅ Готово.")
    print("📁 Файлы:")
    print("   - hemodynamics_comparison_enhanced.png")
    print("   - shunt_effect_analysis.png")
    print("   - vsd_results_*.npz")


if __name__ == "__main__":
    main()