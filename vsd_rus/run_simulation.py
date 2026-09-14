#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенный скрипт для сравнения здорового человека и пациентов с ДМЖП.
Включает расширенные сценарии, более детальный вывод и улучшенную обработку.
"""

import numpy as np
import matplotlib.pyplot as plt
from whole_body import WholeBodyModel
import warnings
warnings.filterwarnings('ignore')


# Единая палитра для всех сценариев
COLORS = {
    'Здоровый': '#2ecc71',
    'Малый ДМЖП (R=5.0)': '#f39c12',
    'Большой ДМЖП (R=1.0)': '#e74c3c',
    'Эйзенменгер (R=0.7)': '#8e44ad',
}
LABELS = {k: k for k in COLORS}


def safe_savgol_filter(data, window_length, polyorder):
    """Безопасная версия savgol_filter, обрабатывающая ошибки"""
    from scipy.signal import savgol_filter

    data_clean = np.where(np.isfinite(data), data, np.nan)

    if np.any(np.isnan(data_clean)):
        nan_mask = np.isnan(data_clean)
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) > 1:
            data_clean[nan_mask] = np.interp(
                np.flatnonzero(nan_mask),
                valid_indices,
                data_clean[valid_indices]
            )
        else:
            return data

    try:
        if window_length > len(data_clean):
            window_length = len(data_clean) if len(data_clean) % 2 == 1 else len(data_clean) - 1
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


def simulate_scenario(vsd_resistance, flow_dependent_lungs, label, color,
                      t_span=(0, 600), t_eval=None, description="",
                      pressure_remodel=False,
                      pressure_sensitivity=0.04):
    """Запускает симуляцию для заданного сопротивления ДМЖП."""
    if t_eval is None:
        # Целевой шаг 0.05 с ≈ 16 точек на кардиоцикл при HR=75
        dt_target = 0.05
        n_pts = int(t_span[1] / dt_target)
        n_pts = max(n_pts, 6000)
        n_pts = min(n_pts, 20000)
        t_eval = np.linspace(t_span[0], t_span[1], n_pts)

    initial_conc = {
        'tox': 0.0,
        'bilirubin': 0.5,
        'ammonia': 0.3,
        'albumin': 4.5,
        'glucose': 5.0,
        'oxygen': 0.15,
        'co2': 0.52,
    }

    blood_params = {
        'initial_concentrations': initial_conc,
        'V0': 5000.0
    }

    heart_params = {}
    if vsd_resistance != np.inf:
        heart_params = {'hr': 75}

    model = WholeBodyModel(
        baroreflex_params={'P_set': 90,
                           'HR_base': 75 if vsd_resistance != np.inf else 70,
                           'gain': 0.01, 'tau': 2.0},
        blood_params=blood_params,
        vsd_resistance=vsd_resistance,
        flow_dependent_lungs=flow_dependent_lungs,
        lungs_params={
            'flow_dependent_resistance': flow_dependent_lungs,
            'flow_sensitivity': 0.15,
            'pressure_remodel': pressure_remodel,
            'P_pa_threshold': 25.0,
            'pressure_sensitivity': pressure_sensitivity,
            'R_remodel_max': 5.0,
            'tau_remodel': 200.0,
        },
        heart_params=heart_params,
        R_sys_peripheral=None,
        target_MAP=85.0, target_CO=83.0,
        C_sys_art=2.0,
        C_sys_ven=12.0,
        C_pul_ven=5.0
    )

    # Проверка калибровки — среднее за один кардиоцикл
    y0 = model.calibrate_initial_state(t_calib=10)
    T_cycle = 60.0 / model.heart.hr_base
    t_samples = np.linspace(0, T_cycle, 25)
    P_sa_vals = []
    Q_aortic_vals = []
    for ti in t_samples:
        out_i = model.compute_outputs(ti, y0)
        P_sa_vals.append(out_i['P_sa'])
        Q_aortic_vals.append(out_i['Q_aortic'])

    P_sa_mean = np.mean(P_sa_vals)
    CO_mean = np.mean(Q_aortic_vals)
    R_total_mean = P_sa_mean / max(CO_mean, 1e-6)
    print(f"  [CHECK] R_total={R_total_mean:.2f} MAP={P_sa_mean:.0f} CO={CO_mean:.0f} "
          f"(цель 85±5 и 80±10)")

    print(f"  Симуляция {label}...", end=" ", flush=True)
    # sol = model.simulate(t_span, t_eval, method='BDF', rtol=1e-5)
    sol = model.simulate(t_span, t_eval, method='RK45', rtol=1e-5, atol=1e-7)

    print(f"завершена за {len(sol.t)} шагов")

    outputs = []
    for i, ti in enumerate(sol.t):
        out = model.compute_outputs(ti, sol.y[:, i])
        outputs.append(out)

    data = {key: np.array([out[key] for out in outputs]) for key in outputs[0].keys()}
    data['t'] = sol.t
    data['description'] = description

    # Очистка от inf/NaN
    for key in data:
        if isinstance(data[key], np.ndarray):
            if np.any(~np.isfinite(data[key])):
                valid_mask = np.isfinite(data[key])
                if np.any(valid_mask):
                    indices = np.arange(len(data[key]))
                    data[key][~valid_mask] = np.interp(
                        indices[~valid_mask],
                        indices[valid_mask],
                        data[key][valid_mask]
                    )
                else:
                    data[key] = np.zeros_like(data[key])

    return data, color, label


def steady_mask(data, frac=0.8):
    """Маска для последних (1-frac) симуляции."""
    t_start = frac * data['t'][-1]
    return data['t'] >= t_start


def plot_enhanced_comparison(results_dict):
    """Улучшенная визуализация сравнения"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Системное и лёгочное давление
    ax1 = fig.add_subplot(gs[0, 0])
    for name, (data, _, _) in results_dict.items():
        if 'P_sa' in data and 'P_pa' in data:
            t, ps = subsample(data, 'P_sa', 5000)
            _, pp = subsample(data, 'P_pa', 5000)
            mask = np.isfinite(ps) & np.isfinite(pp)
            if np.any(mask):
                ax1.plot(t[mask], ps[mask], color=COLORS.get(name, 'gray'),
                         lw=1.5, label=f"{LABELS.get(name, name)} (P_sa)")
                ax1.plot(t[mask], pp[mask], color=COLORS.get(name, 'gray'),
                         lw=1.5, linestyle='--', alpha=0.7)
    ax1.set_ylabel('Давление (мм рт. ст.)')
    ax1.set_xlabel('Время (с)')
    ax1.set_title('Системное (—) и лёгочное (- -) давление')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 160)

    # 2. Системный и лёгочный кровоток
    ax2 = fig.add_subplot(gs[0, 1])
    for name, (data, _, _) in results_dict.items():
        if 'Q_aortic' in data and 'Q_pulmonary' in data:
            t, qa = subsample(data, 'Q_aortic', 5000)
            _, qp = subsample(data, 'Q_pulmonary', 5000)
            mask = np.isfinite(qa) & np.isfinite(qp)
            if np.any(mask):
                ax2.plot(t[mask], qa[mask], color=COLORS.get(name, 'gray'),
                         lw=1.5, label=f"{LABELS.get(name, name)} (Qs)")
                ax2.plot(t[mask], qp[mask], color=COLORS.get(name, 'gray'),
                         lw=1.5, linestyle='--', alpha=0.7)
    ax2.set_ylabel('Кровоток (мл/с)')
    ax2.set_xlabel('Время (с)')
    ax2.set_title('Системный (—) и лёгочный (- -) кровоток')
    ax2.legend(loc='upper right', fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 200)

    # 3. Соотношение Qp/Qs
    ax3 = fig.add_subplot(gs[0, 2])
    for name, (data, _, _) in results_dict.items():
        if 'Qp_Qs' in data:
            t, qp_qs = subsample(data, 'Qp_Qs', 5000)
            mask = np.isfinite(qp_qs)
            if np.any(mask):
                qp_qs_clean = qp_qs[mask]
                t_clean = t[mask]
                if len(qp_qs_clean) > 50:
                    try:
                        window = min(51, len(qp_qs_clean) // 10 * 2 + 1)
                        if window % 2 == 0:
                            window += 1
                        if window >= 3:
                            qp_qs_clean = safe_savgol_filter(qp_qs_clean, window, 3)
                    except Exception:
                        pass
                ax3.plot(t_clean, qp_qs_clean, color=COLORS.get(name, 'gray'),
                         lw=2, label=LABELS.get(name, name))
    ax3.set_ylabel('Qp/Qs')
    ax3.set_xlabel('Время (с)')
    ax3.set_title('Соотношение лёгочного и системного кровотока')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Норма')
    ax3.axhline(y=1.5, color='orange', linestyle=':', alpha=0.5)
    ax3.axhline(y=2.0, color='red', linestyle=':', alpha=0.5)
    ax3.set_ylim(0.5, 3.0)

    # 4. Шунт через ДМЖП
    ax4 = fig.add_subplot(gs[1, 0])
    for name, (data, _, _) in results_dict.items():
        if 'Q_vsd' in data:
            t, q_vsd = subsample(data, 'Q_vsd', 5000)
            mask = np.isfinite(q_vsd)
            if np.any(mask):
                ax4.plot(t[mask], q_vsd[mask], color=COLORS.get(name, 'gray'),
                         lw=1.5, label=LABELS.get(name, name))
    ax4.set_ylabel('Шунт (мл/с)')
    ax4.set_xlabel('Время (с)')
    ax4.set_title('Объём шунта через ДМЖП')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # 5. Объёмы желудочков
    ax5 = fig.add_subplot(gs[1, 1])
    for name, (data, _, _) in results_dict.items():
        if 'V_lv' in data and 'V_rv' in data:
            t, v_lv = subsample(data, 'V_lv', 5000)
            _, v_rv = subsample(data, 'V_rv', 5000)
            mask = np.isfinite(v_lv) & np.isfinite(v_rv)
            if np.any(mask):
                ax5.plot(t[mask], v_lv[mask], color=COLORS.get(name, 'gray'),
                         lw=1.2, alpha=0.7)
                ax5.plot(t[mask], v_rv[mask], color=COLORS.get(name, 'gray'),
                         lw=1.2, linestyle='--', alpha=0.7)
    ax5.set_ylabel('Объём (мл)')
    ax5.set_xlabel('Время (с)')
    ax5.set_title('Объёмы желудочков (— ЛЖ, - - ПЖ)')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 200)

    # 6. Объём крови
    ax6 = fig.add_subplot(gs[1, 2])
    for name, (data, _, _) in results_dict.items():
        if 'V_blood' in data:
            t, v_blood = subsample(data, 'V_blood', 5000)
            mask = np.isfinite(v_blood)
            if np.any(mask):
                ax6.plot(t[mask], v_blood[mask], color=COLORS.get(name, 'gray'),
                         lw=1.5, label=LABELS.get(name, name))
    ax6.set_ylabel('Объём крови (мл)')
    ax6.set_xlabel('Время (с)')
    ax6.set_title('Волемический статус')
    ax6.legend(loc='lower left', fontsize=7)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(4500, 5500)

    # 7. Мозговой кровоток
    ax7 = fig.add_subplot(gs[2, 0])
    for name, (data, _, _) in results_dict.items():
        if 'Q_brain' in data:
            t, q_brain = subsample(data, 'Q_brain', 5000)
            mask = np.isfinite(q_brain)
            if np.any(mask):
                ax7.plot(t[mask], q_brain[mask], color=COLORS.get(name, 'gray'),
                         lw=1.5, label=LABELS.get(name, name))
    ax7.set_ylabel('Мозговой кровоток (мл/с)')
    ax7.set_xlabel('Время (с)')
    ax7.set_title('Церебральная гемодинамика')
    ax7.legend(loc='upper right', fontsize=7)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0, 100)

    # 8. SaO2
    ax8 = fig.add_subplot(gs[2, 1])
    for name, (data, _, _) in results_dict.items():
        if 'SaO2' in data:
            t, sao2 = subsample(data, 'SaO2', 5000)
            mask = np.isfinite(sao2)
            if np.any(mask):
                ax8.plot(t[mask], sao2[mask] * 100, color=COLORS.get(name, 'gray'),
                         lw=1.5, label=LABELS.get(name, name))
    ax8.set_ylabel('SaO₂ (%)')
    ax8.set_xlabel('Время (с)')
    ax8.set_title('Артериальная сатурация O₂')
    ax8.axhline(y=90, color='orange', linestyle=':', alpha=0.5, label='Гипоксемия < 90%')
    ax8.legend(loc='lower right', fontsize=7)
    ax8.set_ylim(60, 100)

    # 9. Сводная таблица — все 4 сценария
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('tight')
    ax9.axis('off')

    table_data = [['Показатель', 'Здоровый', 'Малый ДМЖП', 'Большой ДМЖП', 'Эйзенменгер']]

    steady_metrics = [
        ('P_sa', 'АД сист., мм рт.ст.', '{:.0f}'),
        ('P_pa', 'АД лёг., мм рт.ст.', '{:.0f}'),
        ('Qp_Qs', 'Qp/Qs', '{:.2f}'),
        ('SaO2', 'SaO₂, %', '{:.1f}'),
        ('V_rv', 'Объём ПЖ, мл', '{:.0f}'),
    ]

    scenario_keys = ['Здоровый', 'Малый ДМЖП (R=5.0)',
                     'Большой ДМЖП (R=1.0)', 'Эйзенменгер (R=0.7)']

    for metric, label, fmt in steady_metrics:
        row = [label]
        for name in scenario_keys:
            if name in results_dict:
                data, _, _ = results_dict[name]
                if metric in data:
                    mask = steady_mask(data) & np.isfinite(data[metric])
                    values = data[metric][mask]
                    if len(values) > 0:
                        val = np.mean(values)
                        if metric == 'SaO2':
                            val *= 100
                        row.append(fmt.format(val))
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            else:
                row.append('N/A')
        table_data.append(row)

    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.30, 0.175, 0.175, 0.175, 0.175])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    for i in range(1, len(table_data)):
        try:
            table[(i, 0)].set_facecolor('#f0f0f0')
        except KeyError:
            continue

    ax9.set_title('Сводка установившихся значений', fontsize=10, fontweight='bold')

    fig.suptitle('Сравнение гемодинамики: Здоровый человек vs ДМЖП',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hemodynamics_comparison_enhanced.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_shunt_effect_analysis(results_dict):
    """Анализ влияния размера шунта на гемодинамику"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Анализ влияния размера ДМЖП на гемодинамику',
                 fontsize=14, fontweight='bold')

    scenario_keys = ['Здоровый', 'Малый ДМЖП (R=5.0)',
                     'Большой ДМЖП (R=1.0)', 'Эйзенменгер (R=0.7)']

    # 1. Динамика Qp/Qs (полный временной ряд, прореженный)
    ax = axes[0, 0]
    for name, (data, _, _) in results_dict.items():
        if 'Qp_Qs' in data:
            t, qp_qs = subsample(data, 'Qp_Qs', 800)
            mask = np.isfinite(qp_qs)
            if np.any(mask):
                ax.plot(t[mask], qp_qs[mask], color=COLORS.get(name, 'gray'),
                        lw=2, label=name)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Qp/Qs')
    ax.set_title('Динамика Qp/Qs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

    # 2. Баланс давлений — все 4 сценария, локальное усреднение
    ax = axes[0, 1]
    x = np.arange(2)
    width = 0.20
    for i, name in enumerate(scenario_keys):
        if name in results_dict:
            data, _, _ = results_dict[name]
            if 'P_sa' in data and 'P_pa' in data:
                mask = steady_mask(data) & np.isfinite(data['P_sa']) & np.isfinite(data['P_pa'])
                if np.any(mask):
                    ps = np.mean(data['P_sa'][mask])
                    pp = np.mean(data['P_pa'][mask])
                    ax.bar(x[0] + i * width, ps, width,
                           label=f'{name} (P_sa)',
                           color=COLORS.get(name, 'gray'), alpha=0.7)
                    ax.bar(x[1] + i * width, pp, width,
                           color=COLORS.get(name, 'gray'), alpha=0.4, hatch='/')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['Системное', 'Лёгочное'])
    ax.set_ylabel('Давление (мм рт. ст.)')
    ax.set_title('Сравнение давлений')
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Корреляция шунт → Qp/Qs (локальное усреднение)
    ax = axes[0, 2]
    shunt_sizes = []
    qp_qs_values = []
    for name, (data, _, _) in results_dict.items():
        if 'Q_vsd' in data and 'Qp_Qs' in data:
            mask = steady_mask(data) & np.isfinite(data['Q_vsd']) & np.isfinite(data['Qp_Qs'])
            if np.any(mask):
                shunt_mean = np.mean(data['Q_vsd'][mask])
                qp_qs_mean = np.mean(data['Qp_Qs'][mask])
                if abs(shunt_mean) > 1.0:
                    shunt_sizes.append(shunt_mean)
                    qp_qs_values.append(qp_qs_mean)
                    ax.scatter(shunt_mean, qp_qs_mean, s=100,
                               c=COLORS.get(name, 'gray'),
                               marker='o', label=name,
                               edgecolor='black', linewidth=1.5)
    if len(shunt_sizes) > 1:
        try:
            z = np.polyfit(shunt_sizes, qp_qs_values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(shunt_sizes), max(shunt_sizes), 50)
            ax.plot(x_line, p(x_line), 'k--', alpha=0.5,
                    label=f'Тренд: Qp/Qs = {z[0]:.3f}·Q_vsd + {z[1]:.2f}')
        except Exception:
            pass
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Средний шунт VSD (мл/с)')
    ax.set_ylabel('Qp/Qs')
    ax.set_title('Корреляция: размер шунта → Qp/Qs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Объём крови — полный временной ряд
    ax = axes[1, 0]
    for name, (data, _, _) in results_dict.items():
        if 'V_blood' in data:
            t, v_blood = subsample(data, 'V_blood', 800)
            mask = np.isfinite(v_blood)
            if np.any(mask):
                ax.plot(t[mask], v_blood[mask], color=COLORS.get(name, 'gray'),
                        lw=2, label=name)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Объём крови (мл)')
    ax.set_title('Изменение объёма крови')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(4800, 5200)

    # 5. Фракция R→L шунта (полный временной ряд)
    ax = axes[1, 1]
    ymax = 0.0
    for name, (data, _, _) in results_dict.items():
        if 'shunt_fraction_R2L' in data:
            t, sf = subsample(data, 'shunt_fraction_R2L', 800)
            mask = np.isfinite(sf)
            if np.any(mask):
                sf_pct = sf[mask] * 100
                ax.plot(t[mask], sf_pct, color=COLORS.get(name, 'gray'),
                        lw=2, label=name)
                ymax = max(ymax, float(np.max(sf_pct)))
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Доля R→L шунта, %')
    ax.set_title('Фракция право-левого шунта')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=10, color='orange', linestyle=':', alpha=0.5,
               label='Клинически значимый R→L')
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(10.0, ymax * 1.2))

    # 6. Радарная диаграмма — все 4 сценария
    fig.delaxes(axes[1, 2])
    ax_polar = fig.add_subplot(2, 3, 6, projection='polar')
    categories = ['P_sa', 'Q_aortic', 'V_blood', 'GFR', 'SaO2', 'Q_brain']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    healthy_values = {}
    if 'Здоровый' in results_dict:
        healthy_data, _, _ = results_dict['Здоровый']
        mask_h = steady_mask(healthy_data)
        for cat in categories:
            if cat in healthy_data:
                mask = mask_h & np.isfinite(healthy_data[cat])
                if np.any(mask):
                    healthy_values[cat] = np.mean(healthy_data[cat][mask])

    for name, (data, _, _) in results_dict.items():
        values = []
        mask_local = steady_mask(data)
        for cat in categories:
            if cat in data:
                mask = mask_local & np.isfinite(data[cat])
                if np.any(mask):
                    val = np.mean(data[cat][mask])
                    if name != 'Здоровый' and cat in healthy_values and healthy_values[cat] > 0:
                        val = val / healthy_values[cat]
                    values.append(val)
                else:
                    values.append(0)
            else:
                values.append(0)
        values += values[:1]
        ax_polar.plot(angles, values, 'o-', linewidth=2, label=name,
                      color=COLORS.get(name, 'gray'), markersize=6)
        ax_polar.fill(angles, values, alpha=0.1, color=COLORS.get(name, 'gray'))

    ax_polar.set_xticks(angles[:-1])
    ax_polar.set_xticklabels(categories, fontsize=8)
    ax_polar.set_title('Нормированные показатели (здоровый = 1)', fontsize=10)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=7)
    ax_polar.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig('shunt_effect_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def print_detailed_report(results_dict):
    """Детальный отчёт о результатах симуляции"""
    print("\n" + "=" * 100)
    print("ДЕТАЛЬНЫЙ ОТЧЁТ О РЕЗУЛЬТАТАХ СИМУЛЯЦИИ ДМЖП")
    print("=" * 100)

    for name, (data, _, _) in results_dict.items():
        print(f"\n📊 СЦЕНАРИЙ: {name}")
        print("-" * 50)

        # Локальный t_start для каждого сценария
        mask = steady_mask(data)

        def _mean_std(key, fmt="{:.1f}"):
            if key not in data:
                return None, None
            m = mask & np.isfinite(data[key])
            if not np.any(m):
                return None, None
            return np.mean(data[key][m]), np.std(data[key][m])

        ps_mean, ps_std = _mean_std('P_sa')
        if ps_mean is not None:
            print(f"  • Системное АД: {ps_mean:.1f} ± {ps_std:.1f} мм рт. ст.")

        pp_mean, pp_std = _mean_std('P_pa')
        if pp_mean is not None:
            print(f"  • Лёгочное АД: {pp_mean:.1f} ± {pp_std:.1f} мм рт. ст.")

        qs_mean, _ = _mean_std('Q_aortic')
        if qs_mean is not None:
            print(f"  • Системный выброс: {qs_mean:.1f} мл/с")

        qp_mean, _ = _mean_std('Q_pulmonary')
        if qp_mean is not None:
            print(f"  • Лёгочный кровоток: {qp_mean:.1f} мл/с")

        qp_qs_mean, _ = _mean_std('Qp_Qs')
        if qp_qs_mean is not None:
            print(f"  • Соотношение Qp/Qs: {qp_qs_mean:.2f}")

        vsd_mean, _ = _mean_std('Q_vsd')
        if vsd_mean is not None and abs(vsd_mean) > 1.0:
            direction = 'L→R' if vsd_mean > 0 else 'R→L'
            print(f"  • Шунт через ДМЖП: {vsd_mean:+.1f} мл/с ({direction})")

        lv_mean, _ = _mean_std('V_lv')
        if lv_mean is not None:
            print(f"  • Объём ЛЖ: {lv_mean:.1f} мл")

        rv_mean, _ = _mean_std('V_rv')
        if rv_mean is not None:
            print(f"  • Объём ПЖ: {rv_mean:.1f} мл")

        vb_mean, _ = _mean_std('V_blood')
        if vb_mean is not None:
            print(f"  • Объём крови: {vb_mean:.0f} мл")

        brain_flow, _ = _mean_std('Q_brain')
        if brain_flow is not None:
            print(f"  • Мозговой кровоток: {brain_flow:.2f} мл/с")

        o2_cons, _ = _mean_std('O2_consumption')
        if o2_cons is not None:
            print(f"  • Потребление O₂ мозгом: {o2_cons:.2f} у.е./с")

        gfr, _ = _mean_std('GFR')
        if gfr is not None:
            print(f"  • СКФ: {gfr:.2f} мл/с")

        sao2, _ = _mean_std('SaO2')
        if sao2 is not None:
            print(f"  • SaO₂: {sao2 * 100:.1f}%")

        sh, _ = _mean_std('shunt_fraction_R2L')
        if sh is not None and sh > 0.01:
            print(f"  • Доля R→L шунта: {sh * 100:.1f}%")

    # Сравнительный анализ — с локальными t_start для каждого сценария
    print("\n" + "=" * 100)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 100)

    if 'Здоровый' in results_dict and 'Большой ДМЖП (R=1.0)' in results_dict:
        healthy_data, _, _ = results_dict['Здоровый']
        vsd_data, _, _ = results_dict['Большой ДМЖП (R=1.0)']

        mask_h = steady_mask(healthy_data)
        mask_v = steady_mask(vsd_data)

        print("\n📈 Изменения при большом ДМЖП (R=1.0) относительно здорового состояния:")

        def _compare(key, fmt="{:+.0f}", pct=False):
            if key not in healthy_data or key not in vsd_data:
                return
            mh = mask_h & np.isfinite(healthy_data[key])
            mv = mask_v & np.isfinite(vsd_data[key])
            if not (np.any(mh) and np.any(mv)):
                return
            h = np.mean(healthy_data[key][mh])
            v = np.mean(vsd_data[key][mv])
            if pct and h != 0:
                print(f"  • {key}: {h:.2f} → {v:.2f} ({(v / h - 1) * 100:+.0f}%)")
            else:
                print(f"  • {key}: {h:.1f} → {v:.1f} ({v - h:+.1f})")

        _compare('Qp_Qs', pct=True)
        _compare('P_pa')
        _compare('Q_aortic', pct=True)
        _compare('V_rv', pct=True)
        _compare('V_blood', pct=True)


def main():
    """Основная функция"""
    print("=" * 80)
    print("СИМУЛЯЦИЯ И ВИЗУАЛИЗАЦИЯ: СРАВНЕНИЕ ГЕМОДИНАМИКИ ПРИ ДМЖП")
    print("=" * 80)

    scenarios = {
        'Здоровый': {
            'vsd_resistance': np.inf,
            'flow_dependent_lungs': False,
            'color': '#2ecc71',
            'description': 'Здоровое сердце без дефекта'
        },
        'Малый ДМЖП (R=5.0)': {
            'vsd_resistance': 5.0,
            'flow_dependent_lungs': False,
            'color': '#f39c12',
            'description': 'Небольшой дефект, незначительный шунт'
        },
        'Большой ДМЖП (R=1.0)': {
            'vsd_resistance': 1.0,
            'flow_dependent_lungs': True,
            'color': '#e74c3c',
            'description': 'Большой дефект, значительный лево-правый шунт'
        },
        'Эйзенменгер (R=0.7)': {
            'vsd_resistance': 0.7,
            'flow_dependent_lungs': True,
            'pressure_remodel': True,
            'pressure_sensitivity': 0.06,
            'color': '#8e44ad',
            'description': 'Хроническое ремоделирование → R→L шунт',
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
            t_span=params.get('t_span', (0, 600)),
            description=params['description'],
            pressure_remodel=params.get('pressure_remodel', False),
            pressure_sensitivity=params.get('pressure_sensitivity', 0.06),
        )
        results[name] = (data, color, label)

        filename = f"vsd_results_{name.replace(' ', '_').replace('(', '').replace(')', '')}.npz"
        np.savez(filename, **data)
        print(f"   💾 Данные сохранены в {filename}")

    print("\n📊 Генерация визуализаций...")

    print("  • Улучшенное сравнение гемодинамики")
    plot_enhanced_comparison(results)

    print("  • Анализ влияния размера шунта")
    plot_shunt_effect_analysis(results)

    print_detailed_report(results)

    print("\n✅ Работа завершена!")
    print("📁 Созданные файлы:")
    print("   - hemodynamics_comparison_enhanced.png")
    print("   - shunt_effect_analysis.png")
    print("   - vsd_results_*.npz (файлы данных)")


if __name__ == "__main__":
    main()