#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_vsd_comparison.py
Расширенная визуализация для сравнения гемодинамики при ДМЖП.
Включает: временные ряды, фазовые портреты, статистический анализ.

Загружает .npz, сгенерированные run_simulation.py. Ожидает колонки:
    t, P_sa, P_pa, P_sv, P_pv, Q_aortic, Q_pulmonary, Q_vsd, Qp_Qs,
    V_lv, V_rv, V_blood, GFR, Q_brain, O2_consumption,
    SaO2, shunt_fraction_R2L, P_a_O2, P_v_O2, C_a_O2, C_v_O2,
    P_lv, P_rv, P_la, P_ra  (опционально, если heart.py экспортирует)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from scipy.stats import linregress
from scipy.signal import savgol_filter
import glob
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Стиль и палитра
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9

COLORS = {
    'Здоровый': '#2ecc71',
    'Малый ДМЖП (R=5.0)': '#f39c12',
    'Большой ДМЖП (R=1.0)': '#e74c3c',
    'Эйзенменгер (R=0.7)': '#8e44ad',
}

SCENARIO_ORDER = [
    'Здоровый',
    'Малый ДМЖП (R=5.0)',
    'Большой ДМЖП (R=1.0)',
    'Эйзенменгер (R=0.7)',
]

FILE_NAME_MAPPING = {
    'Здоровый': 'Здоровый',
    'Малый_ДМЖП_R=5.0': 'Малый ДМЖП (R=5.0)',
    'Малый_ДМЖП': 'Малый ДМЖП (R=5.0)',
    'Большой_ДМЖП_R=1.0': 'Большой ДМЖП (R=1.0)',
    'Большой_ДМЖП': 'Большой ДМЖП (R=1.0)',
    'Эйзенменгер_R=0.7': 'Эйзенменгер (R=0.7)',
    'Эйзенменгер': 'Эйзенменгер (R=0.7)',
}


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------
def safe_savgol_filter(data, window_length, polyorder):
    """Безопасная версия savgol_filter, обрабатывающая ошибки."""
    data_clean = np.where(np.isfinite(data), data, np.nan)

    if np.any(np.isnan(data_clean)):
        nan_mask = np.isnan(data_clean)
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) > 1:
            data_clean[nan_mask] = np.interp(
                np.flatnonzero(nan_mask), valid_indices, data_clean[valid_indices]
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


def steady_mask(data, frac=0.8):
    """Маска последних (1-frac) симуляции — локально для каждого сценария."""
    t_start = frac * data['t'][-1]
    return data['t'] >= t_start


def subsample(data, key, n_target=5000):
    """Прореживает временной ряд data[key] до ~n_target точек."""
    step = max(1, len(data['t']) // n_target)
    return data['t'][::step], data[key][::step]


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------
def load_all_results():
    """Загрузка всех результатов симуляции из vsd_results_*.npz."""
    results = {}
    files = glob.glob("vsd_results_*.npz")

    if not files:
        print("❌ Файлы с результатами не найдены!")
        print("   Убедитесь, что сначала запущен run_simulation.py")
        return None

    print(f"Найдено файлов: {len(files)}")

    for filepath in files:
        filename = filepath.replace('vsd_results_', '').replace('.npz', '')

        if filename in FILE_NAME_MAPPING:
            display_name = FILE_NAME_MAPPING[filename]
        elif 'Здоровый' in filename:
            display_name = 'Здоровый'
        elif 'Малый' in filename:
            display_name = 'Малый ДМЖП (R=5.0)'
        elif 'Большой' in filename:
            display_name = 'Большой ДМЖП (R=1.0)'
        elif 'Эйзенменгер' in filename:
            display_name = 'Эйзенменгер (R=0.7)'
        else:
            display_name = filename

        try:
            data = np.load(filepath, allow_pickle=True)
            results[display_name] = {key: data[key] for key in data.files}
            print(f"✓ Загружен: {filename} → {display_name}")
        except Exception as e:
            print(f"✗ Ошибка загрузки {filepath}: {e}")

    if not results:
        print("❌ Не удалось загрузить ни одного файла!")
        return None

    # Упорядочим по SCENARIO_ORDER
    ordered = {}
    for key in SCENARIO_ORDER:
        if key in results:
            ordered[key] = results[key]
    for key in results:
        if key not in ordered:
            ordered[key] = results[key]

    print(f"\nЗагружено сценариев: {list(ordered.keys())}")
    return ordered


# ---------------------------------------------------------------------------
# Рис. 1: Временные ряды
# ---------------------------------------------------------------------------
def plot_hemodynamic_timeseries(results):
    """Рис. 1: Временные ряды основных гемодинамических показателей."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Гемодинамика при ДМЖП: временные ряды', fontsize=14, fontweight='bold')

    metrics = [
        ('P_sa', 'Системное АД (мм рт. ст.)', axes[0, 0]),
        ('P_pa', 'Лёгочное АД (мм рт. ст.)', axes[0, 1]),
        ('Q_aortic', 'Системный выброс (мл/с)', axes[0, 2]),
        ('Q_pulmonary', 'Лёгочный кровоток (мл/с)', axes[1, 0]),
        ('Qp_Qs', 'Соотношение Qp/Qs', axes[1, 1]),
        ('Q_vsd', 'Шунт VSD (мл/с)', axes[1, 2]),
        ('V_lv', 'Объём ЛЖ (мл)', axes[2, 0]),
        ('V_rv', 'Объём ПЖ (мл)', axes[2, 1]),
        ('SaO2', 'SaO₂ (%)', axes[2, 2]),
    ]

    for metric, ylabel, ax in metrics:
        for scenario, data in results.items():
            if metric not in data or len(data['t']) == 0:
                continue
            t, values = subsample(data, metric, 5000)
            if metric == 'SaO2':
                values = values * 100
            mask = np.isfinite(values)
            if np.any(mask):
                ax.plot(t[mask], values[mask],
                        color=COLORS.get(scenario, 'gray'),
                        lw=1.5, label=scenario if metric == metrics[0][0] else '')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Время (с)')
        ax.grid(True, alpha=0.3)
        if metric == 'SaO2':
            ax.axhline(y=90, color='orange', linestyle=':', alpha=0.5)
        if metric == metrics[0][0]:
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('fig1_hemodynamics_timeseries.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Рис. 2: Фазовые портреты (реальные P_lv/P_rv, если есть; иначе — пропуск)
# ---------------------------------------------------------------------------
def plot_phase_portraits(results):
    """Рис. 2: Фазовые портреты желудочков на реальных P_lv/P_rv."""
    have_real = any(
        ('P_lv' in d and 'P_rv' in d) for d in results.values()
    )

    if not have_real:
        print("  ⚠ P_lv/P_rv отсутствуют в .npz — PV-петли пропущены.")
        print("    Добавьте их в heart._current_flows и whole_body.compute_outputs.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Фазовые портреты желудочков', fontsize=14, fontweight='bold')

    for scenario, data in results.items():
        if not ('V_lv' in data and 'P_lv' in data and
                'V_rv' in data and 'P_rv' in data):
            continue

        # Последние ~10 кардиоциклов = окно для чистой петли
        t = data['t']
        n_show = min(len(t), 1500)
        V_lv = data['V_lv'][-n_show:]
        P_lv = data['P_lv'][-n_show:]
        V_rv = data['V_rv'][-n_show:]
        P_rv = data['P_rv'][-n_show:]

        mask_lv = np.isfinite(V_lv) & np.isfinite(P_lv)
        mask_rv = np.isfinite(V_rv) & np.isfinite(P_rv)

        if np.any(mask_lv):
            axes[0].plot(V_lv[mask_lv], P_lv[mask_lv],
                         color=COLORS.get(scenario, 'gray'),
                         lw=1.5, alpha=0.8, label=scenario)
        if np.any(mask_rv):
            axes[1].plot(V_rv[mask_rv], P_rv[mask_rv],
                         color=COLORS.get(scenario, 'gray'),
                         lw=1.5, alpha=0.8, label=scenario)

    axes[0].set_xlabel('Объём ЛЖ (мл)')
    axes[0].set_ylabel('Давление ЛЖ (мм рт. ст.)')
    axes[0].set_title('Левый желудочек')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Объём ПЖ (мл)')
    axes[1].set_ylabel('Давление ПЖ (мм рт. ст.)')
    axes[1].set_title('Правый желудочек')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig2_phase_portraits.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Рис. 3: Столбчатое сравнение (8 панелей: 7 метрик + SaO2)
# ---------------------------------------------------------------------------
def plot_bar_comparison(results):
    """Рис. 3: Сравнение установившихся значений."""
    metrics = [
        ('P_sa', 'Системное АД (мм рт. ст.)'),
        ('P_pa', 'Лёгочное АД (мм рт. ст.)'),
        ('Q_aortic', 'Системный выброс (мл/с)'),
        ('Qp_Qs', 'Qp/Qs'),
        ('V_rv', 'Объём ПЖ (мл)'),
        ('V_blood', 'Объём крови (мл)'),
        ('GFR', 'СКФ (мл/с)'),
        ('SaO2', 'SaO₂ (%)'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Сравнение установившихся показателей', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    scenarios = list(results.keys())

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
        means, stds = [], []

        for scenario in scenarios:
            data = results[scenario]
            if metric in data:
                mask = steady_mask(data) & np.isfinite(data[metric])
                values = data[metric][mask]
                if len(values) > 0:
                    val = np.mean(values) * (100 if metric == 'SaO2' else 1)
                    sd = np.std(values) * (100 if metric == 'SaO2' else 1)
                    means.append(val)
                    stds.append(sd)
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(scenarios, means,
                      color=[COLORS.get(s, 'gray') for s in scenarios],
                      alpha=0.7, edgecolor='black', linewidth=1)
        ax.errorbar(scenarios, means, yerr=stds, fmt='none',
                    ecolor='black', capsize=5, capthick=1)
        ax.set_ylabel(label)
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(True, alpha=0.3, axis='y')

        ymax = max([abs(m) for m in means] + [1e-6])
        for bar, mean in zip(bars, means):
            if mean != 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ymax * 0.02,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('fig3_bar_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Рис. 4: Детальный анализ
# ---------------------------------------------------------------------------
def plot_cardiovascular_parameters(results):
    """Рис. 4: Детальный анализ сердечно-сосудистых параметров."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Детальный анализ сердечно-сосудистых параметров',
                 fontsize=14, fontweight='bold')

    scenarios = list(results.keys())

    # 1. Qp/Qs во времени
    ax = axes[0, 0]
    for scenario, data in results.items():
        if 'Qp_Qs' not in data:
            continue
        t, qp_qs = subsample(data, 'Qp_Qs', 4000)
        mask = np.isfinite(qp_qs)
        if np.any(mask):
            qp_qs_clean = qp_qs[mask]
            if len(qp_qs_clean) > 100:
                qp_qs_clean = safe_savgol_filter(qp_qs_clean, 101, 3)
            ax.plot(t[mask], qp_qs_clean,
                    color=COLORS.get(scenario, 'gray'), lw=1.5, label=scenario)
    ax.set_ylabel('Qp/Qs')
    ax.set_xlabel('Время (с)')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Норма (1.0)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Соотношение лёгочного и системного кровотока')

    # 2. Объём ЛЖ
    ax = axes[0, 1]
    for scenario, data in results.items():
        if 'V_lv' not in data:
            continue
        mask = steady_mask(data) & np.isfinite(data['V_lv'])
        if np.any(mask):
            mean_lv = np.mean(data['V_lv'][mask])
            ax.axhline(y=mean_lv, color=COLORS.get(scenario, 'gray'),
                       linestyle='--', alpha=0.7, label=f'{scenario}: {mean_lv:.0f} мл')
        t, v_lv = subsample(data, 'V_lv', 4000)
        m = np.isfinite(v_lv)
        if np.any(m):
            ax.plot(t[m], v_lv[m], color=COLORS.get(scenario, 'gray'),
                    lw=0.8, alpha=0.4)
    ax.set_ylabel('Объём ЛЖ (мл)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Объём левого желудочка')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Объём ПЖ
    ax = axes[0, 2]
    for scenario, data in results.items():
        if 'V_rv' not in data:
            continue
        mask = steady_mask(data) & np.isfinite(data['V_rv'])
        if np.any(mask):
            mean_rv = np.mean(data['V_rv'][mask])
            ax.axhline(y=mean_rv, color=COLORS.get(scenario, 'gray'),
                       linestyle='--', alpha=0.7, label=f'{scenario}: {mean_rv:.0f} мл')
        t, v_rv = subsample(data, 'V_rv', 4000)
        m = np.isfinite(v_rv)
        if np.any(m):
            ax.plot(t[m], v_rv[m], color=COLORS.get(scenario, 'gray'),
                    lw=0.8, alpha=0.4)
    ax.set_ylabel('Объём ПЖ (мл)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Объём правого желудочка')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4. Корреляция Qp/Qs и Q_vsd
    ax = axes[1, 0]
    for scenario, data in results.items():
        if not ('Qp_Qs' in data and 'Q_vsd' in data):
            continue
        mask = steady_mask(data) & np.isfinite(data['Qp_Qs']) & np.isfinite(data['Q_vsd'])
        if np.any(mask):
            qp_qs = data['Qp_Qs'][mask]
            q_vsd = data['Q_vsd'][mask]
            step = max(1, len(q_vsd) // 500)
            ax.scatter(q_vsd[::step], qp_qs[::step],
                       c=COLORS.get(scenario, 'gray'), s=10, alpha=0.5, label=scenario)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Шунт VSD (мл/с)')
    ax.set_ylabel('Qp/Qs')
    ax.set_title('Корреляция: шунт ↔ Qp/Qs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Сравнение объёмов желудочков
    ax = axes[1, 1]
    chambers = ['V_lv', 'V_rv']
    chamber_names = ['Левый желудочек', 'Правый желудочек']
    x = np.arange(len(chambers))
    width = 0.20

    for i, scenario in enumerate(scenarios):
        means = []
        for chamber in chambers:
            if chamber in results[scenario]:
                mask = steady_mask(results[scenario]) & np.isfinite(results[scenario][chamber])
                values = results[scenario][chamber][mask]
                means.append(np.mean(values) if len(values) > 0 else 0)
            else:
                means.append(0)
        offset = (i - 1.5) * width
        ax.bar(x + offset, means, width, label=scenario,
               color=COLORS.get(scenario, 'gray'), alpha=0.7)
    ax.set_ylabel('Объём (мл)')
    ax.set_xticks(x)
    ax.set_xticklabels(chamber_names)
    ax.set_title('Сравнение объёмов желудочков')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Региональное распределение кровотока
    ax = axes[1, 2]
    organs = ['Мозг', 'Почки', 'ЖКТ', 'Печень']
    organ_keys = ['Q_brain', 'Q_renal', 'Q_gitract_out', 'Q_liver_out']

    for scenario, data in results.items():
        flows = []
        for key in organ_keys:
            if key in data:
                mask = steady_mask(data) & np.isfinite(data[key])
                values = data[key][mask]
                flows.append(np.mean(values) if len(values) > 0 else 0)
            else:
                flows.append(0)
        ax.plot(organs, flows, 'o-', color=COLORS.get(scenario, 'gray'),
                lw=2, markersize=8, label=scenario)
    ax.set_ylabel('Кровоток (мл/с)')
    ax.set_title('Региональное распределение кровотока')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig4_detailed_cardiac.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Рис. 5: Дашборд
# ---------------------------------------------------------------------------
def plot_comprehensive_dashboard(results):
    """Комплексная дашборд-панель."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)

    scenarios = list(results.keys())

    # 1. Qp/Qs
    ax1 = fig.add_subplot(gs[0, :2])
    for scenario, data in results.items():
        if 'Qp_Qs' not in data:
            continue
        t, qp_qs = subsample(data, 'Qp_Qs', 4000)
        mask = np.isfinite(qp_qs)
        if np.any(mask):
            qp_qs_clean = qp_qs[mask]
            if len(qp_qs_clean) > 100:
                qp_qs_clean = safe_savgol_filter(qp_qs_clean, 101, 3)
            ax1.plot(t[mask], qp_qs_clean,
                     color=COLORS.get(scenario, 'gray'), lw=2, label=scenario)
    ax1.set_ylabel('Qp/Qs')
    ax1.set_xlabel('Время (с)')
    ax1.set_title('Соотношение лёгочного и системного кровотока',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax1.axhline(y=1.5, color='orange', linestyle=':', alpha=0.5)
    ax1.axhline(y=2.0, color='red', linestyle=':', alpha=0.5)

    # 2. Давления
    ax2 = fig.add_subplot(gs[0, 2])
    metrics = ['P_sa', 'P_pa']
    x = np.arange(len(metrics))
    width = 0.20

    for i, scenario in enumerate(scenarios):
        data = results[scenario]
        means, stds = [], []
        for metric in metrics:
            if metric in data:
                mask = steady_mask(data) & np.isfinite(data[metric])
                values = data[metric][mask]
                means.append(np.mean(values) if len(values) > 0 else 0)
                stds.append(np.std(values) if len(values) > 0 else 0)
            else:
                means.append(0)
                stds.append(0)
        offset = (i - 1.5) * width
        ax2.bar(x + offset, means, width, label=scenario,
                color=COLORS.get(scenario, 'gray'), alpha=0.7)
        ax2.errorbar(x + offset, means, yerr=stds, fmt='none',
                     ecolor='black', capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Сист. АД', 'Лёг. АД'], fontsize=9)
    ax2.set_ylabel('Давление (мм рт. ст.)')
    ax2.set_title('Сравнение давлений')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. SaO2 (новая панель в gs[0, 3])
    ax3 = fig.add_subplot(gs[0, 3])
    for scenario, data in results.items():
        if 'SaO2' not in data:
            continue
        t, sao2 = subsample(data, 'SaO2', 4000)
        mask = np.isfinite(sao2)
        if np.any(mask):
            ax3.plot(t[mask], sao2[mask] * 100,
                     color=COLORS.get(scenario, 'gray'), lw=1.5, label=scenario)
    ax3.axhline(y=90, color='orange', linestyle=':', alpha=0.5)
    ax3.set_ylabel('SaO₂ (%)')
    ax3.set_xlabel('Время (с)')
    ax3.set_title('Артериальная сатурация O₂')
    ax3.legend(fontsize=7, loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(60, 100)

    # 4. Объёмы желудочков
    ax4 = fig.add_subplot(gs[1, 0])
    chambers = ['ЛЖ', 'ПЖ']
    x = np.arange(len(chambers))
    width = 0.20

    for i, scenario in enumerate(scenarios):
        data = results[scenario]
        volumes = []
        for chamber in ['V_lv', 'V_rv']:
            if chamber in data:
                mask = steady_mask(data) & np.isfinite(data[chamber])
                values = data[chamber][mask]
                volumes.append(np.mean(values) if len(values) > 0 else 0)
            else:
                volumes.append(0)
        offset = (i - 1.5) * width
        ax4.bar(x + offset, volumes, width, label=scenario,
                color=COLORS.get(scenario, 'gray'), alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(chambers)
    ax4.set_ylabel('Объём (мл)')
    ax4.set_title('Сравнение объёмов желудочков')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Региональные кровотоки
    ax5 = fig.add_subplot(gs[1, 1])
    organs = ['Мозг', 'Почки', 'Печень', 'ЖКТ']
    organ_keys = ['Q_brain', 'Q_renal', 'Q_liver_out', 'Q_gitract_out']

    for scenario, data in results.items():
        flows = []
        for key in organ_keys:
            if key in data:
                mask = steady_mask(data) & np.isfinite(data[key])
                values = data[key][mask]
                flows.append(np.mean(values) if len(values) > 0 else 0)
            else:
                flows.append(0)
        ax5.plot(organs, flows, 'o-', lw=2, markersize=8,
                 label=scenario, color=COLORS.get(scenario, 'gray'))
    ax5.set_ylabel('Кровоток (мл/с)')
    ax5.set_title('Региональное распределение кровотока')
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    # 6. Объём крови
    ax6 = fig.add_subplot(gs[1, 2])
    for scenario, data in results.items():
        if 'V_blood' not in data:
            continue
        t, v = subsample(data, 'V_blood', 4000)
        mask = np.isfinite(v)
        if np.any(mask):
            ax6.plot(t[mask], v[mask], color=COLORS.get(scenario, 'gray'),
                     lw=1.5, label=scenario)
    ax6.set_ylabel('Объём крови (мл)')
    ax6.set_xlabel('Время (с)')
    ax6.set_title('Объём циркулирующей крови')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)

    # 7. Фракция R→L шунта (новая панель в gs[1, 3])
    ax7 = fig.add_subplot(gs[1, 3])
    for scenario, data in results.items():
        if 'shunt_fraction_R2L' not in data:
            continue
        t, sf = subsample(data, 'shunt_fraction_R2L', 4000)
        mask = np.isfinite(sf)
        if np.any(mask):
            ax7.plot(t[mask], sf[mask] * 100,
                     color=COLORS.get(scenario, 'gray'), lw=1.5, label=scenario)
    ax7.set_ylabel('R→L шунт, %')
    ax7.set_xlabel('Время (с)')
    ax7.set_title('Фракция право-левого шунта')
    ax7.axhline(y=10, color='orange', linestyle=':', alpha=0.5)
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3)

    # 8. Qp/Qs vs Q_vsd (корреляция)
    ax8 = fig.add_subplot(gs[2, 0])
    shunt_means, qp_qs_means = [], []

    for scenario, data in results.items():
        if not ('Q_vsd' in data and 'Qp_Qs' in data):
            continue
        mask = steady_mask(data) & np.isfinite(data['Q_vsd']) & np.isfinite(data['Qp_Qs'])
        if np.any(mask):
            shunt_mean = np.mean(data['Q_vsd'][mask])
            qp_qs_mean = np.mean(data['Qp_Qs'][mask])
            if abs(shunt_mean) > 1.0:
                shunt_means.append(shunt_mean)
                qp_qs_means.append(qp_qs_mean)
                ax8.scatter(shunt_mean, qp_qs_mean, s=120,
                            c=COLORS.get(scenario, 'gray'), marker='o',
                            edgecolor='black', linewidth=1.5, label=scenario)

    if len(shunt_means) > 1:
        try:
            slope, intercept, r_value, p_value, _ = linregress(shunt_means, qp_qs_means)
            x_line = np.linspace(min(shunt_means), max(shunt_means), 50)
            ax8.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5,
                     label=f'R² = {r_value ** 2:.3f}')
        except Exception:
            pass
    ax8.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Шунт VSD (мл/с)')
    ax8.set_ylabel('Qp/Qs')
    ax8.set_title('Корреляция: размер шунта → Qp/Qs')
    ax8.legend(fontsize=7)
    ax8.grid(True, alpha=0.3)

    # 9. SaO2 vs Qp/Qs (диагностика Эйзенменгера)
    ax9 = fig.add_subplot(gs[2, 1])
    for scenario, data in results.items():
        if not ('SaO2' in data and 'Qp_Qs' in data):
            continue
        mask = steady_mask(data) & np.isfinite(data['SaO2']) & np.isfinite(data['Qp_Qs'])
        if np.any(mask):
            step = max(1, np.sum(mask) // 500)
            ax9.scatter(data['Qp_Qs'][mask][::step],
                        data['SaO2'][mask][::step] * 100,
                        c=COLORS.get(scenario, 'gray'),
                        s=10, alpha=0.5, label=scenario)
    ax9.axhline(y=90, color='orange', linestyle=':', alpha=0.5)
    ax9.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Qp/Qs')
    ax9.set_ylabel('SaO₂ (%)')
    ax9.set_title('SaO₂ vs Qp/Qs (Эйзенменгер)')
    ax9.legend(fontsize=7)
    ax9.grid(True, alpha=0.3)

    # 10. P_aO2 (новая панель в gs[2, 2])
    ax10 = fig.add_subplot(gs[2, 2])
    for scenario, data in results.items():
        if 'P_a_O2' not in data:
            continue
        t, pa = subsample(data, 'P_a_O2', 4000)
        mask = np.isfinite(pa)
        if np.any(mask):
            ax10.plot(t[mask], pa[mask], color=COLORS.get(scenario, 'gray'),
                      lw=1.5, label=scenario)
    ax10.set_ylabel('P_aO₂ (мм рт. ст.)')
    ax10.set_xlabel('Время (с)')
    ax10.set_title('Парциальное давление O₂ в артерии')
    ax10.legend(fontsize=7)
    ax10.grid(True, alpha=0.3)

    # 11. GFR (новая панель в gs[2, 3])
    ax11 = fig.add_subplot(gs[2, 3])
    for scenario, data in results.items():
        if 'GFR' not in data:
            continue
        t, gfr = subsample(data, 'GFR', 4000)
        mask = np.isfinite(gfr)
        if np.any(mask):
            ax11.plot(t[mask], gfr[mask], color=COLORS.get(scenario, 'gray'),
                      lw=1.5, label=scenario)
    ax11.set_ylabel('СКФ (мл/с)')
    ax11.set_xlabel('Время (с)')
    ax11.set_title('Функция почек')
    ax11.legend(fontsize=7)
    ax11.grid(True, alpha=0.3)

    # 12. Статистическая таблица (все 4 сценария)
    ax12 = fig.add_subplot(gs[3, :])
    ax12.axis('tight')
    ax12.axis('off')

    header = ['Показатель'] + [s[:16] for s in scenarios]
    table_data = [header]

    summary_metrics = [
        ('Qp_Qs', 'Qp/Qs', '{:.2f} ± {:.2f}', 1),
        ('SaO2', 'SaO₂, %', '{:.1f} ± {:.2f}', 100),
        ('P_sa', 'АД сист., мм рт. ст.', '{:.0f} ± {:.0f}', 1),
        ('P_pa', 'АД лёг., мм рт. ст.', '{:.0f} ± {:.0f}', 1),
        ('V_lv', 'Объём ЛЖ, мл', '{:.0f} ± {:.0f}', 1),
        ('V_rv', 'Объём ПЖ, мл', '{:.0f} ± {:.0f}', 1),
        ('Q_aortic', 'Сист. выброс, мл/с', '{:.1f} ± {:.1f}', 1),
        ('shunt_fraction_R2L', 'R→L шунт, %', '{:.1f} ± {:.2f}', 100),
        ('V_blood', 'Объём крови, мл', '{:.0f} ± {:.0f}', 1),
        ('GFR', 'СКФ, мл/с', '{:.2f} ± {:.2f}', 1),
    ]

    for metric, label, fmt, scale in summary_metrics:
        row = [label]
        for scenario in scenarios:
            data = results[scenario]
            if metric in data:
                mask = steady_mask(data) & np.isfinite(data[metric])
                values = data[metric][mask]
                if len(values) > 0:
                    row.append(fmt.format(np.mean(values) * scale, np.std(values) * scale))
                else:
                    row.append('N/A')
            else:
                row.append('N/A')
        table_data.append(row)

    n_cols = len(scenarios) + 1
    col_width = 1.0 / n_cols
    table = ax12.table(cellText=table_data, cellLoc='center', loc='center',
                       colWidths=[0.26] + [0.74 / len(scenarios)] * len(scenarios))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    # Цветовая кодировка: столбец 0 — серый
    for i in range(len(table_data)):
        try:
            table[(i, 0)].set_facecolor('#f0f0f0')
        except KeyError:
            continue

    ax12.set_title('Сводка установившихся значений (t > 0.8·t_end)',
                   fontsize=10, fontweight='bold')

    fig.suptitle('📊 ДАШБОРД: Сравнение гемодинамики при ДМЖП\n'
                 'Здоровый человек vs пациенты с дефектом межжелудочковой перегородки',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Рис. 6: Схематическое сравнение (статическая, три варианта)
# ---------------------------------------------------------------------------
def plot_schematic_heart_comparison():
    """Схематическое сравнение: здоровое / L→R ДМЖП / Эйзенменгер (R→L)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle('Схематическое сравнение гемодинамики',
                 fontsize=14, fontweight='bold')

    def _draw_heart(ax, title, title_color,
                    lv_radius=2.0, rv_radius=2.0,
                    shunt_direction=None,
                    qs_label='Qs = Qp', qs_color='green'):
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold', color=title_color)

        lv = Circle((4, 6), lv_radius, facecolor='#e74c3c',
                    edgecolor='black', alpha=0.7)
        rv = Circle((8, 6), rv_radius, facecolor='#3498db',
                    edgecolor='black', alpha=0.7)
        la = Circle((3.5, 9), 1.3, facecolor='#e74c3c',
                    edgecolor='black', alpha=0.6)
        ra = Circle((8.5, 9), 1.3, facecolor='#3498db',
                    edgecolor='black', alpha=0.6)
        for p in (lv, rv, la, ra):
            ax.add_patch(p)

        # Сосуды
        ax.plot([4, 4], [6, 3], 'r-', lw=3)
        ax.plot([8, 8], [6, 3], 'b-', lw=3)
        ax.plot([4, 4], [9, 10.3], 'r-', lw=3)
        ax.plot([8, 8], [9, 10.3], 'b-', lw=3)

        # Стрелки
        ax.annotate('', xy=(4, 3), xytext=(4, 4),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        ax.annotate('', xy=(8, 3), xytext=(8, 4),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        ax.annotate('', xy=(4, 10.3), xytext=(4, 9),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        ax.annotate('', xy=(8, 10.3), xytext=(8, 9),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

        # Шунт
        if shunt_direction == 'L2R':
            ax.annotate('', xy=(6.1, 6), xytext=(5.8, 5.2),
                        arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
        elif shunt_direction == 'R2L':
            ax.annotate('', xy=(5.8, 5.2), xytext=(6.1, 6),
                        arrowprops=dict(arrowstyle='->', lw=2, color='purple'))

        # Подписи
        ax.text(4, 1.5, 'Аорта', ha='center', fontsize=10, fontweight='bold')
        ax.text(8, 1.5, 'Лёгочная артерия', ha='center', fontsize=10, fontweight='bold')
        ax.text(2, 6, 'ЛЖ', ha='center', fontsize=10, fontweight='bold')
        ax.text(10, 6, 'ПЖ', ha='center', fontsize=10, fontweight='bold')
        ax.text(2.5, 9, 'ЛП', ha='center', fontsize=9)
        ax.text(9.5, 9, 'ПП', ha='center', fontsize=9)
        ax.text(6, 11, qs_label, ha='center', fontsize=11,
                fontweight='bold', color=qs_color)

    # Здоровое
    _draw_heart(axes[0], 'Здоровое сердце', 'green')

    # L→R ДМЖП
    _draw_heart(axes[1], 'ДМЖП: лево-правый шунт (L→R)', 'red',
                lv_radius=2.2, rv_radius=2.5,
                shunt_direction='L2R',
                qs_label='Qs < Qp  (L→R)', qs_color='red')

    # Эйзенменгер (R→L)
    _draw_heart(axes[2], 'Эйзенменгер: право-левый шунт (R→L)', 'purple',
                lv_radius=2.3, rv_radius=2.6,
                shunt_direction='R2L',
                qs_label='Qs > Qp  (R→L, цианоз)', qs_color='purple')

    plt.tight_layout()
    plt.savefig('schematic_heart_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Статистическая сводка в консоль
# ---------------------------------------------------------------------------
def print_statistical_summary(results):
    """Вывод статистической сводки в консоль."""
    scenarios = list(results.keys())

    print("\n" + "=" * 100)
    print("СТАТИСТИЧЕСКАЯ СВОДКА (установившийся режим, t > 0.8·t_end)")
    print("=" * 100)

    metrics = [
        ('P_sa', 'Системное АД', 'мм рт. ст.', '{:.1f} ± {:.1f}', 1),
        ('P_pa', 'Лёгочное АД', 'мм рт. ст.', '{:.1f} ± {:.1f}', 1),
        ('Q_aortic', 'Системный выброс', 'мл/с', '{:.1f} ± {:.1f}', 1),
        ('Q_pulmonary', 'Лёгочный кровоток', 'мл/с', '{:.1f} ± {:.1f}', 1),
        ('Qp_Qs', 'Qp/Qs', '', '{:.2f} ± {:.2f}', 1),
        ('Q_vsd', 'Шунт VSD', 'мл/с', '{:+.1f} ± {:.1f}', 1),
        ('SaO2', 'SaO₂', '%', '{:.1f} ± {:.2f}', 100),
        ('shunt_fraction_R2L', 'R→L шунт', '%', '{:.1f} ± {:.2f}', 100),
        ('V_lv', 'Объём ЛЖ', 'мл', '{:.1f} ± {:.1f}', 1),
        ('V_rv', 'Объём ПЖ', 'мл', '{:.1f} ± {:.1f}', 1),
        ('V_blood', 'Объём крови', 'мл', '{:.0f} ± {:.0f}', 1),
        ('GFR', 'СКФ', 'мл/с', '{:.2f} ± {:.2f}', 1),
        ('Q_brain', 'Мозговой кровоток', 'мл/с', '{:.2f} ± {:.2f}', 1),
        ('O2_consumption', 'Потребление O₂ мозгом', 'у.е./с', '{:.2f} ± {:.2f}', 1),
    ]

    header = f"{'Показатель':<32}"
    for s in scenarios:
        header += f"{s[:20]:>22}"
    print(header)
    print("-" * len(header))

    for metric, label, unit, fmt, scale in metrics:
        line = f"{label + ' [' + unit + ']':<32}"
        for scenario in scenarios:
            data = results[scenario]
            if metric in data:
                mask = steady_mask(data) & np.isfinite(data[metric])
                values = data[metric][mask]
                if len(values) > 0:
                    m = np.mean(values) * scale
                    sd = np.std(values) * scale
                    line += f"{fmt.format(m, sd):>22}"
                else:
                    line += f"{'N/A':>22}"
            else:
                line += f"{'N/A':>22}"
        print(line)

    print("=" * 100)

    # Ключевые находки
    print("\n🔍 КЛЮЧЕВЫЕ НАХОДКИ:")

    for scenario in scenarios:
        data = results[scenario]
        if 'SaO2' not in data or 'Qp_Qs' not in data:
            continue
        mask = steady_mask(data) & np.isfinite(data['SaO2']) & np.isfinite(data['Qp_Qs'])
        if not np.any(mask):
            continue
        sao2 = np.mean(data['SaO2'][mask]) * 100
        qp_qs = np.mean(data['Qp_Qs'][mask])
        ppa = np.mean(data['P_pa'][mask]) if 'P_pa' in data else np.nan
        r2l = (np.mean(data['shunt_fraction_R2L'][mask]) * 100
               if 'shunt_fraction_R2L' in data else 0.0)

        tag = []
        if sao2 < 90:
            tag.append('гипоксемия')
        if qp_qs > 2.0:
            tag.append('большой шунт')
        if qp_qs < 1.0:
            tag.append('R→L')
        if ppa > 40:
            tag.append('тяжёлая ЛГ')
        note = f"  [{', '.join(tag)}]" if tag else ""

        print(f"  • {scenario:<24}: Qp/Qs={qp_qs:.2f}, "
              f"SaO₂={sao2:.1f}%, R→L={r2l:.1f}%{note}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ ГЕМОДИНАМИКИ: ЗДОРОВЫЙ vs ДМЖП")
    print("=" * 70)

    results = load_all_results()
    if results is None or len(results) == 0:
        print("\n⚠ Не удалось загрузить данные!")
        print("   Сначала запустите run_simulation.py")
        return

    print("\n📊 Генерация графиков...")

    print("  • Рис. 1: Временные ряды")
    plot_hemodynamic_timeseries(results)

    print("  • Рис. 2: Фазовые портреты")
    plot_phase_portraits(results)

    print("  • Рис. 3: Столбчатые диаграммы")
    plot_bar_comparison(results)

    print("  • Рис. 4: Детальный анализ")
    plot_cardiovascular_parameters(results)

    print("  • Рис. 5: Комплексная дашборд-панель")
    plot_comprehensive_dashboard(results)

    print("  • Рис. 6: Схематическая диаграмма")
    plot_schematic_heart_comparison()

    print_statistical_summary(results)

    print("\n✅ Все графики сохранены в текущей директории!")
    print("📁 Файлы:")
    print("   fig1_hemodynamics_timeseries.png")
    print("   fig2_phase_portraits.png")
    print("   fig3_bar_comparison.png")
    print("   fig4_detailed_cardiac.png")
    print("   comprehensive_dashboard.png")
    print("   schematic_heart_comparison.png")


if __name__ == "__main__":
    main()