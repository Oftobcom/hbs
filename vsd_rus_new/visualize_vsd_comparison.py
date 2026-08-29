#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_vsd_comparison.py
Расширенная визуализация для сравнения гемодинамики при ДМЖП.
Включает: временные ряды, фазовые портреты, статистический анализ.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from scipy import signal
from scipy.stats import linregress
from scipy.signal import savgol_filter
import glob
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9

# Цветовая схема (единая для всех функций)
COLORS = {
    'Здоровый': '#2ecc71',
    'Малый_ДМЖП': '#f39c12',
    'Малый ДМЖП (R=5.0)': '#f39c12',
    'Большой_ДМЖП': '#e74c3c',
    'Большой ДМЖП (R=1.0)': '#e74c3c',
    'Критический_ДМЖП': '#8e44ad'
}

# Соответствие между именами файлов и отображаемыми именами
FILE_NAME_MAPPING = {
    'Здоровый': 'Здоровый',
    'Малый_ДМЖП_R=5.0': 'Малый ДМЖП (R=5.0)',
    'Малый_ДМЖП': 'Малый ДМЖП (R=5.0)',
    'Большой_ДМЖП_R=1.0': 'Большой ДМЖП (R=1.0)',
    'Большой_ДМЖП': 'Большой ДМЖП (R=1.0)'
}

def safe_savgol_filter(data, window_length, polyorder):
    """Безопасная версия savgol_filter, обрабатывающая ошибки"""
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

def load_all_results():
    """Загрузка всех результатов симуляции"""
    results = {}
    
    # Поиск всех файлов с результатами
    files = glob.glob("vsd_results_*.npz")
    
    if not files:
        print("❌ Файлы с результатами не найдены!")
        print("   Убедитесь, что сначала запущен run_simulation.py")
        return None
    
    print(f"Найдено файлов: {len(files)}")
    
    for filepath in files:
        filename = filepath.replace('vsd_results_', '').replace('.npz', '')
        
        # Определяем отображаемое имя
        if filename in FILE_NAME_MAPPING:
            display_name = FILE_NAME_MAPPING[filename]
        elif 'Здоровый' in filename:
            display_name = 'Здоровый'
        elif 'Малый' in filename:
            display_name = 'Малый ДМЖП (R=5.0)'
        elif 'Большой' in filename:
            display_name = 'Большой ДМЖП (R=1.0)'
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
    
    print(f"\nЗагружено сценариев: {list(results.keys())}")
    return results

def plot_hemodynamic_timeseries(results):
    """Рис. 1: Временные ряды основных гемодинамических показателей"""
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
        ('V_blood', 'Объём крови (мл)', axes[2, 2])
    ]
    
    for metric, ylabel, ax in metrics:
        for scenario, data in results.items():
            if metric in data and len(data['t']) > 0:
                t = data['t'][:5000]
                values = data[metric][:5000]
                mask = np.isfinite(values)
                if np.any(mask):
                    ax.plot(t[mask], values[mask], color=COLORS.get(scenario, 'gray'), 
                           lw=1.5, label=scenario if metric == metrics[0][0] else '')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Время (с)')
        ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        if metric == metrics[0][0]:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('fig1_hemodynamics_timeseries.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_phase_portraits(results):
    """Рис. 2: Фазовые портреты (давление-объём для желудочков)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Фазовые портреты желудочков', fontsize=14, fontweight='bold')
    
    for scenario, data in results.items():
        if 'V_lv' in data and 'V_rv' in data:
            mask = np.isfinite(data['V_lv']) & np.isfinite(data['V_rv'])
            if np.any(mask):
                V_lv = data['V_lv'][mask][::100]
                V_rv = data['V_rv'][mask][::100]
                
                # Расчёт давления из объёма (упрощённая модель)
                P_lv = 0.5 * (V_lv - 50) ** 2 / 100 + 10
                P_rv = 0.3 * (V_rv - 60) ** 2 / 100 + 8
                
                axes[0].plot(V_lv, P_lv, color=COLORS.get(scenario, 'gray'), 
                            lw=1.5, alpha=0.7, label=scenario)
                axes[1].plot(V_rv, P_rv, color=COLORS.get(scenario, 'gray'), 
                            lw=1.5, alpha=0.7, label=scenario)
    
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

def plot_bar_comparison(results):
    """Рис. 3: Сравнение установившихся значений (столбчатая диаграмма)"""
    t_start = 200
    
    metrics = {
        'P_sa': 'Системное АД (мм рт. ст.)',
        'P_pa': 'Лёгочное АД (мм рт. ст.)',
        'Q_aortic': 'Системный выброс (мл/с)',
        'Q_pulmonary': 'Лёгочный кровоток (мл/с)',
        'Qp_Qs': 'Qp/Qs',
        'V_blood': 'Объём крови (мл)',
        'GFR': 'СКФ (мл/с)'
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    fig.suptitle('Сравнение установившихся показателей', fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    scenarios = list(results.keys())
    
    for idx, (metric, label) in enumerate(metrics.items()):
        ax = axes[idx]
        means = []
        stds = []
        
        for scenario in scenarios:
            data = results[scenario]
            if metric in data:
                mask = (data['t'] >= t_start) & np.isfinite(data[metric])
                values = data[metric][mask]
                if len(values) > 0:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(scenarios, means, color=[COLORS.get(s, 'gray') for s in scenarios], 
                     alpha=0.7, edgecolor='black', linewidth=1)
        ax.errorbar(scenarios, means, yerr=stds, fmt='none', 
                   ecolor='black', capsize=5, capthick=1)
        ax.set_ylabel(label)
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.02,
                       f'{mean:.1f}', ha='center', va='bottom', fontsize=8)
    
    axes[7].axis('off')
    plt.tight_layout()
    plt.savefig('fig3_bar_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_cardiovascular_parameters(results):
    """Рис. 4: Детальный анализ сердечно-сосудистых параметров"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Детальный анализ сердечно-сосудистых параметров', fontsize=14, fontweight='bold')
    
    t_start = 200
    scenarios = list(results.keys())
    
    # 1. Соотношение Qp/Qs во времени
    ax = axes[0, 0]
    for scenario, data in results.items():
        if 'Qp_Qs' in data:
            mask = (data['t'] >= 50) & np.isfinite(data['Qp_Qs'])
            if np.any(mask):
                t = data['t'][mask]
                qp_qs = data['Qp_Qs'][mask]
                if len(qp_qs) > 100:
                    qp_qs = safe_savgol_filter(qp_qs, 101, 3)
                ax.plot(t, qp_qs, color=COLORS.get(scenario, 'gray'), lw=1.5, label=scenario)
    ax.set_ylabel('Qp/Qs')
    ax.set_xlabel('Время (с)')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Норма (1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Соотношение лёгочного и системного кровотока')
    
    # 2. Объём ЛЖ
    ax = axes[0, 1]
    for scenario, data in results.items():
        if 'V_lv' in data:
            mask = (data['t'] >= 50) & np.isfinite(data['V_lv'])
            if np.any(mask):
                t = data['t'][mask]
                v_lv = data['V_lv'][mask]
                steady_mask = data['t'] >= t_start
                if np.any(steady_mask):
                    v_lv_steady = data['V_lv'][steady_mask]
                    v_lv_steady = v_lv_steady[np.isfinite(v_lv_steady)]
                    if len(v_lv_steady) > 0:
                        mean_lv = np.mean(v_lv_steady)
                        ax.axhline(y=mean_lv, color=COLORS.get(scenario, 'gray'), 
                                  linestyle='--', alpha=0.7, label=f'{scenario}: {mean_lv:.0f} мл')
                ax.plot(t, v_lv, color=COLORS.get(scenario, 'gray'), lw=1, alpha=0.5)
    ax.set_ylabel('Объём ЛЖ (мл)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Объём левого желудочка')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Объём ПЖ
    ax = axes[0, 2]
    for scenario, data in results.items():
        if 'V_rv' in data:
            mask = (data['t'] >= 50) & np.isfinite(data['V_rv'])
            if np.any(mask):
                t = data['t'][mask]
                v_rv = data['V_rv'][mask]
                steady_mask = data['t'] >= t_start
                if np.any(steady_mask):
                    v_rv_steady = data['V_rv'][steady_mask]
                    v_rv_steady = v_rv_steady[np.isfinite(v_rv_steady)]
                    if len(v_rv_steady) > 0:
                        mean_rv = np.mean(v_rv_steady)
                        ax.axhline(y=mean_rv, color=COLORS.get(scenario, 'gray'), 
                                  linestyle='--', alpha=0.7, label=f'{scenario}: {mean_rv:.0f} мл')
                ax.plot(t, v_rv, color=COLORS.get(scenario, 'gray'), lw=1, alpha=0.5)
    ax.set_ylabel('Объём ПЖ (мл)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Объём правого желудочка')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Корреляция Qp/Qs и Q_vsd
    ax = axes[1, 0]
    for scenario, data in results.items():
        if 'Qp_Qs' in data and 'Q_vsd' in data:
            mask = (data['t'] >= t_start) & np.isfinite(data['Qp_Qs']) & np.isfinite(data['Q_vsd'])
            if np.any(mask):
                qp_qs = data['Qp_Qs'][mask]
                q_vsd = data['Q_vsd'][mask]
                if len(q_vsd) > 0 and np.max(q_vsd) > 0:
                    ax.scatter(q_vsd[::50], qp_qs[::50], c=COLORS.get(scenario, 'gray'), 
                              s=10, alpha=0.5, label=scenario)
    ax.set_xlabel('Шунт VSD (мл/с)')
    ax.set_ylabel('Qp/Qs')
    ax.set_title('Корреляция: шунт ↔ Qp/Qs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Объёмы желудочков (сравнение)
    ax = axes[1, 1]
    chambers = ['V_lv', 'V_rv']
    chamber_names = ['Левый желудочек', 'Правый желудочек']
    x = np.arange(len(chambers))
    width = 0.25
    
    for i, scenario in enumerate(scenarios):
        means = []
        for chamber in chambers:
            if chamber in results[scenario]:
                mask = (results[scenario]['t'] >= t_start) & np.isfinite(results[scenario][chamber])
                values = results[scenario][chamber][mask]
                if len(values) > 0:
                    means.append(np.mean(values))
                else:
                    means.append(0)
            else:
                means.append(0)
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, label=scenario, 
               color=COLORS.get(scenario, 'gray'), alpha=0.7)
    ax.set_ylabel('Объём (мл)')
    ax.set_xticks(x)
    ax.set_xticklabels(chamber_names)
    ax.set_title('Сравнение объёмов желудочков')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Распределение кровотока
    ax = axes[1, 2]
    organs = ['Мозг', 'Почки', 'ЖКТ', 'Печень']
    organ_keys = ['Q_brain', 'Q_renal', 'Q_gitract_out', 'Q_liver_out']
    
    for scenario, data in results.items():
        flows = []
        for key in organ_keys:
            if key in data:
                mask = (data['t'] >= t_start) & np.isfinite(data[key])
                values = data[key][mask]
                flows.append(np.mean(values) if len(values) > 0 else 0)
            else:
                flows.append(0)
        ax.plot(organs, flows, 'o-', color=COLORS.get(scenario, 'gray'), lw=2, 
               markersize=8, label=scenario)
    ax.set_ylabel('Кровоток (мл/с)')
    ax.set_title('Региональное распределение кровотока')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig4_detailed_cardiac.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_comprehensive_dashboard(results):
    """Комплексная дашборд-панель"""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    t_start = 200
    scenarios = list(results.keys())
    
    # 1. Основная метрика: Qp/Qs
    ax1 = fig.add_subplot(gs[0, :2])
    for scenario, data in results.items():
        if 'Qp_Qs' in data:
            mask = (data['t'] >= 50) & np.isfinite(data['Qp_Qs'])
            if np.any(mask):
                t = data['t'][mask]
                qp_qs = data['Qp_Qs'][mask]
                if len(qp_qs) > 100:
                    qp_qs = safe_savgol_filter(qp_qs, 101, 3)
                ax1.plot(t, qp_qs, color=COLORS.get(scenario, 'gray'), 
                        lw=2, label=scenario)
    ax1.set_ylabel('Qp/Qs')
    ax1.set_xlabel('Время (с)')
    ax1.set_title('Соотношение лёгочного и системного кровотока', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Норма')
    ax1.axhline(y=1.5, color='orange', linestyle=':', alpha=0.5)
    ax1.axhline(y=2.0, color='red', linestyle=':', alpha=0.5)
    ax1.set_ylim(0.8, 2.8)
    
    # 2. Сравнение давлений
    ax2 = fig.add_subplot(gs[0, 2])
    metrics = ['P_sa', 'P_pa']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, scenario in enumerate(scenarios):
        data = results[scenario]
        means = []
        stds = []
        for metric in metrics:
            if metric in data:
                mask = (data['t'] >= t_start) & np.isfinite(data[metric])
                values = data[metric][mask]
                if len(values) > 0:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)
        offset = (i - 1) * width
        ax2.bar(x + offset, means, width, label=scenario,
                color=COLORS.get(scenario, 'gray'), alpha=0.7)
        ax2.errorbar(x + offset, means, yerr=stds, fmt='none', 
                    ecolor='black', capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Системное АД', 'Лёгочное АД'])
    ax2.set_ylabel('Давление (мм рт. ст.)')
    ax2.set_title('Сравнение артериальных давлений')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Объёмы желудочков
    ax3 = fig.add_subplot(gs[1, 0])
    chambers = ['Левый желудочек', 'Правый желудочек']
    x = np.arange(len(chambers))
    width = 0.25
    
    for i, scenario in enumerate(scenarios):
        data = results[scenario]
        volumes = []
        for chamber in ['V_lv', 'V_rv']:
            if chamber in data:
                mask = (data['t'] >= t_start) & np.isfinite(data[chamber])
                values = data[chamber][mask]
                volumes.append(np.mean(values) if len(values) > 0 else 0)
            else:
                volumes.append(0)
        offset = (i - 1) * width
        ax3.bar(x + offset, volumes, width, label=scenario,
               color=COLORS.get(scenario, 'gray'), alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(chambers)
    ax3.set_ylabel('Объём (мл)')
    ax3.set_title('Сравнение объёмов желудочков')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Кровотоки
    ax4 = fig.add_subplot(gs[1, 1])
    organs = ['Мозг', 'Почки', 'Печень', 'ЖКТ']
    organ_keys = ['Q_brain', 'Q_renal', 'Q_liver_out', 'Q_gitract_out']
    
    for scenario, data in results.items():
        flows = []
        for key in organ_keys:
            if key in data:
                mask = (data['t'] >= t_start) & np.isfinite(data[key])
                values = data[key][mask]
                flows.append(np.mean(values) if len(values) > 0 else 0)
            else:
                flows.append(0)
        ax4.plot(organs, flows, 'o-', lw=2, markersize=8,
                label=scenario, color=COLORS.get(scenario, 'gray'))
    ax4.set_ylabel('Кровоток (мл/с)')
    ax4.set_title('Региональное распределение кровотока')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Объём крови
    ax5 = fig.add_subplot(gs[1, 2])
    for scenario, data in results.items():
        if 'V_blood' in data:
            mask = (data['t'] >= 50) & np.isfinite(data['V_blood'])
            if np.any(mask):
                ax5.plot(data['t'][mask], data['V_blood'][mask],
                        color=COLORS.get(scenario, 'gray'), lw=1.5, label=scenario)
    ax5.set_ylabel('Объём крови (мл)')
    ax5.set_xlabel('Время (с)')
    ax5.set_title('Объём циркулирующей крови')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Потребление кислорода
    ax6 = fig.add_subplot(gs[2, 0])
    for scenario, data in results.items():
        if 'O2_consumption' in data:
            mask = (data['t'] >= 50) & np.isfinite(data['O2_consumption'])
            if np.any(mask):
                ax6.plot(data['t'][mask], data['O2_consumption'][mask],
                        color=COLORS.get(scenario, 'gray'), lw=1.5, label=scenario)
    ax6.set_ylabel('Потребление O₂ (у.е./с)')
    ax6.set_xlabel('Время (с)')
    ax6.set_title('Метаболизм мозга')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7. Функция почек
    ax7 = fig.add_subplot(gs[2, 1])
    for scenario, data in results.items():
        if 'GFR' in data:
            mask = (data['t'] >= 50) & np.isfinite(data['GFR'])
            if np.any(mask):
                ax7.plot(data['t'][mask], data['GFR'][mask],
                        color=COLORS.get(scenario, 'gray'), lw=1.5, label=scenario)
    ax7.set_ylabel('СКФ (мл/с)')
    ax7.set_xlabel('Время (с)')
    ax7.set_title('Функция почек')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 8. Корреляция шунт ↔ Qp/Qs
    ax8 = fig.add_subplot(gs[2, 2])
    shunt_means = []
    qp_qs_means = []
    
    for scenario, data in results.items():
        if 'Q_vsd' in data and 'Qp_Qs' in data:
            mask = (data['t'] >= t_start) & np.isfinite(data['Q_vsd']) & np.isfinite(data['Qp_Qs'])
            if np.any(mask):
                shunt_mean = np.mean(data['Q_vsd'][mask])
                qp_qs_mean = np.mean(data['Qp_Qs'][mask])
                if shunt_mean > 0:
                    shunt_means.append(shunt_mean)
                    qp_qs_means.append(qp_qs_mean)
                    ax8.scatter(shunt_mean, qp_qs_mean, s=120, 
                              c=COLORS.get(scenario, 'gray'), marker='o', 
                              edgecolor='black', linewidth=1.5, label=scenario)
    
    if len(shunt_means) > 1:
        slope, intercept, r_value, p_value, _ = linregress(shunt_means, qp_qs_means)
        x_line = np.array([0, max(shunt_means)])
        ax8.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5,
                label=f'R² = {r_value**2:.3f}')
        ax8.text(0.05, 0.95, f'Корреляция: r = {r_value:.3f}\np = {p_value:.4f}',
                transform=ax8.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax8.set_xlabel('Шунт VSD (мл/с)')
    ax8.set_ylabel('Qp/Qs')
    ax8.set_title('Корреляция: размер шунта → Qp/Qs')
    ax8.legend(fontsize=8, loc='lower right')
    ax8.grid(True, alpha=0.3)
    
    # 9. Статистическая таблица
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('tight')
    ax9.axis('off')
    
    table_data = [['Показатель', 'Здоровый', 'Малый ДМЖП', 'Большой ДМЖП']]
    
    summary_metrics = [
        ('Qp_Qs', 'Qp/Qs', '{:.2f} ± {:.2f}'),
        ('P_sa', 'АД сист. (мм рт.ст.)', '{:.0f} ± {:.0f}'),
        ('P_pa', 'АД лёг. (мм рт.ст.)', '{:.0f} ± {:.0f}'),
        ('V_lv', 'Объём ЛЖ (мл)', '{:.0f} ± {:.0f}'),
        ('V_rv', 'Объём ПЖ (мл)', '{:.0f} ± {:.0f}'),
        ('Q_aortic', 'Сист. выброс (мл/с)', '{:.1f} ± {:.1f}'),
        ('Q_pulmonary', 'Лёг. кровоток (мл/с)', '{:.1f} ± {:.1f}'),
        ('V_blood', 'Объём крови (мл)', '{:.0f} ± {:.0f}'),
        ('GFR', 'СКФ (мл/с)', '{:.2f} ± {:.2f}'),
        ('Q_brain', 'Мозг. кровоток (мл/с)', '{:.2f} ± {:.2f}')
    ]
    
    for metric, label, fmt in summary_metrics:
        row = [label]
        for scenario_name in ['Здоровый', 'Малый ДМЖП (R=5.0)', 'Большой ДМЖП (R=1.0)']:
            if scenario_name in results:
                data = results[scenario_name]
                if metric in data:
                    mask = (data['t'] >= t_start) & np.isfinite(data[metric])
                    values = data[metric][mask]
                    if len(values) > 0:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        row.append(fmt.format(mean_val, std_val))
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            else:
                row.append('N/A')
        table_data.append(row)
    
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.28, 0.22, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    
    # Цветовая кодировка
    for j in range(1, len(table_data)):
        for i, col_name in enumerate(['Здоровый', 'Малый ДМЖП (R=5.0)', 'Большой ДМЖП (R=1.0)']):
            try:
                cell = table[(j, i+1)]
                if 'Здоровый' in col_name:
                    cell.set_facecolor('#e8f8e8')
                elif 'Малый' in col_name:
                    cell.set_facecolor('#fff3e0')
                else:
                    cell.set_facecolor('#ffe8e8')
            except KeyError:
                continue
    
    fig.suptitle('📊 ДАШБОРД: Сравнение гемодинамики при ДМЖП\n' +
                 'Здоровый человек vs пациенты с дефектом межжелудочковой перегородки',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_schematic_heart_comparison():
    """Схематическое сравнение здорового сердца и сердца с ДМЖП"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle('Схематическое сравнение: здоровое сердце и сердце с ДМЖП',
                 fontsize=14, fontweight='bold')
    
    # Здоровое сердце
    ax1 = axes[0]
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 12)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Здоровое сердце', fontsize=12, fontweight='bold', color='green')
    
    # Рисуем камеры
    lv1 = Circle((4, 6), 2, facecolor='#e74c3c', edgecolor='black', alpha=0.7)
    rv1 = Circle((8, 6), 2, facecolor='#3498db', edgecolor='black', alpha=0.7)
    la1 = Circle((3.5, 9), 1.3, facecolor='#e74c3c', edgecolor='black', alpha=0.6)
    ra1 = Circle((8.5, 9), 1.3, facecolor='#3498db', edgecolor='black', alpha=0.6)
    
    ax1.add_patch(lv1)
    ax1.add_patch(rv1)
    ax1.add_patch(la1)
    ax1.add_patch(ra1)
    
    # Сосуды
    ax1.plot([4, 4], [6, 3], 'r-', lw=3)
    ax1.plot([8, 8], [6, 3], 'b-', lw=3)
    ax1.plot([4, 4], [9, 10.3], 'r-', lw=3)
    ax1.plot([8, 8], [9, 10.3], 'b-', lw=3)
    
    # Стрелки кровотока
    ax1.annotate('', xy=(4, 3), xytext=(4, 4), arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax1.annotate('', xy=(8, 3), xytext=(8, 4), arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax1.annotate('', xy=(4, 10.3), xytext=(4, 9), arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax1.annotate('', xy=(8, 10.3), xytext=(8, 9), arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Подписи
    ax1.text(4, 1.5, 'Аорта', ha='center', fontsize=10, fontweight='bold')
    ax1.text(8, 1.5, 'Лёгочная артерия', ha='center', fontsize=10, fontweight='bold')
    ax1.text(2, 6, 'ЛЖ', ha='center', fontsize=10, fontweight='bold')
    ax1.text(10, 6, 'ПЖ', ha='center', fontsize=10, fontweight='bold')
    ax1.text(2.5, 9, 'ЛП', ha='center', fontsize=9)
    ax1.text(9.5, 9, 'ПП', ha='center', fontsize=9)
    ax1.text(6, 11, 'Qs = Qp', ha='center', fontsize=11, fontweight='bold', color='green')
    
    # Сердце с ДМЖП
    ax2 = axes[1]
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 12)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Сердце с ДМЖП (большой дефект)', fontsize=12, fontweight='bold', color='red')
    
    # Рисуем камеры (увеличенный ПЖ)
    lv2 = Circle((4, 6), 2.2, facecolor='#e74c3c', edgecolor='black', alpha=0.7)
    rv2 = Circle((8.2, 6), 2.5, facecolor='#3498db', edgecolor='black', alpha=0.7)
    la2 = Circle((3.5, 9), 1.3, facecolor='#e74c3c', edgecolor='black', alpha=0.6)
    ra2 = Circle((8.7, 9), 1.5, facecolor='#3498db', edgecolor='black', alpha=0.6)
    
    ax2.add_patch(lv2)
    ax2.add_patch(rv2)
    ax2.add_patch(la2)
    ax2.add_patch(ra2)
    
    # Дефект
    vsd = Rectangle((5.5, 5.5), 1.2, 1.2, facecolor='purple', edgecolor='black', alpha=0.8)
    ax2.add_patch(vsd)
    
    # Сосуды
    ax2.plot([4, 4], [6, 3], 'r-', lw=2.5)
    ax2.plot([8.2, 8.2], [6, 3], 'b-', lw=4)
    ax2.plot([4, 4], [9, 10.3], 'r-', lw=2.5)
    ax2.plot([8.2, 8.2], [9, 10.3], 'b-', lw=4)
    
    # Стрелки кровотока
    ax2.annotate('', xy=(4, 3), xytext=(4, 4), arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax2.annotate('', xy=(8.2, 3), xytext=(8.2, 4), arrowprops=dict(arrowstyle='->', lw=4, color='blue'))
    ax2.annotate('', xy=(4, 10.3), xytext=(4, 9), arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax2.annotate('', xy=(8.2, 10.3), xytext=(8.2, 9), arrowprops=dict(arrowstyle='->', lw=4, color='blue'))
    
    # Шунт
    ax2.annotate('', xy=(6.1, 6), xytext=(5.8, 5.2), arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
    
    # Подписи
    ax2.text(4, 1.5, 'Аорта (Qs ↓)', ha='center', fontsize=9, fontweight='bold', color='red')
    ax2.text(8.2, 1.5, 'Лёгочная артерия (Qp ↑↑)', ha='center', fontsize=9, fontweight='bold', color='blue')
    ax2.text(1.8, 6, 'ЛЖ', ha='center', fontsize=10, fontweight='bold')
    ax2.text(10.5, 6, 'ПЖ (дилатация)', ha='center', fontsize=9, fontweight='bold')
    ax2.text(2.5, 9, 'ЛП', ha='center', fontsize=9)
    ax2.text(9.8, 9, 'ПП', ha='center', fontsize=9)
    ax2.text(6.1, 4.5, 'VSD', ha='center', fontsize=9, fontweight='bold', color='purple')
    ax2.text(6, 11, 'Qs < Qp (лево-правый шунт)', ha='center', fontsize=10, fontweight='bold', color='purple')
    
    plt.tight_layout()
    plt.savefig('schematic_heart_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_statistical_summary(results):
    """Вывод статистической сводки"""
    t_start = 200
    scenarios = list(results.keys())
    
    print("\n" + "="*90)
    print("СТАТИСТИЧЕСКАЯ СВОДКА (установившийся режим, t > {:.0f} с)".format(t_start))
    print("="*90)
    
    metrics = [
        ('P_sa', 'Системное АД', 'мм рт. ст.', '{:.1f} ± {:.1f}'),
        ('P_pa', 'Лёгочное АД', 'мм рт. ст.', '{:.1f} ± {:.1f}'),
        ('Q_aortic', 'Системный выброс', 'мл/с', '{:.1f} ± {:.1f}'),
        ('Q_pulmonary', 'Лёгочный кровоток', 'мл/с', '{:.1f} ± {:.1f}'),
        ('Qp_Qs', 'Qp/Qs', '', '{:.2f} ± {:.2f}'),
        ('Q_vsd', 'Шунт VSD', 'мл/с', '{:.1f} ± {:.1f}'),
        ('V_lv', 'Объём ЛЖ', 'мл', '{:.1f} ± {:.1f}'),
        ('V_rv', 'Объём ПЖ', 'мл', '{:.1f} ± {:.1f}'),
        ('V_blood', 'Объём крови', 'мл', '{:.0f} ± {:.0f}'),
        ('GFR', 'СКФ', 'мл/с', '{:.2f} ± {:.2f}'),
        ('Q_brain', 'Мозговой кровоток', 'мл/с', '{:.2f} ± {:.2f}'),
        ('O2_consumption', 'Потребление O₂ мозгом', 'у.е./с', '{:.2f} ± {:.2f}')
    ]
    
    print(f"{'Показатель':<35}", end='')
    for scenario in scenarios:
        display_name = scenario[:18] if len(scenario) > 18 else scenario
        print(f"{display_name:>20}", end='')
    print("\n" + "-"*90)
    
    for metric, label, unit, fmt in metrics:
        print(f"{label} [{unit}]".ljust(35), end='')
        for scenario in scenarios:
            if scenario in results and metric in results[scenario]:
                mask = (results[scenario]['t'] >= t_start) & np.isfinite(results[scenario][metric])
                values = results[scenario][metric][mask]
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"{fmt.format(mean_val, std_val):>20}", end='')
                else:
                    print(f"{'N/A':>20}", end='')
            else:
                print(f"{'N/A':>20}", end='')
        print()
    
    print("="*90)
    
    # Дополнительные выводы
    print("\n🔍 КЛЮЧЕВЫЕ НАХОДКИ:")
    
    healthy_scenario = None
    large_vsd_scenario = None
    
    for s in scenarios:
        if 'Здоровый' in s:
            healthy_scenario = s
        if 'Большой' in s:
            large_vsd_scenario = s
    
    if healthy_scenario and large_vsd_scenario:
        try:
            healthy_data = results[healthy_scenario]
            vsd_data = results[large_vsd_scenario]
            
            mask_h = (healthy_data['t'] >= t_start) & np.isfinite(healthy_data['Qp_Qs'])
            mask_v = (vsd_data['t'] >= t_start) & np.isfinite(vsd_data['Qp_Qs'])
            
            if np.any(mask_h) and np.any(mask_v):
                qp_qs_healthy = np.mean(healthy_data['Qp_Qs'][mask_h])
                qp_qs_vsd = np.mean(vsd_data['Qp_Qs'][mask_v])
                print(f"  • Qp/Qs увеличивается с {qp_qs_healthy:.2f} (норма) до {qp_qs_vsd:.2f} (большой ДМЖП)")
            
            mask_h_pp = (healthy_data['t'] >= t_start) & np.isfinite(healthy_data['P_pa'])
            mask_v_pp = (vsd_data['t'] >= t_start) & np.isfinite(vsd_data['P_pa'])
            
            if np.any(mask_h_pp) and np.any(mask_v_pp):
                ppa_healthy = np.mean(healthy_data['P_pa'][mask_h_pp])
                ppa_vsd = np.mean(vsd_data['P_pa'][mask_v_pp])
                print(f"  • Лёгочное АД повышается с {ppa_healthy:.0f} до {ppa_vsd:.0f} мм рт. ст.")
            
            mask_h_qs = (healthy_data['t'] >= t_start) & np.isfinite(healthy_data['Q_aortic'])
            mask_v_qs = (vsd_data['t'] >= t_start) & np.isfinite(vsd_data['Q_aortic'])
            
            if np.any(mask_h_qs) and np.any(mask_v_qs):
                qs_healthy = np.mean(healthy_data['Q_aortic'][mask_h_qs])
                qs_vsd = np.mean(vsd_data['Q_aortic'][mask_v_qs])
                if qs_healthy > 0:
                    print(f"  • Системный выброс снижается на {(1 - qs_vsd/qs_healthy)*100:.0f}%")
            
            mask_h_rv = (healthy_data['t'] >= t_start) & np.isfinite(healthy_data['V_rv'])
            mask_v_rv = (vsd_data['t'] >= t_start) & np.isfinite(vsd_data['V_rv'])
            
            if np.any(mask_h_rv) and np.any(mask_v_rv):
                rv_healthy = np.mean(healthy_data['V_rv'][mask_h_rv])
                rv_vsd = np.mean(vsd_data['V_rv'][mask_v_rv])
                if rv_healthy > 0:
                    print(f"  • Правый желудочек дилатируется: {rv_healthy:.0f} → {rv_vsd:.0f} мл (+{(rv_vsd/rv_healthy-1)*100:.0f}%)")
        except Exception as e:
            print(f"  • Некоторые данные недоступны для сравнения: {e}")

def main():
    """Основная функция"""
    print("="*70)
    print("ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ ГЕМОДИНАМИКИ: ЗДОРОВЫЙ vs ДМЖП")
    print("="*70)
    
    # Загрузка данных
    results = load_all_results()
    if results is None or len(results) == 0:
        print("\n⚠️ Не удалось загрузить данные!")
        print("   Убедитесь, что сначала запущен run_simulation.py и файлы существуют.")
        return
    
    # Генерация всех графиков
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
    
    # Статистическая сводка
    print_statistical_summary(results)
    
    print("\n✅ Все графики сохранены в текущей директории!")
    print("📁 Файлы: fig1_hemodynamics_timeseries.png, fig2_phase_portraits.png, ...")

if __name__ == "__main__":
    main()
