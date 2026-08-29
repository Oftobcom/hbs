#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_vsd_comparison.py
Расширенная визуализация сравнения гемодинамики здорового человека и пациента с ДМЖП.
Включает: временные ряды, фазовые портреты, статистические диаграммы.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.patches as mpatches
from scipy import signal
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

def load_simulation_results():
    """Загрузка результатов симуляции из файлов"""
    results = {}
    scenarios = ['Здоровый', 'Малый_ДМЖП', 'Большой_ДМЖП']
    
    for scenario in scenarios:
        filename = f"vsd_results_{scenario}.npz"
        try:
            data = np.load(filename, allow_pickle=True)
            results[scenario] = {key: data[key] for key in data.files}
            print(f"Загружен {filename}")
        except FileNotFoundError:
            print(f"Файл {filename} не найден. Запустите run_simulation.py сначала.")
            return None
    
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
    
    colors = {'Здоровый': '#2ecc71', 'Малый_ДМЖП': '#f39c12', 'Большой_ДМЖП': '#e74c3c'}
    labels = {'Здоровый': 'Здоровый', 'Малый_ДМЖП': 'Малый ДМЖП (R=5.0)', 
              'Большой_ДМЖП': 'Большой ДМЖП (R=1.0)'}
    
    for metric, ylabel, ax in metrics:
        for scenario, data in results.items():
            if metric in data:
                t = data['t']
                values = data[metric]
                # Сглаживание для уменьшения шума
                if len(values) > 100:
                    values = signal.savgol_filter(values, min(51, len(values)//10*2+1), 3)
                ax.plot(t, values, color=colors[scenario], lw=1.5, 
                       label=labels[scenario] if metric == metrics[0][0] else '')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Время (с)')
        ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='Установившийся режим' if metric == metrics[0][0] else '')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('fig1_hemodynamics_timeseries.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_phase_portraits(results):
    """Рис. 2: Фазовые портреты (давление-объём для желудочков)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Фазовые портреты желудочков', fontsize=14, fontweight='bold')
    
    colors = {'Здоровый': '#2ecc71', 'Малый_ДМЖП': '#f39c12', 'Большой_ДМЖП': '#e74c3c'}
    
    # Данные для P-V петли (нужно рассчитать из существующих данных)
    # Для демонстрации используем аппроксимацию
    t = results['Здоровый']['t']
    
    for scenario, data in results.items():
        # Аппроксимация давления из объёма (упрощённо)
        V_lv = data['V_lv']
        V_rv = data['V_rv']
        
        # Модель P-V зависимости (эластансная модель)
        P_lv_approx = 0.5 * (V_lv - 50) ** 2 / 100 + 10  # Упрощённая парабола
        P_rv_approx = 0.3 * (V_rv - 60) ** 2 / 100 + 8
        
        # Левый желудочек
        axes[0].plot(V_lv[::50], P_lv_approx[::50], color=colors[scenario], 
                    lw=1.5, alpha=0.7, label=scenario)
        # Правый желудочек
        axes[1].plot(V_rv[::50], P_rv_approx[::50], color=colors[scenario], 
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
    t_start = 150  # Время установления
    
    # Определяем метрики для сравнения
    metrics = {
        'P_sa': 'Системное АД\n(мм рт. ст.)',
        'P_pa': 'Лёгочное АД\n(мм рт. ст.)',
        'Q_aortic': 'Системный выброс\n(мл/с)',
        'Q_pulmonary': 'Лёгочный кровоток\n(мл/с)',
        'Qp_Qs': 'Qp/Qs',
        'V_blood': 'Объём крови\n(мл)',
        'GFR': 'СКФ\n(мл/с)'
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    fig.suptitle('Сравнение установившихся показателей', fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    scenarios = list(results.keys())
    
    for idx, (metric, label) in enumerate(metrics.items()):
        ax = axes[idx]
        means = []
        stds = []
        
        for scenario in scenarios:
            data = results[scenario]
            if metric in data:
                mask = data['t'] >= t_start
                values = data[metric][mask]
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(scenarios, means, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=1)
        ax.errorbar(scenarios, means, yerr=stds, fmt='none', 
                   ecolor='black', capsize=5, capthick=1)
        ax.set_ylabel(label)
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for bar, mean in zip(bars, means):
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
    
    t_start = 150
    scenarios = list(results.keys())
    colors = {'Здоровый': '#2ecc71', 'Малый_ДМЖП': '#f39c12', 'Большой_ДМЖП': '#e74c3c'}
    
    # 1. Соотношение Qp/Qs во времени
    ax = axes[0, 0]
    for scenario, data in results.items():
        if 'Qp_Qs' in data:
            t = data['t']
            qp_qs = data['Qp_Qs']
            ax.plot(t, qp_qs, color=colors[scenario], lw=1.5, label=scenario)
    ax.set_ylabel('Qp/Qs')
    ax.set_xlabel('Время (с)')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Норма (1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Соотношение лёгочного и системного кровотока')
    
    # 2. Конечный диастолический объём ЛЖ
    ax = axes[0, 1]
    for scenario, data in results.items():
        if 'V_lv' in data:
            t = data['t']
            # Находим максимумы (конечный диастолический объём)
            v_lv = data['V_lv']
            # Простое сглаживание для поиска пиков
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(v_lv, distance=50, prominence=5)
            if len(peaks) > 10:
                edv = np.mean(v_lv[peaks[-20:]])
                ax.axhline(y=edv, color=colors[scenario], linestyle='--', 
                          alpha=0.7, label=f'{scenario}: {edv:.0f} мл')
            ax.plot(t, v_lv, color=colors[scenario], lw=1, alpha=0.5)
    ax.set_ylabel('Объём ЛЖ (мл)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Конечно-диастолический объём ЛЖ')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Фракция выброса ЛЖ (расчётная)
    ax = axes[0, 2]
    for scenario, data in results.items():
        if 'V_lv' in data and 'Q_aortic' in data:
            v_lv = data['V_lv']
            # Упрощённая оценка: max - min / max
            mask = data['t'] >= t_start
            v_lv_steady = v_lv[mask]
            edv_est = np.max(v_lv_steady)
            esv_est = np.min(v_lv_steady)
            ef = (edv_est - esv_est) / edv_est * 100 if edv_est > 0 else 0
            # Создаём массив для отображения
            ef_array = np.ones_like(data['t'][mask]) * ef
            ax.plot(data['t'][mask], ef_array, color=colors[scenario], 
                   lw=2, label=f'{scenario}: {ef:.1f}%')
    ax.set_ylabel('Фракция выброса (%)')
    ax.set_xlabel('Время (с)')
    ax.set_ylim(0, 80)
    ax.set_title('Фракция выброса левого желудочка')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Корреляция Qp/Qs и Q_vsd
    ax = axes[1, 0]
    for scenario, data in results.items():
        if 'Qp_Qs' in data and 'Q_vsd' in data:
            mask = data['t'] >= t_start
            qp_qs = data['Qp_Qs'][mask]
            q_vsd = data['Q_vsd'][mask]
            ax.scatter(q_vsd, qp_qs, c=colors[scenario], s=10, alpha=0.5, label=scenario)
    ax.set_xlabel('Шунт VSD (мл/с)')
    ax.set_ylabel('Qp/Qs')
    ax.set_title('Корреляция: шунт ↔ Qp/Qs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Давление в камерах сердца
    ax = axes[1, 1]
    # Используем объёмы как суррогат давления
    chambers = ['V_lv', 'V_rv']
    chamber_names = ['Левый желудочек', 'Правый желудочек']
    x = np.arange(len(chambers))
    width = 0.25
    
    for i, scenario in enumerate(scenarios):
        means = []
        for chamber in chambers:
            if chamber in results[scenario]:
                mask = results[scenario]['t'] >= t_start
                means.append(np.mean(results[scenario][chamber][mask]))
            else:
                means.append(0)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width, label=scenario, 
                     color=colors[scenario], alpha=0.7)
    
    ax.set_ylabel('Объём (мл)')
    ax.set_xticks(x)
    ax.set_xticklabels(chamber_names)
    ax.set_title('Сравнение объёмов желудочков')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Распределение кровотока
    ax = axes[1, 2]
    organs = ['Мозг', 'Почки', 'ЖКТ', 'Печень']
    for scenario, data in results.items():
        flows = []
        mask = data['t'] >= t_start
        flows.append(np.mean(data.get('Q_brain', [0])[mask]) if 'Q_brain' in data else 0)
        flows.append(np.mean(data.get('Q_renal', [0])[mask]) if 'Q_renal' in data else 0)
        flows.append(np.mean(data.get('Q_gitract_out', [0])[mask]) if 'Q_gitract_out' in data else 0)
        flows.append(np.mean(data.get('Q_liver_out', [0])[mask]) if 'Q_liver_out' in data else 0)
        
        ax.plot(organs, flows, 'o-', color=colors[scenario], lw=2, 
               markersize=8, label=scenario)
    ax.set_ylabel('Кровоток (мл/с)')
    ax.set_title('Региональное распределение кровотока')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig4_detailed_cardiac.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_metabolic_impact(results):
    """Рис. 5: Влияние ДМЖП на метаболические показатели"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Влияние ДМЖП на метаболические показатели', fontsize=14, fontweight='bold')
    
    t_start = 150
    colors = {'Здоровый': '#2ecc71', 'Малый_ДМЖП': '#f39c12', 'Большой_ДМЖП': '#e74c3c'}
    scenarios = list(results.keys())
    
    # 1. Церебральный кровоток и потребление O2
    ax = axes[0, 0]
    for scenario, data in results.items():
        if 'Q_brain' in data and 'O2_consumption' in data:
            mask = data['t'] >= t_start
            q_brain = data['Q_brain'][mask]
            o2_cons = data['O2_consumption'][mask]
            ax.scatter(q_brain[::50], o2_cons[::50], c=colors[scenario], 
                      s=15, alpha=0.5, label=scenario)
    ax.set_xlabel('Мозговой кровоток (мл/с)')
    ax.set_ylabel('Потребление O₂ (у.е./с)')
    ax.set_title('Метаболизм мозга')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Функция печени
    ax = axes[0, 1]
    for scenario, data in results.items():
        if 'liver_functional' in data:
            t = data['t']
            func = data['liver_functional']
            mask = t >= t_start
            # Нормализованный показатель
            ax.plot(t[mask], func[mask], color=colors[scenario], lw=1.5, label=scenario)
    ax.set_ylabel('Функциональная активность печени')
    ax.set_xlabel('Время (с)')
    ax.set_ylim(0.5, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Норма')
    ax.set_title('Функция печени')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Оксигенация
    ax = axes[1, 0]
    for scenario, data in results.items():
        if 'oxygenation_index' in data:
            t = data['t']
            ox = data['oxygenation_index']
            mask = t >= t_start
            ax.plot(t[mask], ox[mask], color=colors[scenario], lw=1.5, label=scenario)
    ax.set_ylabel('Индекс оксигенации')
    ax.set_xlabel('Время (с)')
    ax.set_ylim(0.7, 1.05)
    ax.axhline(y=0.98, color='gray', linestyle='--', alpha=0.5, label='Норма')
    ax.set_title('Оксигенация крови')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Объём крови (волемический статус)
    ax = axes[1, 1]
    for scenario, data in results.items():
        if 'V_blood' in data:
            t = data['t']
            v_blood = data['V_blood']
            ax.plot(t, v_blood, color=colors[scenario], lw=1.5, label=scenario)
    ax.set_ylabel('Объём крови (мл)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Волемический статус')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig5_metabolic_impact.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_schematic_diagram():
    """Рис. 6: Схематическое представление гемодинамики при ДМЖП"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Схематическое представление гемодинамики при ДМЖП', fontsize=14, fontweight='bold')
    
    # Здоровое сердце
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Здоровое сердце', fontsize=12, fontweight='bold')
    
    # Рисуем камеры здорового сердца
    lv1 = Circle((4, 5), 1.5, facecolor='#e74c3c', edgecolor='black', alpha=0.7)
    rv1 = Circle((6, 5), 1.5, facecolor='#3498db', edgecolor='black', alpha=0.7)
    la1 = Circle((3.5, 7.5), 1.0, facecolor='#e74c3c', edgecolor='black', alpha=0.6)
    ra1 = Circle((6.5, 7.5), 1.0, facecolor='#3498db', edgecolor='black', alpha=0.6)
    
    ax1.add_patch(lv1)
    ax1.add_patch(rv1)
    ax1.add_patch(la1)
    ax1.add_patch(ra1)
    
    # Стрелки кровотока
    ax1.annotate('', xy=(4, 3.2), xytext=(4, 3.8), 
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax1.annotate('', xy=(6, 3.2), xytext=(6, 3.8), 
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax1.annotate('', xy=(8, 5), xytext=(7.8, 5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax1.annotate('', xy=(2, 5), xytext=(2.2, 5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    ax1.text(4, 2.5, 'Аорта', ha='center', fontsize=9)
    ax1.text(6, 2.5, 'Лёгочная\nартерия', ha='center', fontsize=9)
    ax1.text(1, 5, 'Системный\nкровоток (Qs)', ha='center', fontsize=8)
    ax1.text(9, 5, 'Лёгочный\nкровоток (Qp)', ha='center', fontsize=8)
    ax1.text(4, 9, 'Qs = Qp', ha='center', fontsize=10, fontweight='bold', color='green')
    
    # Сердце с ДМЖП
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Сердце с ДМЖП (большой дефект)', fontsize=12, fontweight='bold')
    
    # Рисуем камеры
    lv2 = Circle((3.5, 5), 1.8, facecolor='#e74c3c', edgecolor='black', alpha=0.7)
    rv2 = Circle((6.5, 5), 2.0, facecolor='#3498db', edgecolor='black', alpha=0.7)
    la2 = Circle((3, 7.5), 1.2, facecolor='#e74c3c', edgecolor='black', alpha=0.6)
    ra2 = Circle((7, 7.5), 1.2, facecolor='#3498db', edgecolor='black', alpha=0.6)
    
    ax2.add_patch(lv2)
    ax2.add_patch(rv2)
    ax2.add_patch(la2)
    ax2.add_patch(ra2)
    
    # Дефект
    vsd = FancyBboxPatch((4.7, 4.5), 1.0, 1.0, 
                         boxstyle="round,pad=0.1", 
                         facecolor='purple', edgecolor='black', alpha=0.8)
    ax2.add_patch(vsd)
    
    # Стрелки
    ax2.annotate('', xy=(3.5, 3.2), xytext=(3.5, 3.8), 
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax2.annotate('', xy=(6.5, 3.0), xytext=(6.5, 3.8), 
                arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    ax2.annotate('', xy=(8.5, 5), xytext=(8.3, 5), 
                arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    ax2.annotate('', xy=(1.5, 5), xytext=(1.7, 5), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
    
    # Шунт
    ax2.annotate('', xy=(5.2, 5), xytext=(5, 4.2), 
                arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
    
    ax2.text(3.5, 2.2, 'Аорта', ha='center', fontsize=9)
    ax2.text(6.5, 2.0, 'Лёгочная\nартерия', ha='center', fontsize=9)
    ax2.text(0.5, 5, 'Qs ↓', ha='center', fontsize=9, color='red')
    ax2.text(9, 5, 'Qp ↑↑', ha='center', fontsize=9, color='blue')
    ax2.text(5, 3.5, 'VSD', ha='center', fontsize=8, fontweight='bold', color='purple')
    ax2.text(4, 9, 'Qs < Qp (лево-правый шунт)', ha='center', fontsize=10, 
            fontweight='bold', color='purple')
    
    plt.tight_layout()
    plt.savefig('fig6_schematic_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_statistical_summary(results):
    """Вывод статистической сводки"""
    t_start = 150
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
        print(f"{scenario:>20}", end='')
    print("\n" + "-"*90)
    
    for metric, label, unit, fmt in metrics:
        print(f"{label} [{unit}]".ljust(35), end='')
        for scenario in scenarios:
            if metric in results[scenario]:
                mask = results[scenario]['t'] >= t_start
                values = results[scenario][metric][mask]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{fmt.format(mean_val, std_val):>20}", end='')
            else:
                print(f"{'N/A':>20}", end='')
        print()
    
    print("="*90)
    
    # Дополнительные выводы
    print("\n🔍 КЛЮЧЕВЫЕ НАХОДКИ:")
    
    if 'Здоровый' in results and 'Большой_ДМЖП' in results:
        # Сравнение Qp/Qs
        qp_qs_healthy = np.mean(results['Здоровый']['Qp_Qs'][results['Здоровый']['t'] >= t_start])
        qp_qs_vsd = np.mean(results['Большой_ДМЖП']['Qp_Qs'][results['Большой_ДМЖП']['t'] >= t_start])
        print(f"  • Qp/Qs увеличивается с {qp_qs_healthy:.2f} (норма) до {qp_qs_vsd:.2f} (большой ДМЖП)")
        
        # Сравнение лёгочного давления
        ppa_healthy = np.mean(results['Здоровый']['P_pa'][results['Здоровый']['t'] >= t_start])
        ppa_vsd = np.mean(results['Большой_ДМЖП']['P_pa'][results['Большой_ДМЖП']['t'] >= t_start])
        print(f"  • Лёгочное АД повышается с {ppa_healthy:.0f} до {ppa_vsd:.0f} мм рт. ст.")
        
        # Изменение системного выброса
        qs_healthy = np.mean(results['Здоровый']['Q_aortic'][results['Здоровый']['t'] >= t_start])
        qs_vsd = np.mean(results['Большой_ДМЖП']['Q_aortic'][results['Большой_ДМЖП']['t'] >= t_start])
        print(f"  • Системный выброс снижается на {(1 - qs_vsd/qs_healthy)*100:.0f}%")
        
        # Объём ПЖ
        rv_healthy = np.mean(results['Здоровый']['V_rv'][results['Здоровый']['t'] >= t_start])
        rv_vsd = np.mean(results['Большой_ДМЖП']['V_rv'][results['Большой_ДМЖП']['t'] >= t_start])
        print(f"  • Правый желудочек дилатируется: {rv_healthy:.0f} → {rv_vsd:.0f} мл (+{(rv_vsd/rv_healthy-1)*100:.0f}%)")

def create_dashboard(results):
    """Создание единой дашборд-панели"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = {'Здоровый': '#2ecc71', 'Малый_ДМЖП': '#f39c12', 'Большой_ДМЖП': '#e74c3c'}
    t_start = 150
    
    # 1. Qp/Qs во времени
    ax1 = fig.add_subplot(gs[0, 0])
    for scenario, data in results.items():
        if 'Qp_Qs' in data:
            ax1.plot(data['t'], data['Qp_Qs'], color=colors[scenario], lw=1.5, label=scenario)
    ax1.set_ylabel('Qp/Qs')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Соотношение Qp/Qs')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Давления
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(2)
    width = 0.25
    for i, scenario in enumerate(scenarios):
        ps = np.mean(results[scenario]['P_sa'][results[scenario]['t'] >= t_start]) if 'P_sa' in results[scenario] else 0
        pp = np.mean(results[scenario]['P_pa'][results[scenario]['t'] >= t_start]) if 'P_pa' in results[scenario] else 0
        ax2.bar(x[0] + i*width, ps, width, label=scenario, color=colors[scenario], alpha=0.7)
        ax2.bar(x[1] + i*width, pp, width, color=colors[scenario], alpha=0.4)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(['Системное', 'Лёгочное'])
    ax2.set_ylabel('Давление (мм рт. ст.)')
    ax2.set_title('Сравнение давлений')
    ax2.legend(fontsize=8)
    
    # 3. Объёмы желудочков (радарная диаграмма)
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')
    chambers = ['ЛЖ', 'ПЖ']
    angles = np.linspace(0, 2*np.pi, len(chambers), endpoint=False).tolist()
    angles += angles[:1]
    
    for scenario in scenarios:
        values = []
        for chamber in ['V_lv', 'V_rv']:
            if chamber in results[scenario]:
                v = np.mean(results[scenario][chamber][results[scenario]['t'] >= t_start])
                values.append(v)
        values += values[:1]
        ax3.plot(angles, values, 'o-', linewidth=2, label=scenario, color=colors[scenario])
        ax3.fill(angles, values, alpha=0.1, color=colors[scenario])
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(chambers)
    ax3.set_title('Объёмы желудочков')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=8)
    
    # 4. Распределение кровотока
    ax4 = fig.add_subplot(gs[1, :])
    organs = ['Мозг', 'Почки', 'ЖКТ', 'Печень']
    for scenario in scenarios:
        flows = []
        for organ in ['Q_brain', 'Q_renal', 'Q_gitract_out', 'Q_liver_out']:
            if organ in results[scenario]:
                f = np.mean(results[scenario][organ][results[scenario]['t'] >= t_start])
                flows.append(f)
            else:
                flows.append(0)
        ax4.plot(organs, flows, 'o-', lw=2, markersize=8, label=scenario, color=colors[scenario])
    ax4.set_ylabel('Кровоток (мл/с)')
    ax4.set_title('Региональное распределение кровотока')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Метрболические показатели
    ax5 = fig.add_subplot(gs[2, 0])
    for scenario in scenarios:
        if 'O2_consumption' in results[scenario]:
            mask = results[scenario]['t'] >= t_start
            ax5.plot(results[scenario]['t'][mask], 
                    results[scenario]['O2_consumption'][mask], 
                    color=colors[scenario], lw=1.5, label=scenario)
    ax5.set_ylabel('Потребление O₂ (у.е./с)')
    ax5.set_title('Метаболизм мозга')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Объём крови
    ax6 = fig.add_subplot(gs[2, 1])
    for scenario in scenarios:
        if 'V_blood' in results[scenario]:
            ax6.plot(results[scenario]['t'], results[scenario]['V_blood'], 
                    color=colors[scenario], lw=1.5, label=scenario)
    ax6.set_ylabel('Объём крови (мл)')
    ax6.set_title('Волемический статус')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7. Статистическая таблица
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('tight')
    ax7.axis('off')
    
    # Создаём таблицу
    table_data = []
    headers = ['Показатель'] + scenarios
    table_data.append(['Qp/Qs', 
                      f"{np.mean(results['Здоровый']['Qp_Qs'][results['Здоровый']['t']>=t_start]):.2f}" if 'Здоровый' in results else 'N/A',
                      f"{np.mean(results['Малый_ДМЖП']['Qp_Qs'][results['Малый_ДМЖП']['t']>=t_start]):.2f}" if 'Малый_ДМЖП' in results else 'N/A',
                      f"{np.mean(results['Большой_ДМЖП']['Qp_Qs'][results['Большой_ДМЖП']['t']>=t_start]):.2f}" if 'Большой_ДМЖП' in results else 'N/A'])
    table_data.append(['ФВ ЛЖ (%)', 
                      'N/A', 'N/A', 'N/A'])  # Можно добавить расчёт
    
    table = ax7.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax7.set_title('Ключевые показатели', fontsize=10, fontweight='bold')
    
    fig.suptitle('📊 ДАШБОРД: Сравнение гемодинамики при ДМЖП', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig7_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Основная функция"""
    print("="*70)
    print("ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ ГЕМОДИНАМИКИ: ЗДОРОВЫЙ vs ДМЖП")
    print("="*70)
    
    # Загрузка данных
    results = load_simulation_results()
    if results is None:
        print("\n⚠️ Сначала запустите run_simulation.py для генерации данных!")
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
    
    print("  • Рис. 5: Метаболические эффекты")
    plot_metabolic_impact(results)
    
    print("  • Рис. 6: Схематическая диаграмма")
    plot_schematic_diagram()
    
    print("  • Рис. 7: Дашборд")
    create_dashboard(results)
    
    # Статистическая сводка
    print_statistical_summary(results)
    
    print("\n✅ Все графики сохранены в текущей директории!")
    print("📁 Файлы: fig1_hemodynamics_timeseries.png, fig2_phase_portraits.png, ...")
    
if __name__ == "__main__":
    main()
