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

def safe_savgol_filter(data, window_length, polyorder):
    """Безопасная версия savgol_filter, обрабатывающая ошибки"""
    from scipy.signal import savgol_filter
    
    # Очищаем данные от inf/NaN
    data_clean = np.where(np.isfinite(data), data, np.nan)
    
    # Заполняем NaN интерполяцией
    if np.any(np.isnan(data_clean)):
        nan_mask = np.isnan(data_clean)
        # Находим индексы валидных значений
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) > 1:
            data_clean[nan_mask] = np.interp(
                np.flatnonzero(nan_mask),
                valid_indices,
                data_clean[valid_indices]
            )
        else:
            # Если нет валидных значений для интерполяции, возвращаем исходные
            return data
    
    try:
        # Убеждаемся, что window_length нечётный и меньше длины данных
        if window_length > len(data_clean):
            window_length = len(data_clean) if len(data_clean) % 2 == 1 else len(data_clean) - 1
        if window_length < 3:
            return data_clean
        # Убеждаемся, что polyorder меньше window_length
        if polyorder >= window_length:
            polyorder = window_length - 1
        if polyorder < 1:
            return data_clean
        return savgol_filter(data_clean, window_length, polyorder)
    except Exception:
        return data

def simulate_scenario(vsd_resistance, flow_dependent_lungs, label, color, 
                      t_span=(0, 300), t_eval=None, description=""):
    """
    Запускает симуляцию для заданного сопротивления ДМЖП.
    
    Parameters:
    -----------
    vsd_resistance : float
        Сопротивление дефекта (Ом). inf = здоровый
    flow_dependent_lungs : bool
        Учитывать ли рост сопротивления лёгких при перегрузке
    label : str
        Метка для легенды
    color : str
        Цвет для графиков
    t_span : tuple
        Интервал времени (start, end)
    t_eval : array
        Точки времени для оценки
    description : str
        Описание сценария
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 8000)
    
    # Начальные концентрации веществ
    initial_conc = {
        'tox': 0.0,
        'bilirubin': 0.5,
        'ammonia': 0.3,
        'albumin': 4.5,
        'glucose': 5.0,
        'oxygen': 0.15
    }
    
    blood_params = {
        'initial_concentrations': initial_conc,
        'V0': 5000.0
    }
    
    # Параметры сердца в зависимости от сценария
    heart_params = {}
    if vsd_resistance != np.inf:
        # При ДМЖП возможна компенсаторная тахикардия
        heart_params = {'hr': 75}  # небольшое увеличение ЧСС
    
    # Создание модели
    model = WholeBodyModel(
        baroreflex_params={'P_set': 90, 'HR_base': 70, 'gain': 0.01, 'tau': 2.0},
        blood_params=blood_params,
        vsd_resistance=vsd_resistance,
        flow_dependent_lungs=flow_dependent_lungs,
        lungs_params={'flow_dependent_resistance': flow_dependent_lungs,
                      'flow_sensitivity': 0.08},
        heart_params=heart_params,
        R_sys_peripheral=None, # посчитается под MAP 85
        target_MAP=85.0, target_CO=83.0,
        C_sys_art=2.0,
        C_sys_ven=12.0,
        C_pul_ven=5.0
    )

    # проверка критерия - СРЕДНЕЕ за 1 кардиоцикл, а не мгновенное в t=0
    # y0 = model.get_initial_state()
    y0 = model.calibrate_initial_state(t_calib=10)
    T_cycle = 60.0 / model.heart.hr_base  # 0.857с при HR=70
    t_samples = np.linspace(0, T_cycle, 25) # 25 точек за цикл
    P_sa_vals = []
    Q_aortic_vals = []
    for ti in t_samples:
        out_i = model.compute_outputs(ti, y0)
        P_sa_vals.append(out_i['P_sa'])
        Q_aortic_vals.append(out_i['Q_aortic'])
    
    P_sa_mean = np.mean(P_sa_vals)
    # CO - среднее по циклу, а не пиковое
    CO_mean = np.mean(Q_aortic_vals)  # или np.trapz(Q)/T для ударного объема
    # или ударный объем * ЧСС
    # SV = np.trapz(Q_aortic_vals, t_samples)
    # CO_mean = SV * (60/T_cycle)

    R_total_mean = P_sa_mean / max(CO_mean, 1e-6)
    print(f"  [CHECK] R_total={R_total_mean:.2f} MAP={P_sa_mean:.0f} CO={CO_mean:.0f} (цель 85±5 и 80±10)")
    
    print(f"  Симуляция {label}...", end=" ", flush=True)
    sol = model.simulate(t_span, t_eval, method='RK45', rtol=1e-5, atol=1e-7)
    # sol = model.simulate(t_span, t_eval, method='BDF', rtol=1e-6)
    print(f"завершена за {len(sol.t)} шагов")
    
    # Сбор выходных переменных
    outputs = []
    for i, ti in enumerate(sol.t):
        out = model.compute_outputs(ti, sol.y[:, i])
        outputs.append(out)
    
    data = {key: np.array([out[key] for out in outputs]) for key in outputs[0].keys()}
    data['t'] = sol.t
    data['description'] = description
    
    # Очистка данных от inf и NaN
    for key in data:
        if isinstance(data[key], np.ndarray):
            # Заменяем inf и NaN на ближайшие валидные значения
            if np.any(~np.isfinite(data[key])):
                # Находим валидные значения
                valid_mask = np.isfinite(data[key])
                if np.any(valid_mask):
                    # Интерполяция для заполнения пропусков
                    indices = np.arange(len(data[key]))
                    data[key][~valid_mask] = np.interp(
                        indices[~valid_mask],
                        indices[valid_mask],
                        data[key][valid_mask]
                    )
                else:
                    # Если нет валидных значений, заполняем нулями
                    data[key] = np.zeros_like(data[key])
            
            # Для Qp_Qs, если есть нули, заменяем их на 1 (норма)
            if key == 'Qp_Qs':
                data[key] = np.where(data[key] == 0, 1.0, data[key])
                data[key] = np.clip(data[key], 0.5, 3.0)  # Ограничиваем разумными пределами
    
    return data, color, label

def analyze_steady_state(data, t_start=200):
    """Анализ установившегося режима"""
    mask = data['t'] >= t_start
    results = {}
    
    metrics = ['P_sa', 'P_pa', 'Q_aortic', 'Q_pulmonary', 'Qp_Qs', 
               'Q_vsd', 'V_lv', 'V_rv', 'V_blood', 'GFR', 'Q_brain',
               'C_bilirubin_blood', 'C_ammonia_blood', 'C_albumin_blood']
    
    for metric in metrics:
        if metric in data:
            values = data[metric][mask]
            # Очистка от inf/NaN
            values = values[np.isfinite(values)]
            if len(values) > 0:
                results[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
    
    # Дополнительные расчётные показатели
    if 'V_lv' in data:
        v_lv = data['V_lv'][mask]
        v_lv = v_lv[np.isfinite(v_lv)]
        if len(v_lv) > 0:
            results['EDV_LV'] = {'mean': np.max(v_lv)}
            results['ESV_LV'] = {'mean': np.min(v_lv)}
            if results['EDV_LV']['mean'] > 0:
                results['EF_LV'] = {'mean': (results['EDV_LV']['mean'] - results['ESV_LV']['mean']) / results['EDV_LV']['mean'] * 100}
    
    return results

def plot_enhanced_comparison(results_dict):
    """Улучшенная визуализация сравнения"""
    
    # Определяем цвета и метки
    colors = {'Здоровый': '#2ecc71', 'Малый ДМЖП (R=5.0)': '#f39c12', 
              'Большой ДМЖП (R=1.0)': '#e74c3c'}
    labels = {'Здоровый': 'Здоровый', 'Малый ДМЖП (R=5.0)': 'Малый ДМЖП (R=5.0)', 
              'Большой ДМЖП (R=1.0)': 'Большой ДМЖП (R=1.0)'}
    
    # Создаём фигуру с сеткой 3x3
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Системное и лёгочное давление
    ax1 = fig.add_subplot(gs[0, 0])
    for name, (data, _, _) in results_dict.items():
        if 'P_sa' in data and len(data['t']) > 0:
            t = data['t'][:5000]
            ps = data['P_sa'][:5000]
            mask = np.isfinite(ps)
            if np.any(mask):
                ax1.plot(t[mask], ps[mask], color=colors[name], lw=1.5, 
                        label=f"{labels[name]} (P_sa)")
    for name, (data, _, _) in results_dict.items():
        if 'P_pa' in data and len(data['t']) > 0:
            t = data['t'][:5000]
            pp = data['P_pa'][:5000]
            mask = np.isfinite(pp)
            if np.any(mask):
                ax1.plot(t[mask], pp[mask], color=colors[name], lw=1.5, 
                        linestyle='--', alpha=0.7, 
                        label=f"{labels[name]} (P_pa)" if name == 'Здоровый' else "")
    ax1.set_ylabel('Давление (мм рт. ст.)')
    ax1.set_xlabel('Время (с)')
    ax1.set_title('Системное и лёгочное давление')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 160)
    
    # 2. Системный и лёгочный кровоток
    ax2 = fig.add_subplot(gs[0, 1])
    for name, (data, _, _) in results_dict.items():
        if 'Q_aortic' in data and len(data['t']) > 0:
            t = data['t'][:5000]
            qa = data['Q_aortic'][:5000]
            mask = np.isfinite(qa)
            if np.any(mask):
                ax2.plot(t[mask], qa[mask], color=colors[name], lw=1.5,
                        label=f"{labels[name]} (Qs)")
    for name, (data, _, _) in results_dict.items():
        if 'Q_pulmonary' in data and len(data['t']) > 0:
            t = data['t'][:5000]
            qp = data['Q_pulmonary'][:5000]
            mask = np.isfinite(qp)
            if np.any(mask):
                ax2.plot(t[mask], qp[mask], color=colors[name], lw=1.5,
                        linestyle='--', alpha=0.7)
    ax2.set_ylabel('Кровоток (мл/с)')
    ax2.set_xlabel('Время (с)')
    ax2.set_title('Системный (Qs) и лёгочный (Qp) кровоток')
    ax2.legend(loc='upper right', fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 200)
    
    # 3. Соотношение Qp/Qs
    ax3 = fig.add_subplot(gs[0, 2])
    for name, (data, _, _) in results_dict.items():
        if 'Qp_Qs' in data and len(data['t']) > 0:
            t = data['t'][:5000]
            qp_qs = data['Qp_Qs'][:5000]
            # Очистка от inf/NaN
            mask = np.isfinite(qp_qs)
            if np.any(mask):
                qp_qs_clean = qp_qs[mask]
                t_clean = t[mask]
                # Применяем сглаживание только если данных достаточно
                if len(qp_qs_clean) > 50:
                    try:
                        window = min(51, len(qp_qs_clean) // 10 * 2 + 1)
                        if window % 2 == 0:
                            window += 1
                        if window >= 3:
                            qp_qs_clean = safe_savgol_filter(qp_qs_clean, window, 3)
                    except Exception:
                        pass
                ax3.plot(t_clean, qp_qs_clean, color=colors[name], lw=2, label=labels[name])
    ax3.set_ylabel('Qp/Qs')
    ax3.set_xlabel('Время (с)')
    ax3.set_title('Соотношение лёгочного и системного кровотока')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Норма')
    ax3.axhline(y=1.5, color='orange', linestyle=':', alpha=0.5, label='Умеренный шунт')
    ax3.axhline(y=2.0, color='red', linestyle=':', alpha=0.5, label='Большой шунт')
    ax3.set_ylim(0.5, 3.0)
    
    # 4. Шунт через ДМЖП
    ax4 = fig.add_subplot(gs[1, 0])
    for name, (data, _, _) in results_dict.items():
        if 'Q_vsd' in data and len(data['t']) > 0:
            t = data['t'][:5000]
            q_vsd = data['Q_vsd'][:5000]
            mask = np.isfinite(q_vsd)
            if np.any(mask) and np.max(q_vsd[mask]) > 0:
                ax4.plot(t[mask], q_vsd[mask], color=colors[name], lw=1.5, label=labels[name])
    ax4.set_ylabel('Шунт (мл/с)')
    ax4.set_xlabel('Время (с)')
    ax4.set_title('Объём шунта через ДМЖП')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Объёмы желудочков
    ax5 = fig.add_subplot(gs[1, 1])
    for name, (data, _, _) in results_dict.items():
        if 'V_lv' in data and len(data['t']) > 0:
            t = data['t'][:5000]
            v_lv = data['V_lv'][:5000]
            mask = np.isfinite(v_lv)
            if np.any(mask):
                ax5.plot(t[mask], v_lv[mask], color=colors[name], lw=1.2, alpha=0.7)
    for name, (data, _, _) in results_dict.items():
        if 'V_rv' in data and len(data['t']) > 0:
            t = data['t'][:5000]
            v_rv = data['V_rv'][:5000]
            mask = np.isfinite(v_rv)
            if np.any(mask):
                ax5.plot(t[mask], v_rv[mask], color=colors[name], lw=1.2, 
                        linestyle='--', alpha=0.7)
    ax5.set_ylabel('Объём (мл)')
    ax5.set_xlabel('Время (с)')
    ax5.set_title('Объёмы желудочков (сплошные - ЛЖ, пунктир - ПЖ)')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 200)
    
    # 6. Объём крови
    ax6 = fig.add_subplot(gs[1, 2])
    for name, (data, _, _) in results_dict.items():
        if 'V_blood' in data and len(data['t']) > 0:
            t = data['t'][:5000]
            v_blood = data['V_blood'][:5000]
            mask = np.isfinite(v_blood)
            if np.any(mask):
                ax6.plot(t[mask], v_blood[mask], color=colors[name], lw=1.5, label=labels[name])
    ax6.set_ylabel('Объём крови (мл)')
    ax6.set_xlabel('Время (с)')
    ax6.set_title('Волемический статус')
    ax6.legend(loc='lower left', fontsize=7)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(4500, 5500)
    
    # 7. Мозговой кровоток
    ax7 = fig.add_subplot(gs[2, 0])
    for name, (data, _, _) in results_dict.items():
        if 'Q_brain' in data and len(data['t']) > 0:
            t = data['t'][:5000]
            q_brain = data['Q_brain'][:5000]
            mask = np.isfinite(q_brain)
            if np.any(mask):
                ax7.plot(t[mask], q_brain[mask], color=colors[name], lw=1.5, label=labels[name])
    ax7.set_ylabel('Мозговой кровоток (мл/с)')
    ax7.set_xlabel('Время (с)')
    ax7.set_title('Церебральная гемодинамика')
    ax7.legend(loc='upper right', fontsize=7)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0, 100)
    
    # 8. Потребление кислорода мозгом
    ax8 = fig.add_subplot(gs[2, 1])
    for name, (data, _, _) in results_dict.items():
        if 'O2_consumption' in data and len(data['t']) > 0:
            t = data['t'][:5000]
            o2 = data['O2_consumption'][:5000]
            mask = np.isfinite(o2)
            if np.any(mask):
                ax8.plot(t[mask], o2[mask], color=colors[name], lw=1.5, label=labels[name])
    ax8.set_ylabel('Потребление O₂ (у.е./с)')
    ax8.set_xlabel('Время (с)')
    ax8.set_title('Метаболизм мозга')
    ax8.legend(loc='upper right', fontsize=7)
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 30)
    
    # 9. Сводная статистика
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('tight')
    ax9.axis('off')
    
    # Создаём информационную таблицу
    table_data = [['Показатель', 'Здоровый', 'Малый ДМЖП', 'Большой ДМЖП']]
    
    steady_metrics = [
        ('P_sa', 'АД сист., мм рт.ст.', '{:.0f}'),
        ('P_pa', 'АД лёг., мм рт.ст.', '{:.0f}'),
        ('Qp_Qs', 'Qp/Qs', '{:.2f}'),
        ('V_lv', 'Объём ЛЖ, мл', '{:.0f}'),
        ('V_rv', 'Объём ПЖ, мл', '{:.0f}'),
        ('V_blood', 'Объём крови, мл', '{:.0f}')
    ]
    
    for metric, label, fmt in steady_metrics:
        row = [label]
        for name in ['Здоровый', 'Малый ДМЖП (R=5.0)', 'Большой ДМЖП (R=1.0)']:
            if name in results_dict:
                data, _, _ = results_dict[name]
                if metric in data:
                    mask = (data['t'] >= 200) & np.isfinite(data[metric])
                    values = data[metric][mask]
                    if len(values) > 0:
                        val = np.mean(values)
                        row.append(fmt.format(val))
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            else:
                row.append('N/A')
        table_data.append(row)
    
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Цветовая кодировка строк таблицы (исправленная версия)
    n_rows = len(table_data)
    for i in range(1, n_rows):
        try:
            cell = table[(i, 0)]
            cell.set_facecolor('#f0f0f0')
        except KeyError:
            continue
    
    ax9.set_title('Сводка установившихся значений', fontsize=10, fontweight='bold')
    
    fig.suptitle('Сравнение гемодинамики: Здоровый человек vs ДМЖП', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hemodynamics_comparison_enhanced.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_shunt_effect_analysis(results_dict):
    """Анализ влияния размера шунта на гемодинамику"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Анализ влияния размера ДМЖП на гемодинамику', 
                 fontsize=14, fontweight='bold')
    
    colors = {'Здоровый': '#2ecc71', 'Малый ДМЖП (R=5.0)': '#f39c12', 
              'Большой ДМЖП (R=1.0)': '#e74c3c'}
    
    t_start = 200
    
    # 1. Зависимость Qp/Qs от времени для разных размеров шунта
    ax = axes[0, 0]
    for name, (data, _, _) in results_dict.items():
        if 'Qp_Qs' in data:
            mask = (data['t'] >= t_start) & np.isfinite(data['Qp_Qs'])
            if np.any(mask):
                t_steady = data['t'][mask]
                qp_qs = data['Qp_Qs'][mask]
                # Ограничиваем количество точек для производительности
                step = max(1, len(t_steady) // 500)
                ax.plot(t_steady[::step], qp_qs[::step], color=colors.get(name, 'gray'), 
                       lw=2, label=name)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Qp/Qs')
    ax.set_title('Динамика Qp/Qs в установившемся режиме')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 3.0)
    
    # 2. Баланс давлений
    ax = axes[0, 1]
    x = np.arange(2)
    width = 0.25
    for i, name in enumerate(['Здоровый', 'Малый ДМЖП (R=5.0)', 'Большой ДМЖП (R=1.0)']):
        if name in results_dict:
            data, _, _ = results_dict[name]
            if 'P_sa' in data and 'P_pa' in data:
                mask = (data['t'] >= t_start) & np.isfinite(data['P_sa']) & np.isfinite(data['P_pa'])
                if np.any(mask):
                    ps = np.mean(data['P_sa'][mask])
                    pp = np.mean(data['P_pa'][mask])
                    ax.bar(x[0] + i*width, ps, width, label=f'{name} (P_sa)', 
                           color=colors.get(name, 'gray'), alpha=0.7)
                    ax.bar(x[1] + i*width, pp, width, color=colors.get(name, 'gray'), 
                           alpha=0.4, hatch='/')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Системное', 'Лёгочное'])
    ax.set_ylabel('Давление (мм рт. ст.)')
    ax.set_title('Сравнение давлений')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Корреляция шунт-Qp/Qs
    ax = axes[0, 2]
    shunt_sizes = []
    qp_qs_values = []
    for name, (data, _, _) in results_dict.items():
        if 'Q_vsd' in data and 'Qp_Qs' in data:
            mask = (data['t'] >= t_start) & np.isfinite(data['Q_vsd']) & np.isfinite(data['Qp_Qs'])
            if np.any(mask):
                shunt_mean = np.mean(data['Q_vsd'][mask])
                qp_qs_mean = np.mean(data['Qp_Qs'][mask])
                if shunt_mean > 0:
                    shunt_sizes.append(shunt_mean)
                    qp_qs_values.append(qp_qs_mean)
                    ax.scatter(shunt_mean, qp_qs_mean, s=100, c=colors.get(name, 'gray'), 
                              marker='o', label=name, edgecolor='black', linewidth=1.5)
    if len(shunt_sizes) > 1:
        z = np.polyfit(shunt_sizes, qp_qs_values, 1)
        p = np.poly1d(z)
        x_line = np.array([0, max(shunt_sizes)])
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5,
               label=f'Тренд: Qp/Qs = {z[0]:.2f}·Q_vsd + {z[1]:.2f}')
    ax.set_xlabel('Средний шунт VSD (мл/с)')
    ax.set_ylabel('Qp/Qs')
    ax.set_title('Корреляция: размер шунта → Qp/Qs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Изменение объёма крови
    ax = axes[1, 0]
    for name, (data, _, _) in results_dict.items():
        if 'V_blood' in data:
            mask = (data['t'] >= t_start) & np.isfinite(data['V_blood'])
            if np.any(mask):
                t_steady = data['t'][mask]
                v_blood = data['V_blood'][mask]
                step = max(1, len(t_steady) // 500)
                ax.plot(t_steady[::step], v_blood[::step], color=colors.get(name, 'gray'), 
                       lw=2, label=name)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Объём крови (мл)')
    ax.set_title('Изменение объёма крови')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(4800, 5200)
    
    # 5. Функция почек
    ax = axes[1, 1]
    for name, (data, _, _) in results_dict.items():
        if 'GFR' in data:
            mask = (data['t'] >= t_start) & np.isfinite(data['GFR'])
            if np.any(mask):
                t_steady = data['t'][mask]
                gfr = data['GFR'][mask] / 60.0 # <-- /60 если GFR_base у тебя в мл/мин
                # или если GFR_base уже в мл/с, то просто gfr без деления
                step = max(1, len(t_steady) // 500)
                ax.plot(t_steady[::step], gfr[::step], color=colors.get(name, 'gray'), lw=2, label=name)

    ax.set_xlabel('Время (с)')
    ax.set_ylabel('СКФ (мл/с)')
    ax.set_title('Функция почек')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3.0) # <-- было 0.5-2.5, стало 0-3.0 чтобы видеть падение до 0
    
    # 6. Радарная диаграмма
    ax_polar = fig.add_subplot(2, 3, 6, projection='polar')
    categories = ['P_sa', 'Q_aortic', 'V_blood', 'GFR', 'Q_brain']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # Находим здоровые значения для нормализации
    healthy_values = {}
    if 'Здоровый' in results_dict:
        healthy_data, _, _ = results_dict['Здоровый']
        mask_h = (healthy_data['t'] >= t_start)
        for cat in categories:
            if cat in healthy_data:
                mask = mask_h & np.isfinite(healthy_data[cat])
                if np.any(mask):
                    healthy_values[cat] = np.mean(healthy_data[cat][mask])
    
    for name, (data, _, _) in results_dict.items():
        values = []
        for cat in categories:
            if cat in data:
                mask = (data['t'] >= t_start) & np.isfinite(data[cat])
                if np.any(mask):
                    val = np.mean(data[cat][mask])
                    # Нормализация относительно здорового
                    if name != 'Здоровый' and cat in healthy_values and healthy_values[cat] > 0:
                        val = val / healthy_values[cat]
                    values.append(val)
                else:
                    values.append(0)
            else:
                values.append(0)
        values += values[:1]
        ax_polar.plot(angles, values, 'o-', linewidth=2, label=name, 
                     color=colors.get(name, 'gray'), markersize=6)
        ax_polar.fill(angles, values, alpha=0.1, color=colors.get(name, 'gray'))
    
    ax_polar.set_xticks(angles[:-1])
    ax_polar.set_xticklabels(categories, fontsize=8)
    ax_polar.set_title('Нормированные показатели (здоровый = 1)', fontsize=10)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=7)
    ax_polar.set_ylim(0, 1.5)
    
    plt.tight_layout()
    plt.savefig('shunt_effect_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_detailed_report(results_dict):
    """Детальный отчёт о результатах симуляции"""
    print("\n" + "="*100)
    print("ДЕТАЛЬНЫЙ ОТЧЁТ О РЕЗУЛЬТАТАХ СИМУЛЯЦИИ ДМЖП")
    print("="*100)
    
    t_start = 200
    
    for name, (data, _, _) in results_dict.items():
        print(f"\n📊 СЦЕНАРИЙ: {name}")
        print("-" * 50)
        
        mask = (data['t'] >= t_start)
        
        # Гемодинамика
        if 'P_sa' in data:
            mask_ps = mask & np.isfinite(data['P_sa'])
            if np.any(mask_ps):
                ps_mean = np.mean(data['P_sa'][mask_ps])
                ps_std = np.std(data['P_sa'][mask_ps])
                print(f"  • Системное АД: {ps_mean:.1f} ± {ps_std:.1f} мм рт. ст.")
        
        if 'P_pa' in data:
            mask_pp = mask & np.isfinite(data['P_pa'])
            if np.any(mask_pp):
                pp_mean = np.mean(data['P_pa'][mask_pp])
                pp_std = np.std(data['P_pa'][mask_pp])
                print(f"  • Лёгочное АД: {pp_mean:.1f} ± {pp_std:.1f} мм рт. ст.")
        
        if 'Q_aortic' in data and 'Q_pulmonary' in data:
            mask_qs = mask & np.isfinite(data['Q_aortic'])
            mask_qp = mask & np.isfinite(data['Q_pulmonary'])
            if np.any(mask_qs):
                qs_mean = np.mean(data['Q_aortic'][mask_qs])
                print(f"  • Системный выброс: {qs_mean:.1f} мл/с")
            if np.any(mask_qp):
                qp_mean = np.mean(data['Q_pulmonary'][mask_qp])
                print(f"  • Лёгочный кровоток: {qp_mean:.1f} мл/с")
            
            if 'Qp_Qs' in data:
                mask_qp_qs = mask & np.isfinite(data['Qp_Qs'])
                if np.any(mask_qp_qs):
                    qp_qs_mean = np.mean(data['Qp_Qs'][mask_qp_qs])
                    print(f"  • Соотношение Qp/Qs: {qp_qs_mean:.2f}")
        
        if 'Q_vsd' in data:
            mask_vsd = mask & np.isfinite(data['Q_vsd'])
            if np.any(mask_vsd) and np.max(data['Q_vsd'][mask_vsd]) > 0:
                vsd_mean = np.mean(data['Q_vsd'][mask_vsd])
                print(f"  • Шунт через ДМЖП: {vsd_mean:.1f} мл/с")
        
        # Объёмы
        if 'V_lv' in data:
            mask_lv = mask & np.isfinite(data['V_lv'])
            if np.any(mask_lv):
                lv_mean = np.mean(data['V_lv'][mask_lv])
                print(f"  • Объём ЛЖ: {lv_mean:.1f} мл")
        
        if 'V_rv' in data:
            mask_rv = mask & np.isfinite(data['V_rv'])
            if np.any(mask_rv):
                rv_mean = np.mean(data['V_rv'][mask_rv])
                print(f"  • Объём ПЖ: {rv_mean:.1f} мл")
        
        if 'V_blood' in data:
            mask_vb = mask & np.isfinite(data['V_blood'])
            if np.any(mask_vb):
                vb_mean = np.mean(data['V_blood'][mask_vb])
                print(f"  • Объём крови: {vb_mean:.0f} мл")
        
        # Метаболизм
        if 'Q_brain' in data:
            mask_brain = mask & np.isfinite(data['Q_brain'])
            if np.any(mask_brain):
                brain_flow = np.mean(data['Q_brain'][mask_brain])
                print(f"  • Мозговой кровоток: {brain_flow:.2f} мл/с")
        
        if 'O2_consumption' in data:
            mask_o2 = mask & np.isfinite(data['O2_consumption'])
            if np.any(mask_o2):
                o2_cons = np.mean(data['O2_consumption'][mask_o2])
                print(f"  • Потребление O₂ мозгом: {o2_cons:.2f} у.е./с")
        
        if 'GFR' in data:
            mask_gfr = mask & np.isfinite(data['GFR'])
            if np.any(mask_gfr):
                gfr = np.mean(data['GFR'][mask_gfr])
                print(f"  • СКФ: {gfr:.2f} мл/с")
    
    # Сравнительный анализ
    print("\n" + "="*100)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("="*100)
    
    if 'Здоровый' in results_dict and 'Большой ДМЖП (R=1.0)' in results_dict:
        healthy_data, _, _ = results_dict['Здоровый']
        vsd_data, _, _ = results_dict['Большой ДМЖП (R=1.0)']
        
        mask_h = (healthy_data['t'] >= t_start)
        mask_v = (vsd_data['t'] >= t_start)
        
        print("\n📈 Изменения при большом ДМЖП (R=1.0) относительно здорового состояния:")
        
        if 'Qp_Qs' in healthy_data and 'Qp_Qs' in vsd_data:
            mask_h_qp = mask_h & np.isfinite(healthy_data['Qp_Qs'])
            mask_v_qp = mask_v & np.isfinite(vsd_data['Qp_Qs'])
            if np.any(mask_h_qp) and np.any(mask_v_qp):
                qp_qs_h = np.mean(healthy_data['Qp_Qs'][mask_h_qp])
                qp_qs_v = np.mean(vsd_data['Qp_Qs'][mask_v_qp])
                print(f"  • Qp/Qs: {qp_qs_h:.2f} → {qp_qs_v:.2f} (+{(qp_qs_v/qp_qs_h-1)*100:.0f}%)")
        
        if 'P_pa' in healthy_data and 'P_pa' in vsd_data:
            mask_h_pp = mask_h & np.isfinite(healthy_data['P_pa'])
            mask_v_pp = mask_v & np.isfinite(vsd_data['P_pa'])
            if np.any(mask_h_pp) and np.any(mask_v_pp):
                pp_h = np.mean(healthy_data['P_pa'][mask_h_pp])
                pp_v = np.mean(vsd_data['P_pa'][mask_v_pp])
                print(f"  • Лёгочное давление: {pp_h:.0f} → {pp_v:.0f} мм рт. ст. (+{pp_v-pp_h:.0f})")
        
        if 'Q_aortic' in healthy_data and 'Q_aortic' in vsd_data:
            mask_h_qs = mask_h & np.isfinite(healthy_data['Q_aortic'])
            mask_v_qs = mask_v & np.isfinite(vsd_data['Q_aortic'])
            if np.any(mask_h_qs) and np.any(mask_v_qs):
                qs_h = np.mean(healthy_data['Q_aortic'][mask_h_qs])
                qs_v = np.mean(vsd_data['Q_aortic'][mask_v_qs])
                change = (qs_v/qs_h-1)*100 if qs_h > 0 else 0
                print(f"  • Системный выброс: {qs_h:.1f} → {qs_v:.1f} мл/с ({change:.0f}%)")
        
        if 'V_rv' in healthy_data and 'V_rv' in vsd_data:
            mask_h_rv = mask_h & np.isfinite(healthy_data['V_rv'])
            mask_v_rv = mask_v & np.isfinite(vsd_data['V_rv'])
            if np.any(mask_h_rv) and np.any(mask_v_rv):
                rv_h = np.mean(healthy_data['V_rv'][mask_h_rv])
                rv_v = np.mean(vsd_data['V_rv'][mask_v_rv])
                print(f"  • Объём ПЖ: {rv_h:.0f} → {rv_v:.0f} мл (+{(rv_v/rv_h-1)*100:.0f}%)")
        
        if 'V_blood' in healthy_data and 'V_blood' in vsd_data:
            mask_h_vb = mask_h & np.isfinite(healthy_data['V_blood'])
            mask_v_vb = mask_v & np.isfinite(vsd_data['V_blood'])
            if np.any(mask_h_vb) and np.any(mask_v_vb):
                vb_h = np.mean(healthy_data['V_blood'][mask_h_vb])
                vb_v = np.mean(vsd_data['V_blood'][mask_v_vb])
                print(f"  • Объём крови: {vb_h:.0f} → {vb_v:.0f} мл ({((vb_v-vb_h)/vb_h)*100:+.0f}%)")

def main():
    """Основная функция"""
    print("="*80)
    print("СИМУЛЯЦИЯ И ВИЗУАЛИЗАЦИЯ: СРАВНЕНИЕ ГЕМОДИНАМИКИ ПРИ ДМЖП")
    print("="*80)
    
    # Определяем сценарии
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
        }
    }
    
    # Запуск симуляций
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
            t_span=(0, 300),
            description=params['description']
        )
        results[name] = (data, color, label)
        
        # Сохранение данных
        filename = f"vsd_results_{name.replace(' ', '_').replace('(', '').replace(')', '')}.npz"
        np.savez(filename, **data)
        print(f"   💾 Данные сохранены в {filename}")
    
    # Визуализация
    print("\n📊 Генерация визуализаций...")
    
    print("  • Улучшенное сравнение гемодинамики")
    plot_enhanced_comparison(results)
    
    print("  • Анализ влияния размера шунта")
    plot_shunt_effect_analysis(results)
    
    # Детальный отчёт
    print_detailed_report(results)
    
    print("\n✅ Работа завершена!")
    print("📁 Созданные файлы:")
    print("   - hemodynamics_comparison_enhanced.png")
    print("   - shunt_effect_analysis.png")
    print("   - vsd_results_*.npz (файлы данных)")

if __name__ == "__main__":
    main()
