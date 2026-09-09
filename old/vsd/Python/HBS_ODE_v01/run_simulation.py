#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Расширенный скрипт для симуляции системы
«сердце–лёгкие–печень–почки–ЖКТ–головной мозг»
с целью исследования влияния гепатита B на всю систему.

Варьируются параметры, связанные с вирусной нагрузкой и повреждением печени.
Строятся сравнительные графики для нескольких сценариев (здоровый, лёгкий гепатит, тяжёлый гепатит).
"""

import numpy as np
import matplotlib.pyplot as plt
from whole_body import WholeBodyModel


# ----------------------------------------------------------------------
# Функция прогона одного сценария
# ----------------------------------------------------------------------
def simulate_scenario(vsd_resistance=np.inf,
                      t_span=(0, 60),
                      t_eval=None,
                      heart_params=None,
                      liver_params=None,
                      kidney_params=None,
                      blood_params=None,
                      gitract_params=None,
                      brain_params=None):
    """
    Запускает симуляцию для заданных параметров.

    Параметры:
        vsd_resistance : float, optional
            Сопротивление дефекта межжелудочковой перегородки (по умолчанию нет дефекта).
        t_span : tuple (start, end)
        t_eval : ndarray или None
        heart_params, liver_params, kidney_params, blood_params, gitract_params, brain_params : dict
            Параметры соответствующих органов.

    Возвращает:
        sol : объект решения solve_ivp
        data : словарь с массивами выходных переменных (включая время)
    """
    # Объединяем все параметры в один вызов WholeBodyModel
    model = WholeBodyModel(
        heart_params=heart_params or {},
        lungs_params=None,  # можно добавить при необходимости
        liver_params=liver_params or {},
        kidney_params=kidney_params or {},
        blood_params=blood_params or {},
        gitract_params=gitract_params or {},
        brain_params=brain_params or {},
        vsd_resistance=vsd_resistance
    )

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 5000)

    sol = model.simulate(t_span, t_eval)

    # Определяем все ключи, которые могут быть в compute_outputs (из whole_body.py)
    # Это позволит собирать их динамически, чтобы не пропустить новые.
    # Возьмём пробный вызов для первого момента времени, чтобы узнать ключи.
    sample_out = model.compute_outputs(sol.t[0], sol.y[:, 0])
    data_keys = list(sample_out.keys())

    # Сбор выходных переменных для каждого момента времени
    n_points = len(sol.t)
    data = {key: [] for key in data_keys}

    for i in range(n_points):
        out = model.compute_outputs(sol.t[i], sol.y[:, i])
        for k in data_keys:
            data[k].append(out[k])

    # Преобразуем списки в массивы
    for k in data_keys:
        data[k] = np.array(data[k])

    data['t'] = sol.t
    return sol, data


# ----------------------------------------------------------------------
# Визуализация результатов (5x5 графиков)
# ----------------------------------------------------------------------
def plot_comparison(results_dict):
    """
    Строит сравнительные графики для нескольких сценариев.

    Параметры:
        results_dict : dict
            Ключи — названия сценариев, значения — (data, color, label)
    """
    # Выбираем наиболее интересные показатели для отображения
    # (можно расширить при необходимости)
    plot_items = [
        ('P_sa', 'Системное АД (мм рт. ст.)'),
        ('P_pa', 'Лёгочное АД (мм рт. ст.)'),
        ('P_sv', 'ЦВД (мм рт. ст.)'),
        ('V_lv', 'Объём ЛЖ (мл)'),
        ('V_rv', 'Объём ПЖ (мл)'),
        ('Q_aortic', 'Сердечный выброс (мл/с)'),
        ('Q_pulmonary', 'Лёгочный кровоток (мл/с)'),
        ('Q_renal', 'Почечный кровоток (мл/с)'),
        ('Q_liver_out', 'Печёночный венозный отток (мл/с)'),
        ('Q_gitract_out', 'Кровоток ЖКТ (мл/с)'),
        ('Q_brain', 'Мозговой кровоток (мл/с)'),
        ('absorption_water', 'Всасывание воды (мл/с)'),
        ('O2_consumption', 'Потребление O₂ мозгом (у.е./с)'),
        ('glucose_consumption', 'Потребление глюкозы мозгом (у.е./с)'),
        ('V_blood', 'Объём крови (мл)'),
        ('C_tox_blood', 'Конц. токсина (у.е./мл)'),
        ('C_bilirubin_blood', 'Билирубин крови (у.е./мл)'),
        ('C_ammonia_blood', 'Аммиак крови (у.е./мл)'),
        ('C_albumin_blood', 'Альбумин крови (у.е./мл)'),
        ('C_virus_blood', 'Вирус в крови (у.е./мл)'),
        ('liver_damage', 'Повреждение печени (0-1)'),
        ('C_virus_liver', 'Вирус в печени (у.е./мл)'),
        ('oxygenation_index', 'Индекс оксигенации (0-1)'),
        ('metabolic_inhibition', 'Метаболич. ингибирование мозга'),
        ('GFR', 'СКФ (мл/с)')
    ]

    n_plots = len(plot_items)
    cols = 5
    rows = (n_plots + cols - 1) // cols   # округление вверх

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten()

    for idx, (key, ylabel) in enumerate(plot_items):
        ax = axes[idx]
        for name, (data, color, label) in results_dict.items():
            if key in data:
                ax.plot(data['t'], data[key], color=color, label=label)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Время (с)')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True)
        ax.set_title(ylabel.split('(')[0].strip())

    # Скрываем оставшиеся подграфики
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('hepatitis_comparison.png', dpi=150)
    plt.show()


# ----------------------------------------------------------------------
# Дополнительная статистика (средние значения в установившемся режиме)
# ----------------------------------------------------------------------
def print_steady_state_stats(results_dict, t_start=40):
    """
    Выводит средние значения показателей за последний отрезок времени (от t_start до конца).
    """
    print("\n" + "=" * 100)
    print("Средние значения в установившемся режиме (t > {:.1f} с)".format(t_start))
    print("=" * 100)

    # Выберем ключевые метрики для отчёта
    metrics = [
        ('P_sa', 'P_sa, mmHg', '{:<12.2f}'),
        ('P_pa', 'P_pa, mmHg', '{:<12.2f}'),
        ('P_sv', 'P_sv, mmHg', '{:<12.2f}'),
        ('V_lv', 'V_LV, мл', '{:<12.1f}'),
        ('V_rv', 'V_RV, мл', '{:<12.1f}'),
        ('Q_aortic', 'CO, мл/с', '{:<12.2f}'),
        ('Q_renal', 'Q_renal, мл/с', '{:<12.2f}'),
        ('Q_liver_out', 'Q_liver, мл/с', '{:<12.2f}'),
        ('Q_gitract_out', 'Q_git, мл/с', '{:<12.2f}'),
        ('Q_brain', 'Q_brain, мл/с', '{:<12.2f}'),
        ('V_blood', 'V_blood, мл', '{:<12.1f}'),
        ('C_tox_blood', 'C_tox, у.е./мл', '{:<12.3f}'),
        ('C_bilirubin_blood', 'Билирубин', '{:<12.3f}'),
        ('C_ammonia_blood', 'Аммиак', '{:<12.3f}'),
        ('C_albumin_blood', 'Альбумин', '{:<12.3f}'),
        ('C_virus_blood', 'Вирус кровь', '{:<12.3f}'),
        ('C_virus_liver', 'Вирус печень', '{:<12.3f}'),
        ('liver_damage', 'Повреждение', '{:<12.3f}'),
        ('oxygenation_index', 'Оксигенация', '{:<12.3f}'),
        ('GFR', 'GFR, мл/с', '{:<12.2f}')
    ]

    # Заголовок таблицы
    header = "{:<20}".format("Показатель")
    for name in results_dict.keys():
        header += " {:>12}".format(name)
    print(header)
    print("-" * 100)

    for key, label, fmt in metrics:
        row = "{:<20}".format(label)
        for name in results_dict.keys():
            data = results_dict[name][0]
            if key in data:
                mask = data['t'] >= t_start
                if np.any(mask):
                    mean_val = np.mean(data[key][mask])
                    row += fmt.format(mean_val)
                else:
                    row += " {:>12}".format("—")
            else:
                row += " {:>12}".format("—")
        print(row)
    print("=" * 100)


# ----------------------------------------------------------------------
# Основная часть скрипта
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Определяем сценарии: здоровый, лёгкий гепатит, тяжёлый гепатит
    # Для каждого задаём параметры печени и начальные концентрации в крови.
    # В здоровом случае вирус отсутствует, повреждение 0.
    # В лёгком гепатите небольшая вирусная нагрузка, медленное прогрессирование.
    # В тяжёлом - высокая нагрузка, быстрое повреждение.

    # Общие параметры симуляции
    t_span = (0, 200)          # увеличим время, чтобы увидеть динамику повреждения
    t_eval = np.linspace(0, 200, 10000)

    # Базовые параметры для всех органов (кроме печени)
    heart_params = {'hr': 75}
    kidney_params = {}
    gitract_params = {'intake_water': 0.0, 'intake_nutrients': 0.0}
    brain_params = {}

    scenarios = {
        'healthy': {
            'color': 'green',
            'label': 'Здоровый',
            'liver_params': {
                # начальные значения вируса и повреждения
                'C_virus0': 0.0,
                'damage0': 0.0
            },
            'blood_params': {
                'initial_concentrations': {
                    'virus': 0.0,
                    'bilirubin': 0.2,    # нормальные значения
                    'ammonia': 0.5,
                    'albumin': 4.0,
                    'tox': 0.0
                }
            }
        },
        'mild_hepatitis': {
            'color': 'orange',
            'label': 'Лёгкий гепатит',
            'liver_params': {
                'C_virus0': 1.0,          # небольшая вирусная нагрузка
                'damage0': 0.0,
                'virus_growth_rate': 0.03,  # медленное размножение
                'damage_progression_rate': 0.005
            },
            'blood_params': {
                'initial_concentrations': {
                    'virus': 0.5,          # начальная вирусемия
                    'bilirubin': 0.2,
                    'ammonia': 0.5,
                    'albumin': 4.0,
                    'tox': 0.0
                }
            }
        },
        'severe_hepatitis': {
            'color': 'red',
            'label': 'Тяжёлый гепатит',
            'liver_params': {
                'C_virus0': 5.0,           # высокая нагрузка
                'damage0': 0.0,
                'virus_growth_rate': 0.08,  # быстрое размножение
                'damage_progression_rate': 0.02
            },
            'blood_params': {
                'initial_concentrations': {
                    'virus': 2.0,
                    'bilirubin': 0.3,
                    'ammonia': 0.6,
                    'albumin': 3.8,
                    'tox': 0.0
                }
            }
        }
    }

    results = {}
    for name, params in scenarios.items():
        print(f"Запуск сценария: {params['label']} ...")
        sol, data = simulate_scenario(
            vsd_resistance=np.inf,                     # без VSD
            t_span=t_span,
            t_eval=t_eval,
            heart_params=heart_params,
            liver_params=params['liver_params'],
            kidney_params=kidney_params,
            blood_params=params['blood_params'],
            gitract_params=gitract_params,
            brain_params=brain_params
        )
        results[name] = (data, params['color'], params['label'])

    plot_comparison(results)
    print_steady_state_stats(results, t_start=150)   # анализируем после выхода на режим

    # Сохранение данных в файл
    for name, (data, color, label) in results.items():
        np.savez(f"hepatitis_results_{name}.npz", **data)