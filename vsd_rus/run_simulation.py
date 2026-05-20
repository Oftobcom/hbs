#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для сравнения здорового человека и пациентов с дефектом межжелудочковой перегородки (ДМЖП).
Запускает три сценария: здоровый, малый ДМЖП, большой ДМЖП.
Строит графики и выводит таблицу установившихся значений.
"""
import numpy as np
import matplotlib.pyplot as plt
from whole_body import WholeBodyModel

def simulate_scenario(vsd_resistance, flow_dependent_lungs, label, color, t_span=(0, 200), t_eval=None):
    """Запускает симуляцию для заданного сопротивления ДМЖП."""
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 5000)

    # Здоровые начальные концентрации (одинаковы для всех)
    initial_conc = {
        'tox': 0.0,
        'bilirubin': 0.2,
        'ammonia': 0.5,
        'albumin': 4.0
    }
    blood_params = {'initial_concentrations': initial_conc}

    model = WholeBodyModel(
        blood_params=blood_params,
        vsd_resistance=vsd_resistance,
        flow_dependent_lungs=flow_dependent_lungs,
        lungs_params={'flow_dependent_resistance': flow_dependent_lungs}
    )
    sol = model.simulate(t_span, t_eval)

    # Сбор выходных переменных
    outputs = []
    for i, ti in enumerate(sol.t):
        out = model.compute_outputs(ti, sol.y[:, i])
        outputs.append(out)

    data = {key: np.array([out[key] for out in outputs]) for key in outputs[0].keys()}
    data['t'] = sol.t
    return data, color, label

def plot_comparison(results_dict):
    """Строит сравнительные графики для нескольких сценариев."""
    plot_items = [
        ('P_sa', 'Системное АД (мм рт. ст.)'),
        ('P_pa', 'Лёгочное АД (мм рт. ст.)'),
        ('Q_aortic', 'Системный выброс (мл/с)'),
        ('Q_pulmonary', 'Лёгочный кровоток (мл/с)'),
        ('Qp_Qs', 'Соотношение Qp/Qs'),
        ('Q_vsd', 'Шунт через ДМЖП (мл/с)'),
        ('V_lv', 'Объём левого желудочка (мл)'),
        ('V_rv', 'Объём правого желудочка (мл)'),
        ('V_blood', 'Объём крови (мл)'),
        ('C_bilirubin_blood', 'Билирубин крови (у.е./мл)'),
        ('C_ammonia_blood', 'Аммиак крови (у.е./мл)'),
        ('C_albumin_blood', 'Альбумин крови (у.е./мл)'),
        ('GFR', 'СКФ (мл/с)'),
        ('Q_brain', 'Мозговой кровоток (мл/с)'),
        ('O2_consumption', 'Потребление O₂ мозгом (у.е./с)'),
        ('oxygenation_index', 'Индекс оксигенации (0-1)')
    ]
    n_plots = len(plot_items)
    cols = 4
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3*rows))
    axes = axes.flatten()

    for idx, (key, ylabel) in enumerate(plot_items):
        ax = axes[idx]
        for name, (data, color, label) in results_dict.items():
            if key in data:
                ax.plot(data['t'], data[key], color=color, lw=1.5, label=label)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Время (с)')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True)
        ax.set_title(key)
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig('vsd_comparison.png', dpi=150)
    plt.show()

def print_steady_state(results_dict, t_start=150):
    """Выводит средние значения показателей в установившемся режиме."""
    print("\n" + "="*100)
    print("Сравнение установившихся значений (t > {:.0f} с)".format(t_start))
    print("="*100)

    metrics = [
        ('P_sa', 'АД сист., мм рт. ст.', '{:.1f}'),
        ('P_pa', 'АД лёг., мм рт. ст.', '{:.1f}'),
        ('Q_aortic', 'Сист. выброс, мл/с', '{:.1f}'),
        ('Q_pulmonary', 'Лёг. кровоток, мл/с', '{:.1f}'),
        ('Qp_Qs', 'Qp/Qs', '{:.2f}'),
        ('Q_vsd', 'Шунт VSD, мл/с', '{:.1f}'),
        ('V_lv', 'Объём ЛЖ, мл', '{:.1f}'),
        ('V_rv', 'Объём ПЖ, мл', '{:.1f}'),
        ('V_blood', 'Объём крови, мл', '{:.0f}'),
        ('C_bilirubin_blood', 'Билирубин, у.е./мл', '{:.3f}'),
        ('C_ammonia_blood', 'Аммиак, у.е./мл', '{:.3f}'),
        ('C_albumin_blood', 'Альбумин, у.е./мл', '{:.2f}'),
        ('GFR', 'СКФ, мл/с', '{:.2f}'),
        ('Q_brain', 'Мозг. кровоток, мл/с', '{:.2f}')
    ]

    # Заголовок
    header = "{:<25}".format("Показатель")
    for name in results_dict.keys():
        header += f" {name:>15}"
    print(header)
    print("-"*100)

    for key, label, fmt in metrics:
        row = f"{label:<25}"
        for name, (data, _, _) in results_dict.items():
            if key in data:
                mask = data['t'] >= t_start
                mean_val = np.mean(data[key][mask])
                row += f" {fmt.format(mean_val):>15}"
            else:
                row += " " + " " * 15
        print(row)
    print("="*100)

if __name__ == "__main__":
    # Определяем сценарии
    scenarios = {
        'Здоровый': {
            'vsd_resistance': np.inf,
            'flow_dependent_lungs': False,
            'color': 'green',
            'label': 'Здоровый'
        },
        'Малый ДМЖП (R=5.0)': {
            'vsd_resistance': 5.0,
            'flow_dependent_lungs': False,
            'color': 'orange',
            'label': 'Малый ДМЖП'
        },
        'Большой ДМЖП (R=1.0)': {
            'vsd_resistance': 1.0,
            'flow_dependent_lungs': True,   # включаем адаптацию лёгких
            'color': 'red',
            'label': 'Большой ДМЖП'
        }
    }

    results = {}
    for name, params in scenarios.items():
        print(f"Запуск сценария: {name} ...")
        data, color, label = simulate_scenario(
            vsd_resistance=params['vsd_resistance'],
            flow_dependent_lungs=params['flow_dependent_lungs'],
            label=params['label'],
            color=params['color'],
            t_span=(0, 200)
        )
        results[name] = (data, color, label)

    plot_comparison(results)
    print_steady_state(results, t_start=150)

    # Сохраняем данные
    for name, (data, _, _) in results.items():
        filename = f"vsd_results_{name.replace(' ', '_')}.npz"
        np.savez(filename, **data)
        print(f"Результаты сохранены в {filename}")