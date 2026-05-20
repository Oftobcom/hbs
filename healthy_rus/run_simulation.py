#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для симуляции здорового человека.
Запускает модель, строит графики, выводит установившиеся значения.
"""
import numpy as np
import matplotlib.pyplot as plt
from whole_body import WholeBodyModel

def simulate_healthy(t_span=(0, 200), t_eval=None):
    """Запускает симуляцию здорового человека."""
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 5000)

    # Здоровые начальные концентрации
    initial_conc = {
        'tox': 0.0,
        'bilirubin': 0.2,
        'ammonia': 0.5,
        'albumin': 4.0
    }
    blood_params = {'initial_concentrations': initial_conc}

    model = WholeBodyModel(blood_params=blood_params)
    sol = model.simulate(t_span, t_eval)

    # Сбор выходных переменных
    outputs = []
    for i, ti in enumerate(sol.t):
        out = model.compute_outputs(ti, sol.y[:, i])
        outputs.append(out)

    data = {key: np.array([out[key] for out in outputs]) for key in outputs[0].keys()}
    data['t'] = sol.t
    return data

def plot_healthy(data):
    """Строит графики основных физиологических показателей."""
    plot_items = [
        ('P_sa', 'Системное АД (мм рт. ст.)'),
        ('P_pa', 'Лёгочное АД (мм рт. ст.)'),
        ('Q_aortic', 'Сердечный выброс (мл/с)'),
        ('V_lv', 'Объём левого желудочка (мл)'),
        ('V_blood', 'Объём крови (мл)'),
        ('C_bilirubin_blood', 'Билирубин крови (у.е./мл)'),
        ('C_ammonia_blood', 'Аммиак крови (у.е./мл)'),
        ('C_albumin_blood', 'Альбумин крови (у.е./мл)'),
        ('GFR', 'СКФ (мл/с)'),
        ('Q_brain', 'Мозговой кровоток (мл/с)'),
        ('O2_consumption', 'Потребление O₂ мозгом (у.е./с)'),
        ('oxygenation_index', 'Индекс оксигенации (0-1)')
    ]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    for idx, (key, ylabel) in enumerate(plot_items):
        ax = axes[idx]
        ax.plot(data['t'], data[key], 'b-', lw=1.5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Время (с)')
        ax.grid(True)
        ax.set_title(key)
    for idx in range(len(plot_items), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig('healthy_simulation.png', dpi=150)
    plt.show()

def print_steady_state(data, t_start=150):
    """Выводит средние значения показателей в установившемся режиме."""
    print("\n" + "="*80)
    print("Здоровый человек – установившиеся значения (t > {:.0f} с)".format(t_start))
    print("="*80)
    mask = data['t'] >= t_start
    metrics = [
        ('P_sa', 'мм рт. ст.', '{:.1f}'),
        ('P_pa', 'мм рт. ст.', '{:.1f}'),
        ('Q_aortic', 'мл/с', '{:.1f}'),
        ('V_lv', 'мл', '{:.1f}'),
        ('V_blood', 'мл', '{:.0f}'),
        ('C_bilirubin_blood', 'у.е./мл', '{:.3f}'),
        ('C_ammonia_blood', 'у.е./мл', '{:.3f}'),
        ('C_albumin_blood', 'у.е./мл', '{:.2f}'),
        ('GFR', 'мл/с', '{:.2f}'),
        ('Q_brain', 'мл/с', '{:.2f}')
    ]
    for key, unit, fmt in metrics:
        mean_val = np.mean(data[key][mask])
        print(f"{key:20} = {fmt.format(mean_val)} {unit}")

if __name__ == "__main__":
    print("Запуск симуляции здорового человека...")
    data = simulate_healthy(t_span=(0, 200))
    plot_healthy(data)
    print_steady_state(data)
    np.savez('healthy_results.npz', **data)
    print("\nРезультаты сохранены в healthy_results.npz")