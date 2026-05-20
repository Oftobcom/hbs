#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Healthy human simulation – no disease, no virus, normal parameters.
"""
import numpy as np
import matplotlib.pyplot as plt
from whole_body import WholeBodyModel

def simulate_healthy(t_span=(0, 200), t_eval=None):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 5000)

    # Healthy initial concentrations
    initial_conc = {
        'tox': 0.0,
        'bilirubin': 0.2,
        'ammonia': 0.5,
        'albumin': 4.0
    }
    blood_params = {'initial_concentrations': initial_conc}

    model = WholeBodyModel(blood_params=blood_params)
    sol = model.simulate(t_span, t_eval)

    # Collect outputs
    outputs = []
    for i, ti in enumerate(sol.t):
        out = model.compute_outputs(ti, sol.y[:, i])
        outputs.append(out)

    data = {key: np.array([out[key] for out in outputs]) for key in outputs[0].keys()}
    data['t'] = sol.t
    return data

def plot_healthy(data):
    plot_items = [
        ('P_sa', 'Systemic arterial pressure (mmHg)'),
        ('P_pa', 'Pulmonary arterial pressure (mmHg)'),
        ('Q_aortic', 'Cardiac output (ml/s)'),
        ('V_lv', 'Left ventricle volume (ml)'),
        ('V_blood', 'Blood volume (ml)'),
        ('C_bilirubin_blood', 'Bilirubin concentration (a.u./ml)'),
        ('C_ammonia_blood', 'Ammonia concentration (a.u./ml)'),
        ('C_albumin_blood', 'Albumin concentration (a.u./ml)'),
        ('GFR', 'Glomerular filtration rate (ml/s)'),
        ('Q_brain', 'Cerebral blood flow (ml/s)'),
        ('O2_consumption', 'Brain O₂ consumption (a.u./s)'),
        ('oxygenation_index', 'Oxygenation index (0-1)')
    ]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    for idx, (key, ylabel) in enumerate(plot_items):
        ax = axes[idx]
        ax.plot(data['t'], data[key], 'b-', lw=1.5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax.set_title(key)
    for idx in range(len(plot_items), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig('healthy_simulation.png', dpi=150)
    plt.show()

def print_steady_state(data, t_start=150):
    print("\n" + "="*80)
    print("Healthy human – steady‑state values (t > {:.0f} s)".format(t_start))
    print("="*80)
    mask = data['t'] >= t_start
    metrics = [
        ('P_sa', 'mmHg', '{:.1f}'),
        ('P_pa', 'mmHg', '{:.1f}'),
        ('Q_aortic', 'ml/s', '{:.1f}'),
        ('V_lv', 'ml', '{:.1f}'),
        ('V_blood', 'ml', '{:.0f}'),
        ('C_bilirubin_blood', 'a.u./ml', '{:.3f}'),
        ('C_ammonia_blood', 'a.u./ml', '{:.3f}'),
        ('C_albumin_blood', 'a.u./ml', '{:.2f}'),
        ('GFR', 'ml/s', '{:.2f}'),
        ('Q_brain', 'ml/s', '{:.2f}')
    ]
    for key, unit, fmt in metrics:
        mean_val = np.mean(data[key][mask])
        print(f"{key:20} = {fmt.format(mean_val)} {unit}")

if __name__ == "__main__":
    print("Running healthy human simulation...")
    data = simulate_healthy(t_span=(0, 200))
    plot_healthy(data)
    print_steady_state(data)
    np.savez('healthy_results.npz', **data)
    print("\nResults saved to healthy_results.npz")