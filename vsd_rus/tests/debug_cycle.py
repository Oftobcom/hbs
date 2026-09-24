# tests/debug_cycle.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel

model = WholeBodyModel(vsd_resistance=5.0)
# y0 = model.get_initial_state()
y0 = model.calibrate_initial_state(t_calib=60, p_sa_lo=20.0)

# Длинная симуляция до стационара
sol = model.simulate((0, 400), y0=y0, method='LSODA',
                     max_step=0.1, t_eval=np.linspace(0, 400, 3000))

# Усреднение по последнему циклу
HR_end = model.compute_outputs(sol.t[-1], sol.y[:, -1])['HR']
T_cycle = 60.0 / HR_end
mask = sol.t > (sol.t[-1] - T_cycle)

# Средние за цикл
keys = ['Q_aortic', 'Q_art_out', 'Q_peripheral', 'Q_renal',
        'Q_brain', 'Q_liver_out', 'Q_gitract_out', 'P_sa']
means = {k: [] for k in keys}
for i in np.where(mask)[0]:
    out = model.compute_outputs(sol.t[i], sol.y[:, i])
    for k in keys:
        means[k].append(out[k])

print(f"Средние за последний цикл (HR={HR_end:.1f}, T={T_cycle:.2f} с):")
for k in keys:
    print(f"  {k:15s} = {np.mean(means[k]):8.2f}")

# Баланс
Q_aort_mean = np.mean(means['Q_aortic'])
Q_out_mean  = np.mean(means['Q_art_out'])
print()
print(f"Баланс: Q_aortic = {Q_aort_mean:.2f}, Q_art_out = {Q_out_mean:.2f}, "
      f"разница = {Q_aort_mean - Q_out_mean:+.2f} мл/с")