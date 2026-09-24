# tests/check_baro.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel

model = WholeBodyModel(heart_params={'R_vsd': 5.0})
# HR_base = 75                              # как в run_simulation для VSD-сценариев
# model = WholeBodyModel(
#     heart_params={'hr': HR_base, 'R_vsd': 5.0},
#     baroreflex_params={'P_set': 80.0, 'HR_base': HR_base},
# )
# y0 = model.get_initial_state()
y0 = model.calibrate_initial_state(t_calib=800, p_sa_lo=20.0)

sol = model.simulate((0, 400), y0=y0, method='LSODA',
                     max_step=0.1, t_eval = np.linspace(0, 400, 40001))

# --- Усреднение по последним ~5 кардиоциклам ---
# Итеративно уточняем окно: сначала грубая T из HR в последней точке,
# затем HR_mean по текущей маске и пересчёт T = 60/HR_mean.
# Сходится за 2 итерации, потому что HR меняется медленно.

HR_est = float(model.compute_outputs(sol.t[-1], sol.y[:, -1])['HR'])
HR_est = max(HR_est, 20.0)                  # защита от деления на ~0
T_est  = 60.0 / HR_est

mask = sol.t > (sol.t[-1] - 5.0 * T_est)    # первичная маска

for _ in range(3):                           # 3 прохода с запасом
    idx = np.where(mask)[0]
    if idx.size < 2:
        break
    hrs = np.array([model.compute_outputs(sol.t[i], sol.y[:, i])['HR']
                    for i in idx])
    HR_mean = float(hrs.mean())
    T_mean  = 60.0 / max(HR_mean, 20.0)

    new_mask = sol.t > (sol.t[-1] - 5.0 * T_mean)
    if new_mask.sum() == mask.sum():         # окно не изменилось — сошлись
        mask = new_mask
        break
    mask = new_mask
else:
    HR_mean = HR_est                         # если не сошлось — fallback
    T_mean  = T_est

print(f"HR_mean={HR_mean:.2f} уд/мин, T_mean={T_mean:.4f} с, "
      f"окно={mask.sum()} точек")

# keys = ['P_sa', 'Q_aortic', 'Q_art_out', 'HR', 'V_lv', 'V_rv']
keys = ['P_sa','P_sv','P_pv','P_pa','Q_aortic','Q_art_out','Q_ven_in','Q_ven_out','Q_sv_to_ra','Q_pv_to_la','V_blood','HR','V_lv']
means = {k: [] for k in keys}
for i in np.where(mask)[0]:
    out = model.compute_outputs(sol.t[i], sol.y[:, i])
    for k in keys:
        means[k].append(out[k])

print(f"HR_mean={HR_mean:.1f} уд/мин, T={T_mean:.3f} с, "
      f"окно={mask.sum()} точек")

sv = np.mean(means['Q_aortic']) / (HR_mean / 60.0)
print(f"  SV = {sv:8.2f} мл")