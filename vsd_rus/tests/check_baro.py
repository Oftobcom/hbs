# tests/check_baro.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel

model = WholeBodyModel(vsd_resistance=5.0)
# y0 = model.get_initial_state()
y0 = model.calibrate_initial_state(t_calib=60, p_sa_lo=20.0)

sol = model.simulate((0, 400), y0=y0, method='LSODA',
                     max_step=0.05, t_eval=np.linspace(0, 400, 3000))

# Усреднение по последнему циклу
HR_end = model.compute_outputs(sol.t[-1], sol.y[:, -1])['HR']
T = 60.0 / HR_end
# mask = sol.t > (sol.t[-1] - T)
mask = sol.t > (sol.t[-1] - 5 * T)

# keys = ['P_sa', 'Q_aortic', 'Q_art_out', 'HR', 'V_lv', 'V_rv']
keys = ['P_sa','P_sv','P_pv','P_pa','Q_aortic','Q_art_out','Q_ven_in','Q_ven_out','Q_sv_to_ra','Q_pv_to_la','V_blood','HR','V_lv']
means = {k: [] for k in keys}
for i in np.where(mask)[0]:
    out = model.compute_outputs(sol.t[i], sol.y[:, i])
    for k in keys:
        means[k].append(out[k])

print(f"HR={HR_end:.1f}, T={T:.3f} с")
for k in keys:
    print(f"  {k:12s} = {np.mean(means[k]):8.2f}")

# SV = Q_aortic / (HR/60)
sv = np.mean(means['Q_aortic']) / (HR_end / 60.0)
print(f"  SV           = {sv:8.2f} мл")