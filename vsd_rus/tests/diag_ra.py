import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel

model = WholeBodyModel(vsd_resistance=5.0)
y0 = model.get_initial_state()
sol = model.simulate((0, 500), y0=y0, method='LSODA',
                     max_step=0.1, t_eval=np.linspace(0, 500, 3000))

# Усреднение по 5 последним циклам
out_end = model.compute_outputs(sol.t[-1], sol.y[:, -1])
T_cycle = 60.0 / out_end['HR']
mask = sol.t > (sol.t[-1] - 5 * T_cycle)

keys = ['V_la', 'V_lv', 'V_ra', 'V_rv',
        'P_la', 'P_lv', 'P_ra', 'P_rv',
        'Q_mitral', 'Q_tricuspid', 'Q_aortic', 'Q_pulmonary',
        'Q_sv_to_ra', 'Q_pv_to_la', 'Q_vsd']

print(f"HR={out_end['HR']:.1f}, T={T_cycle:.3f} с, "
      f"P_sa={out_end['P_sa']:.1f}")
print(f"{'key':>15}  {'mean':>10}")
print("-" * 30)
for k in keys:
    vals = [model.compute_outputs(sol.t[i], sol.y[:, i]).get(k)
            for i in np.where(mask)[0]]
    vals = [v for v in vals if v is not None]
    if vals:
        print(f"{k:>15}  {np.mean(vals):10.3f}")