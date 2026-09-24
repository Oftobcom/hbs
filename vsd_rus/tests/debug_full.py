# tests/debug_full.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel

model = WholeBodyModel(vsd_resistance=5.0)
y0 = model.get_initial_state()

# Очень длинная симуляция
sol = model.simulate((0, 1000), y0=y0, method='LSODA',
                     max_step=0.1, t_eval=np.linspace(0, 1000, 5000))

# P_sa по точкам
print(f"{'t, с':>6}  {'P_sa':>7}  {'HR':>6}  {'Q_aortic':>9}")
print("-" * 35)
for t_check in [10, 30, 60, 120, 200, 300, 400, 600, 800, 1000]:
    idx = int(np.argmin(np.abs(sol.t - t_check)))
    out = model.compute_outputs(sol.t[idx], sol.y[:, idx])
    print(f"{sol.t[idx]:6.0f}  {out['P_sa']:7.1f}  {out['HR']:6.1f}  {out['Q_aortic']:9.2f}")