# tests/check_stationary.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel

model = WholeBodyModel(vsd_resistance=5.0)
y0 = model.calibrate_initial_state(t_calib=60, p_sa_lo=20.0)
sol = model.simulate((0, 600), y0=y0, method='LSODA',
                     max_step=0.05, t_eval=np.linspace(0, 600, 4000))

# Проверка стационарности по точкам
print(f"{'t, с':>6}  {'P_sa':>7}  {'P_sv':>7}  {'P_pv':>7}  {'P_pa':>7}  {'Q_aortic':>10}")
print("-" * 60)
for t_check in [100, 200, 300, 400, 500, 600]:
    idx = int(np.argmin(np.abs(sol.t - t_check)))
    out = model.compute_outputs(sol.t[idx], sol.y[:, idx])
    T = 60.0 / out['HR']
    mask_cycle = (sol.t > sol.t[idx] - T) & (sol.t <= sol.t[idx])
    q_mean = np.mean([model.compute_outputs(sol.t[i], sol.y[:, i])['Q_aortic']
                    for i in np.where(mask_cycle)[0]])
    print(f"{t_check:6d}  {out['P_sa']:7.2f}  {out['P_sv']:7.2f}  "
          f"{out['P_pv']:7.2f}  {out['P_pa']:7.2f}  {q_mean:10.2f}")

# Проверка, сдвинулись ли давления между 400 и 600
idx_400 = int(np.argmin(np.abs(sol.t - 400)))
idx_600 = int(np.argmin(np.abs(sol.t - 600)))
out_400 = model.compute_outputs(sol.t[idx_400], sol.y[:, idx_400])
out_600 = model.compute_outputs(sol.t[idx_600], sol.y[:, idx_600])

print("\nДрейф между t=400 и t=600:")
for key in ['P_sa', 'P_sv', 'P_pv', 'P_pa']:
    drift = abs(out_600[key] - out_400[key])
    flag = "OK" if drift < 1.0 else "DRIFT"
    print(f"  {key:6s}: {out_400[key]:6.2f} → {out_600[key]:6.2f}  "
          f"(Δ={drift:5.2f})  [{flag}]")