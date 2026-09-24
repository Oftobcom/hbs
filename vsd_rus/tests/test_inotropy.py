import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel

for label, h_params, br_params, per_params in [
    ('baseline k_inotropy=0.5', {}, {'gain':0.01, 'k_inotropy':0.5}, {'k_O2_autoreg':1.5}),
    ('k_inotropy=0.9 правильный', {}, {'gain':0.01, 'k_inotropy':0.9}, {'k_O2_autoreg':1.5}),
    ('E_max_lv=4.0', {'E_max_lv':4.0}, {'gain':0.01, 'k_inotropy':0.9}, {'k_O2_autoreg':1.5}),
    ('k_O2=4.0 старый', {}, {'gain':0.01, 'k_inotropy':0.9}, {'k_O2_autoreg':4.0}),
]:
    model = WholeBodyModel(
        vsd_resistance=5.0,
        heart_params=h_params,
        baroreflex_params=br_params,
        peripheral_params=per_params,
    )
    y0 = model.calibrate_initial_state(t_calib=60, p_sa_lo=20.0)
    sol = model.simulate((0, 300), y0=y0, method='LSODA',
                         max_step=0.1, t_eval=np.linspace(0, 300, 2000))
    HR = model.compute_outputs(sol.t[-1], sol.y[:, -1])['HR']
    T = 60.0 / HR
    # mask = sol.t > (sol.t[-1] - T)
    mask = sol.t > (sol.t[-1] - 5 * T)
    outs = [model.compute_outputs(sol.t[i], sol.y[:, i])
            for i in np.where(mask)[0]]
    P_sa = np.mean([o['P_sa'] for o in outs])
    Q_aor = np.mean([o['Q_aortic'] for o in outs])
    SV = Q_aor / (HR / 60)
    print(f"{label:40s}  P_sa={P_sa:6.1f}  Q={Q_aor:6.1f}  SV={SV:5.1f}  HR={HR:5.1f}")