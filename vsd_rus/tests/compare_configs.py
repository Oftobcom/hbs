import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel
from utils import HR_base

configs = [
    ('baseline (R_aort=0.15, E_max_lv=2.5)',
     dict(heart_params={'R_aortic': 0.15})),
    ('R_aort=0.20, E_max_lv=2.5',
     dict(heart_params={'R_aortic': 0.20})),
    ('R_aort=0.15, E_max_lv=4.0',
     dict(heart_params={'R_aortic': 0.15, 'E_max_lv': 4.0})),
    ('R_aort=0.15, E_max_lv=5.0',
     dict(heart_params={'R_aortic': 0.15, 'E_max_lv': 5.0})),
]

for name, kwargs in configs:
    model = WholeBodyModel(
        heart_params={'hr': HR_base, 'R_vsd': 5.0},
        baroreflex_params={'P_set': 80.0, 'HR_base': HR_base},
    )
    y0 = model.calibrate_initial_state(t_calib=800)
    sol = model.simulate((0, 600), y0=y0, method='LSODA', max_step=0.07,
                            t_eval=np.arange(0.0, 600.005, 0.05))
    HR = model.compute_outputs(sol.t[-1], sol.y[:, -1])['HR']
    T = 60.0 / HR
    mask = sol.t > (sol.t[-1] - T)
    outs = [model.compute_outputs(sol.t[i], sol.y[:, i])
            for i in np.where(mask)[0]]
    P_sa = np.mean([o['P_sa'] for o in outs])
    Q_aor = np.mean([o['Q_aortic'] for o in outs])
    SV = Q_aor / (HR / 60)
    print(f"{name:45s}  P_sa={P_sa:5.1f}  Q_aortic={Q_aor:6.1f}  SV={SV:5.1f}")