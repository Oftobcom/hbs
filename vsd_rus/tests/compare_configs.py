import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel

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
    model = WholeBodyModel(vsd_resistance=5.0, **kwargs)
    y0 = model.get_initial_state()
    sol = model.simulate((0, 300), y0=y0, method='LSODA',
                         max_step=0.05, t_eval=np.linspace(0, 300, 2000))
    HR = model.compute_outputs(sol.t[-1], sol.y[:, -1])['HR']
    T = 60.0 / HR
    mask = sol.t > (sol.t[-1] - T)
    outs = [model.compute_outputs(sol.t[i], sol.y[:, i])
            for i in np.where(mask)[0]]
    P_sa = np.mean([o['P_sa'] for o in outs])
    Q_aor = np.mean([o['Q_aortic'] for o in outs])
    SV = Q_aor / (HR / 60)
    print(f"{name:45s}  P_sa={P_sa:5.1f}  Q_aortic={Q_aor:6.1f}  SV={SV:5.1f}")