import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel
from utils import HR_base


model = WholeBodyModel(
    heart_params={'hr': HR_base, 'R_vsd': 5.0},
    baroreflex_params={'P_set': 80.0, 'HR_base': HR_base},
)
y0 = model.calibrate_initial_state(t_calib=800)
sol = model.simulate((0, 600), y0=y0, method='LSODA', max_step=0.07,
                        t_eval=np.arange(0.0, 600.005, 0.05))


out = model.compute_outputs(sol.t[-1], sol.y[:, -1])

# --- Объёмы Windkessel-компартментов ---
V_sa = model.sys_art.C * out['P_sa']
V_sv = model.sys_ven.C * out['P_sv']
V_pv = model.pul_ven.C * out['P_pv']

# --- Объёмы камер сердца ---
V_la = out['V_la']
V_lv = out['V_lv']
V_ra = out['V_ra']
V_rv = out['V_rv']
V_heart = V_la + V_lv + V_ra + V_rv

# --- Печать в едином формате ---
print(f"V_sa    = {V_sa:6.1f} мл  (артерии)")
print(f"V_sv    = {V_sv:6.1f} мл  (системные вены)")
print(f"V_pv    = {V_pv:6.1f} мл  (лёгочные вены)")
print(f"V_la    = {V_la:6.1f} мл  (левое предсердие)")
print(f"V_lv    = {V_lv:6.1f} мл  (левый желудочек)")
print(f"V_ra    = {V_ra:6.1f} мл  (правое предсердие)")
print(f"V_rv    = {V_rv:6.1f} мл  (правый желудочек)")
print(f"V_heart = {V_heart:6.1f} мл  (все 4 камеры)")
print("-" * 50)
print(f"Сумма   = {V_sa+V_sv+V_pv+V_heart:6.1f} мл  из 5000")