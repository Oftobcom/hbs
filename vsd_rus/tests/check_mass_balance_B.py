import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel

# # --- Сценарий 1: инфузия 2 мл/с в течение 600 с ---
# model = WholeBodyModel(
#     vsd_resistance=5.0,
#     fluid_intake_rate=2.0,
# )
# --- Сценарий 2: кровопотеря 1 мл/с в течение 600 с ---
model = WholeBodyModel(
    vsd_resistance=5.0,
    insensible_loss_rate=1.0,    # 1 мл/с потерь
)
y0 = model.get_initial_state()
sol = model.simulate((0, 600), y0=y0, method='LSODA',
                     max_step=0.05, t_eval=np.linspace(0, 600, 3000))

print(f"{'t':>5}  {'V_blood':>9}  {'V_sv':>9}  {'V_target':>9}  {'V_sv/Vb':>9}  {'P_sv':>7}  {'P_sa':>7}")
print("-" * 70)
for tc in [0, 100, 200, 400, 600]:
    idx = int(np.argmin(np.abs(sol.t - tc)))
    out = model.compute_outputs(sol.t[idx], sol.y[:, idx])
    V_target = 0.5 * out['V_blood']
    frac = out['V_sv'] / max(out['V_blood'], 1e-6)
    print(f"{tc:5d}  {out['V_blood']:9.1f}  {out['V_sv']:9.1f}  "
          f"{V_target:9.1f}  {frac:9.3f}  {out['P_sv']:7.2f}  {out['P_sa']:7.2f}")