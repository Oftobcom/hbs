import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel

# # --- Сценарий 1: инфузия 2 мл/с в течение 600 с ---
# model = WholeBodyModel(
#     heart_params={'R_vsd': 5.0},
#     fluid_intake_rate=2.0,
# )

# --- Сценарий 2: кровопотеря 1 мл/с в течение 600 с ---
# HR_base согласован с run_simulation.py для VSD-сценариев (75, а не 70):
# значение уходит и в heart.hr_base, и в baroreflex.HR_base, чтобы
# hr_factor = HR / HR_base был согласован между органами.
HR_base = 75
model = WholeBodyModel(
    heart_params={'hr': HR_base, 'R_vsd': 5.0},
    baroreflex_params={'P_set': 80.0, 'HR_base': HR_base},
)
y0 = model.calibrate_initial_state(t_calib=800.0)
model.insensible_loss_rate = 1.0    # включаем ПОСЛЕ калибровки
sol = model.simulate((0, 600), y0=y0, method='LSODA',
                     max_step=0.1, t_eval=np.arange(0.0, 600.005, 0.1))

print(f"{'t':>5}  {'V_blood':>9}  {'V_sv':>9}  {'V_target':>9}  {'V_sv/Vb':>9}  {'P_sv':>7}  {'P_sa':>7}")
print("-" * 70)
for tc in (0, 100, 200, 400, 600):
    idx = int(np.argmin(np.abs(sol.t - tc)))
    out = model.compute_outputs(sol.t[idx], sol.y[:, idx])
    V_target = 0.5 * out['V_blood']
    frac = out['V_sv'] / max(out['V_blood'], 1e-6)
    print(f"{tc:5d}  {out['V_blood']:9.1f}  {out['V_sv']:9.1f}  "
          f"{V_target:9.1f}  {frac:9.3f}  {out['P_sv']:7.2f}  {out['P_sa']:7.2f}")

# Диагностика разрыва pull-to-target при постоянном оттоке
last = model.compute_outputs(sol.t[-1], sol.y[:, -1])
dV = last['V_sv'] - 0.5 * last['V_blood']
print(f"\nV_sv − 0.5·V_blood = {dV:+.2f} мл "
        f"(ожидается ≈ +0.5·tau_target·|dV/dt| = "
        f"{0.5 * 300.0 * 1.0:+.0f} мл)")