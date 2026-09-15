import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel

model = WholeBodyModel(vsd_resistance=5.0)

# Симулируем с увеличенным fluid_intake
# model.fluid_intake_rate = 2.0  # мл/с, инфузия
model.fluid_intake_rate = 0
y0 = model.get_initial_state()
sol = model.simulate((0, 600), y0=y0, method='LSODA',
                     max_step=0.05, t_eval=np.linspace(0, 600, 3000))

print(f"{'t':>6}  {'V_blood':>9}  {'V_sv':>9}  {'P_sv':>7}  {'P_sa':>7}  {'Q_aortic':>10}")
print("-" * 60)
for tc in [0, 100, 200, 400, 600]:
    idx = int(np.argmin(np.abs(sol.t - tc)))
    out = model.compute_outputs(sol.t[idx], sol.y[:, idx])
    print(f"{tc:6d}  {out['V_blood']:9.1f}  {out['V_sv']:9.2f}  "
          f"{out['P_sv']:7.2f}  {out['P_sa']:7.2f}  {out['Q_aortic']:10.2f}")

# Симулируем с очень сильным оттоком: insensible_loss > intake
model = WholeBodyModel(
    vsd_resistance=5.0,
    fluid_intake_rate=0.0,
    insensible_loss_rate=5.0,   # 5 мл/с потерь — экстремально
)

y0 = model.get_initial_state()
sol = model.simulate((0, 600), y0=y0, method='LSODA',
                     max_step=0.05, t_eval=np.linspace(0, 600, 3000))

V_sv_arr = sol.y[model.idx['sys_ven']][0]
print(f"V_sv: min = {V_sv_arr.min():.2f}, max = {V_sv_arr.max():.2f}")
print(f"V0_sv = {model.sys_ven.V0:.2f}")
print(f"0.5·V0 = {0.5 * model.sys_ven.V0:.2f}")

# Проверка: V_sv не должен уйти ниже 0.5·V0
if V_sv_arr.min() < 0.5 * model.sys_ven.V0 - 1e-3:
    print("❌ V_sv ушёл ниже мягкого пола")
else:
    print("✅ V_sv удержан в допустимом диапазоне")    