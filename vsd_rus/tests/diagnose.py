import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel

model = WholeBodyModel(vsd_resistance=5.0)
y0 = model.get_initial_state()  # аналитический y0, без калибровки

# Один вызов derivatives из начальной точки
out = model.compute_outputs(0.0, y0)

print(f"P_sa       = {out['P_sa']:.1f}")
print(f"P_sv       = {out['P_sv']:.1f}")
print(f"P_pa       = {out['P_pa']:.1f}")
print()
print(f"Q_aortic      = {out['Q_aortic']:.2f} мл/с")
print(f"Q_pulmonary   = {out['Q_pulmonary']:.2f} мл/с")
print()
print(f"Q_peripheral  = {out['Q_peripheral']:.2f} мл/с")
print(f"Q_renal       = {out['Q_renal']:.2f} мл/с")
print(f"Q_liver_out   = {out['Q_liver_out']:.2f} мл/с")
print(f"Q_gitract_out = {out['Q_gitract_out']:.2f} мл/с")
print(f"Q_brain       = {out['Q_brain']:.2f} мл/с")
print(f"Q_art_out     = {out['Q_art_out']:.2f} мл/с")
print()
print(f"Сумма Q_art_out должна ≈ Q_aortic: {out['Q_art_out']:.1f} vs {out['Q_aortic']:.1f}")