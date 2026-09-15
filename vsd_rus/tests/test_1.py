#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np

# --- Путь к корню проекта (там, где whole_body.py) ------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whole_body import WholeBodyModel

# model = WholeBodyModel(vsd_resistance=1.0)   # большой ДМЖП
model = WholeBodyModel(vsd_resistance=5.0)   # малый ДМЖП
y0 = model.calibrate_initial_state()
sol = model.simulate((0, 400), t_eval=np.linspace(0, 400, 1000),
                     y0=y0, method='LSODA', rtol=1e-5, max_step=0.05)

assert np.all(np.isfinite(sol.y)), "Решение содержит NaN/Inf!"

out = model.compute_outputs(sol.t[-1], sol.y[:, -1])
print(f"C_lactate_blood  = {out['C_lactate_blood']:.4f} мг/мл")
print(f"dC_lactate_liver = {out['dC_lactate_liver']:+.5f} мг/мл/с")
print(f"lactate_production (periph) = {out['lactate_production']:+.5f} мг/(мл·с)")
print(f"dC_lactate_periph_to_blood = {out['dC_lactate_periph_to_blood']:+.5f} мг/мл/с")
print(f"Total dC_lactate = {out['dC_lactate_periph_to_blood'] + out['dC_lactate_liver']:+.5f} мг/мл/с")
print(f"P_sa = {out['P_sa']:.1f}")   # ожидаем 75-95
print(f"Q_aortic = {out['Q_aortic']:.1f}")  # ожидаем ~80-100
print(f"P_pa = {out['P_pa']:.1f}")   # ожидаем 15-25