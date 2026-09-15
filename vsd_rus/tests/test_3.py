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

from heart import Heart4Chambers

h = Heart4Chambers()
print(f"k_valve = {h.k_valve}")

# Проверка поведения _valve_flow
for dP in (-20.0, -1.0, -0.1, 0.0, 0.1, 1.0, 20.0):
    q = h._valve_flow(dP, h.R_valve['mitral'])
    print(f"  dP={dP:+6.2f} → Q_mitral = {q:+.6f}")