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

from peripheral_tissues import PeripheralTissues
from liver import Liver

for cls in (PeripheralTissues, Liver):
    obj = cls()
    attrs = {k: v for k, v in vars(obj).items() if not k.startswith('_')}
    print(f"\n{cls.__name__}:")
    for k, v in sorted(attrs.items()):
        print(f"  {k:30s} = {v}")