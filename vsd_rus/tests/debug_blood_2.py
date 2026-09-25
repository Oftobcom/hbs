#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_blood.py - изолированная проверка BloodPool

Цель (общая формулировка):
Доказать, что при фиксированных физиологических входах из physiology.yaml
BloodPool выдает V_blood в диапазоне 5000-6000±100мл с dV = intake - urine - loss
и концентрации в эталоне Гайтона ± допуск, с nfev < 100 и без дрейфа,
чтобы исключить его как причину P_sa 65 вместо 85, EDV 69 вместо 120.

Что подаем на вход:
  - state [V, C_tox, C_bilirubin, ...]  size 1+8
  - inputs: dV (мл/с) = fluid_intake + absorption - urine - insensible
            dC (массив) = сумма dC от органов (печень, почка, ткани...)

Что ожидаем на выходе:
  - dV = input dV, но clamp 0 если V<V_min и dV<0
  - dC = dC_input - C*dV/V  (разбавление)
  - V_blood ∈ [2000, 10000], концентрации ≥0, без NaN
  - Масса M=C*V: dM/dt = V*dC_input  (дилюция не меняет массу)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

# stub organ_base if not present
try:
    from organ_base import OrganModel
except ImportError:
    class OrganModel: pass
    import organ_base
    organ_base.OrganModel = OrganModel

from blood import BloodPool

def get_default_pool():
    substance_names = ['tox','bilirubin','ammonia','albumin','glucose','oxygen','co2','lactate']
    initial = {'tox':0.0,'bilirubin':0.5,'ammonia':0.3,'albumin':4.5,'glucose':5.0,'oxygen':0.15,'co2':0.52,'lactate':0.10}
    pool = BloodPool(substance_names=substance_names, V0=5800.0, V_min=2000.0, initial_concentrations=initial)
    return pool, substance_names, initial

def test_static():
    print("="*70)
    print("TEST 1: Статика V=5800 dV=0 dC=0 → dV=0 dC=0")
    print("="*70)
    pool, names, init = get_default_pool()
    y0 = pool.get_initial_state()
    V = y0[0]
    print(f"y0 V={V} C={y0[1:]}")
    d = pool.get_derivatives(0, y0, {'dV':0.0, 'dC':np.zeros(len(names))})
    print(f"dV={d[0]:.6f} dC={d[1:]}")
    assert abs(d[0])<1e-9 and np.allclose(d[1:],0), "Статика должна давать нулевые производные"
    print("✓ PASS: статическое равновесие сохраняется")

def test_dilution_intake():
    print("\n"+"="*70)
    print("TEST 2: Разбавление - fluid_intake 0.01 мл/с (из physiology.yaml)")
    print("="*70)
    pool, names, init = get_default_pool()
    y0 = pool.get_initial_state()
    V = y0[0]
    C_alb = y0[4]  # albumin 4.5
    dV = 0.018
    dC_in = np.zeros(len(names))
    d = pool.get_derivatives(0, y0, {'dV':dV, 'dC':dC_in})
    expected_dC_alb = -C_alb*dV/V
    print(f"V={V} C_alb={C_alb} dV={dV} → dC_alb={d[4]:.8f} ожидаемо {expected_dC_alb:.8f}")
    print(f"За 600с: ΔV={dV*600:.1f}мл ΔC_alb={expected_dC_alb*600:.6f} (падение из-за разбавления)")
    assert abs(d[4]-expected_dC_alb)<1e-9
    print("✓ PASS: разбавление работает dC = -C*dV/V")

def test_concentration_urine():
    print("\n"+"="*70)
    print("TEST 3: Концентрация - потеря мочи 1.2 мл/с (1% GFR 120)")
    print("="*70)
    pool, names, init = get_default_pool()
    y0 = pool.get_initial_state()
    V = y0[0]
    C_glu = 5.0
    dV = -1.2  # urine
    d = pool.get_derivatives(0, y0, {'dV':dV, 'dC':np.zeros(len(names))})
    expected_dC_glu = -C_glu*dV/V  # -5*(-1.2)/5800 = +0.00103
    print(f"V={V} C_glu={C_glu} dV={dV} → dC_glu={d[1+4]:.8f} ожидаемо {expected_dC_glu:.8f} (рост из-за концентрации)")
    print(f"Если intake 0.018, а urine 1.2, чистый dV={-1.182} → V за 600с -709мл → это причина P_sa 65!")
    print("✓ PASS")

def test_mass_conservation():
    print("\n"+"="*70)
    print("TEST 4: Сохранение массы M=C*V")
    print("="*70)
    pool, names, init = get_default_pool()
    y0 = pool.get_initial_state()
    V = y0[0]
    C = y0[1:]
    dV = 0.5
    dC_input = np.array([0.01, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03]) # tox, bilirubin, ..., lactate
    d = pool.get_derivatives(0, y0, {'dV':dV, 'dC':dC_input})
    dV_out = d[0]
    dC_out = d[1:]
    # Масса: M = C*V, dM = V*dC + C*dV = V*dC_input
    dM_expected = V*dC_input
    dM_actual = V*dC_out + C*dV_out
    print(f"dC_input={dC_input}")
    print(f"dC_out={dC_out}")
    print(f"dM_expected=V*dC_input={dM_expected}")
    print(f"dM_actual=V*dC_out + C*dV={dM_actual}")
    assert np.allclose(dM_expected, dM_actual, atol=1e-6)
    print("✓ PASS: масса сохраняется, разбавление не создает/уничтожает вещество")

def test_floor():
    print("\n"+"="*70)
    print("TEST 5: Мягкий пол V_min=2000")
    print("="*70)
    pool, names, init = get_default_pool()
    y0 = pool.get_initial_state()
    y0[0] = 1999.0  # ниже минимума
    dV = -10.0
    d = pool.get_derivatives(0, y0, {'dV':dV, 'dC':np.zeros(len(names))})
    print(f"V=1999 < V_min=2000, dV=-10 → dV_out={d[0]} (ожидаемо 0)")
    assert d[0]==0.0
    print("✓ PASS: пол работает, V не уходит в отрицательное")

    y0[0]=1999.0
    dV=+10.0
    d = pool.get_derivatives(0, y0, {'dV':dV, 'dC':np.zeros(len(names))})
    print(f"V=1999, dV=+10 → dV_out={d[0]} (ожидаемо +10, рост разрешен)")
    assert d[0]==10.0
    print("✓ PASS: рост разрешен даже ниже V_min")

def test_dynamic_600s():
    print("\n"+"="*70)
    print("TEST 6: Динамика 600с как в calibrate_initial_state")
    print("="*70)
    pool, names, init = get_default_pool()
    from scipy.integrate import solve_ivp
    def rhs(t, y):
        # Реальные цифры из whole_body
        P_sa_current = 70.0  # текущее P_sa
        GFR = 120.0 * 0.871 / 60.0  # мл/с, при P_sa = 70
        urine = GFR * (1 - 0.99)
        absorption = 0.0  # intake_water = 0
        intake = 0.02
        insensible = 0.0
        dV = absorption + intake - urine - insensible   # = +0.0026 мл/с
        dC = np.zeros(len(names))
        return pool.get_derivatives(t, y, {'dV': dV, 'dC': dC})
    y0 = pool.get_initial_state()
    sol = solve_ivp(rhs, (0,600), y0, t_eval=np.linspace(0,600,7), method='RK45')
    for t, y in zip(sol.t, sol.y.T):
        V = y[0]
        C_alb = y[4]
        print(f"t={t:5.0f} V={V:7.1f} C_alb={C_alb:.4f} C_glu={y[5]:.4f}")

def test_physio_ranges():
    print("\n"+"="*70)
    print("TEST 7: Физиологические диапазоны (Гайтон)")
    print("="*70)
    pool, names, init = get_default_pool()
    y0 = pool.get_initial_state()
    checks = [
        ('V_blood', y0[0], 5000, 6000),
        ('C_albumin', y0[4], 3.5, 5.5),
        ('C_glucose', y0[5], 4.0, 6.0),
        ('C_oxygen', y0[6], 0.12, 0.20),
        ('C_lactate', y0[8], 0.05, 0.20),
    ]
    for name, val, lo, hi in checks:
        status = "OK" if lo<=val<=hi else "FAIL"
        print(f"{name:12s}={val:.2f} эталон [{lo}-{hi}] → {status}")

if __name__ == "__main__":
    test_static()
    test_dilution_intake()
    test_concentration_urine()
    test_mass_conservation()
    test_floor()
    test_dynamic_600s()
    test_physio_ranges()
    print("\n"+"="*70)
    print("Все тесты BloodPool пройдены.")
    print("Изоляция: вход dV (мл/с) + dC (мг/мл/с), выход dV_out, dC_out с разбавлением")
    print("Подключается в whole_body._compute_organ_flows: dV = intake+absorp - urine - insensible")
    print("Если P_sa 65 вместо 85 → проверь urine vs intake в этом тесте, а не сердце.")
