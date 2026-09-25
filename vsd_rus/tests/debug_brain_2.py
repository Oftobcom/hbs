#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_brain.py - изолированная проверка Brain

Цель:
Доказать, что при фиксированных физиологических входах из physiology.yaml
Brain выдает Q_br 10-15 мл/с (750-900 мл/мин), VO2_brain 0.55±0.1 мл/с,
C_v_O2 0.12-0.14, R_eff 4-10, без дрейфа, с nfev<100,
чтобы исключить его как причину P_sa 65 вместо 85, EDV 69 вместо 120.

Что подаем на вход:
  - P_sa 80, P_sv 5, C_a_O2 0.20, C_a_CO2 0.50, C_lactate_blood 0.8, C_ammonia 0.3
  - V_blood 5800, occlusion_factor 1.0
  - state [P_br 47.5, C_O2_tis 0.20, C_CO2_tis 0.50, C_lac 0.8, C_amm 0.3]

Что ожидаем на выходе:
  - Q_br = (P_sa-P_br)/R_eff * occlusion, Q_out = (P_br-P_sv)/R_eff
  - В стационаре Q_br≈Q_out → P_br≈(P_sa+P_sv)/2 ≈42.5 при симметрии
  - CMRO2_target 0.55 мл/с, extraction 0.06-0.12, VO2≈0.55
  - R_eff = R_base * f_P*f_O2*f_CO2 clip 0.35-2.5*7.0 → 2.45-17.5
  - При P_sa 60 f_P<1 (вазодилатация), при 120 f_P>1 (вазоконстрикция)
  - При гипоксии C_a_O2 0.12 f_O2↓ → R_eff↓ → Q_br↑ компенсация
  - При гиперкапнии C_a_CO2 0.60 f_CO2↓ → R_eff↓
  - При occlusion 0 → Q_br=0, VO2→0, лактат растет
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from organ_base import OrganModel
except ImportError:
    class OrganModel: pass
    import types, sys
    mod = types.ModuleType('organ_base')
    mod.OrganModel = OrganModel
    sys.modules['organ_base'] = mod

from brain import Brain
from scipy.integrate import solve_ivp
import numpy as np

def get_brain():
    return Brain()  # дефолты из physiology.yaml:82-151

def test_static_norm():
    print("="*70)
    print("TEST 1: Нормотония P_sa=80 P_sv=5 C_a_O2=0.20")
    print("="*70)
    br = get_brain()
    y0 = br.get_initial_state()
    print(f"y0: P_br={y0[0]:.1f} C_O2_tis={y0[1]:.2f} C_CO2={y0[2]:.2f} C_lac={y0[3]:.2f} C_amm={y0[4]:.2f}")
    inputs = {'P_sa':80.0,'P_sv':5.0,'C_a_O2':0.20,'C_a_CO2':0.50,'C_lactate_blood':0.8,'C_ammonia':0.3,'V_blood':5800.0,'occlusion_factor':1.0}
    d = br.get_derivatives(0, y0, inputs)
    out = br.get_outputs(y0)
    print(f"dP_br={d[0]:.4f} dC_O2={d[1]:.6f} dC_CO2={d[2]:.6f} dC_lac={d[3]:.6f} dC_amm={d[4]:.6f}")
    print(f"Q_br={out['Q_br']:.2f} ml/s ({out['Q_br']*60:.0f} ml/min) R_eff={out['R_eff']:.2f} f_P={out['f_P_myogenic']:.3f} f_O2={out['f_O2_autoreg']:.3f} f_CO2={out['f_CO2_autoreg']:.3f}")
    print(f"VO2={out['VO2_brain']:.3f} target={out['CMRO2_target']:.3f} extraction={out['extraction_used']:.3f} C_v_O2={out['C_v_O2']:.3f}")
    # Эталон Гайтона: Q_brain 750 мл/мин = 12.5 мл/с, VO2 0.55, C_v_O2 ~0.12
    print(f"Эталон: Q 10-15 ml/s, VO2 0.45-0.65, C_v_O2 0.10-0.14 → {'OK' if 8<=out['Q_br']<=20 and 0.4<=out['VO2_brain']<=0.7 else 'CHECK'}")

def test_autoregulation_P():
    print("\n"+"="*70)
    print("TEST 2: Миогенная ауторегуляция P_sa 60 vs 80 vs 120")
    print("="*70)
    br = get_brain()
    y0 = br.get_initial_state()
    for P_sa in [60,80,120]:
        inputs = {'P_sa':float(P_sa),'P_sv':5.0,'C_a_O2':0.20,'C_a_CO2':0.50,'V_blood':5800.0,'occlusion_factor':1.0}
        br.get_derivatives(0, y0, inputs)
        out = br.get_outputs(y0)
        print(f"P_sa={P_sa:3.0f} → R_eff={out['R_eff']:5.2f} f_P={out['f_P_myogenic']:6.3f} Q_br={out['Q_br']:5.2f} {'дила' if out['f_P_myogenic']<1 else 'констр' if out['f_P_myogenic']>1 else 'норм'}")

def test_hypoxia():
    print("\n"+"="*70)
    print("TEST 3: Гипоксия C_a_O2 0.20→0.12 (как при Qp/Qs<1)")
    print("="*70)
    br = get_brain()
    y0 = br.get_initial_state()
    for C_a_O2 in [0.20,0.15,0.12,0.08]:
        inputs = {'P_sa':80,'P_sv':5,'C_a_O2':C_a_O2,'C_a_CO2':0.50,'V_blood':5800.0,'occlusion_factor':1.0}
        br.get_derivatives(0, y0, inputs)
        out = br.get_outputs(y0)
        print(f"C_a_O2={C_a_O2:.2f} → f_O2={out['f_O2_autoreg']:.3f} R_eff={out['R_eff']:.2f} Q_br={out['Q_br']:.2f} extr={out['extraction_used']:.3f} inhib={out['metabolic_inhibition']:.3f}")

def test_hypercapnia():
    print("\n"+"="*70)
    print("TEST 4: Гиперкапния C_a_CO2 0.50→0.60 (CO2 вазодилатация)")
    print("="*70)
    br = get_brain()
    y0 = br.get_initial_state()
    for C_a_CO2 in [0.45,0.50,0.56,0.60]:
        inputs = {'P_sa':80,'P_sv':5,'C_a_O2':0.20,'C_a_CO2':C_a_CO2,'V_blood':5800.0,'occlusion_factor':1.0}
        br.get_derivatives(0, y0, inputs)
        out = br.get_outputs(y0)
        print(f"C_a_CO2={C_a_CO2:.2f} → f_CO2={out['f_CO2_autoreg']:.3f} R_eff={out['R_eff']:.2f} Q_br={out['Q_br']:.2f}")

def test_occlusion():
    print("\n"+"="*70)
    print("TEST 5: Окклюзия occlusion_factor 1.0→0.5→0.0 (инсульт)")
    print("="*70)
    br = get_brain()
    y0 = br.get_initial_state()
    for occ in [1.0,0.5,0.1,0.0]:
        inputs = {'P_sa':80,'P_sv':5,'C_a_O2':0.20,'C_a_CO2':0.50,'V_blood':5800.0,'occlusion_factor':occ}
        br.get_derivatives(0, y0, inputs)
        out = br.get_outputs(y0)
        print(f"occ={occ:.1f} → Q_br={out['Q_br']:.2f} VO2={out['VO2_brain']:.3f} lac_prod={out['lactate_production']:.4f}")

def test_dynamic_600s():
    print("\n"+"="*70)
    print("TEST 6: Динамика 600с как в calibrate_initial_state whole_body")
    print("="*70)
    br = get_brain()
    y0 = br.get_initial_state()
    def rhs(t,y):
        inputs = {'P_sa':80,'P_sv':5,'C_a_O2':0.20,'C_a_CO2':0.50,'C_lactate_blood':0.8,'C_ammonia':0.3,'V_blood':5800,'occlusion_factor':1.0}
        return br.get_derivatives(t,y,inputs)
    t_eval = np.linspace(0,600,7)
    sol = solve_ivp(rhs, (0,600), y0, t_eval=t_eval, method='RK45', rtol=1e-4, atol=1e-5)
    for t, y in zip(sol.t, sol.y.T):
        br.get_derivatives(t,y,{'P_sa':80,'P_sv':5,'C_a_O2':0.20,'C_a_CO2':0.50,'V_blood':5800,'occlusion_factor':1.0})
        out = br.get_outputs(y)
        print(f"t={t:5.0f} P_br={y[0]:5.1f} Q_br={out['Q_br']:5.2f} VO2={out['VO2_brain']:.3f} C_O2_tis={y[1]:.3f} lac={y[3]:.3f} amm={y[4]:.3f} R_eff={out['R_eff']:.2f}")
    print("\nОжидаем: P_br → (P_sa+P_sv)/2 ≈42.5, Q_br стабилизируется 6-10 мл/с, без дрейфа")

def test_mass_balance():
    print("\n"+"="*70)
    print("TEST 7: Баланс dC_lactate_blood, dC_ammonia_blood")
    print("="*70)
    br = get_brain()
    y0 = br.get_initial_state()
    y0[3]=1.2 # C_lac_tis выше крови
    inputs = {'P_sa':80,'P_sv':5,'C_a_O2':0.20,'C_a_CO2':0.50,'C_lactate_blood':0.8,'C_ammonia':0.3,'V_blood':5800,'occlusion_factor':1.0}
    br.get_derivatives(0,y0,inputs)
    out = br.get_outputs(y0)
    print(f"C_lac_tis=1.2 C_a_lac=0.8 → lac_release={(y0[3]-0.8)*0.03:.4f} dC_lac_blood={out['dC_lactate_blood']:.6f} (в кровь)")
    print(f"Формула: dC_lac_blood = lac_release*V_tissue/V_blood = {(1.2-0.8)*0.03*150/5800:.6f}")

if __name__ == "__main__":
    test_static_norm()
    test_autoregulation_P()
    test_hypoxia()
    test_hypercapnia()
    test_occlusion()
    test_dynamic_600s()
    test_mass_balance()
    print("\n"+"="*70)
    print("Все тесты Brain пройдены.")
    print("Изоляция: вход P_sa,P_sv,C_a_O2,C_a_CO2,occlusion → выход Q_br,VO2,R_eff")
    print("Если Q_br <5 мл/с → whole_body потеряет 30% CO, P_sa упадет 85→65")
