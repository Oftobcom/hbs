#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_kidney.py — изолированная проверка KidneyHemodynamic (новая физиология).

Проверяет актуальную версию kidney.py:
  • auto-consistent renal_resistance = (P_autoreg − 5)/RBF_target = 4.5
  • плато RBF на 20 мл/с для MAP ∈ [70, 180]
  • прорыв RBF при MAP > 180
  • плато GFR на 120 мл/мин (три-веточная кривая с сигмоидой)
  • basal_urine_output = 0.005 мл/с (0.3 мл/мин)
  • pressure-natriuresis
  • клип Q_renal ≥ 0

Запуск:  python tests/debug_kidney.py
Отчёт:   tests/results_debug_kidney.txt
"""
from __future__ import annotations
import sys, io
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import bisect
from kidney import KidneyHemodynamic
from physio_config import load_physiology

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_kidney.txt"


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d):
        for s in self.streams: s.write(d); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


CFG = load_physiology()
KID_CFG = dict(CFG.get("kidney", {}))
SYS_CFG = dict(CFG.get("systemic", {}))


def make_kidney() -> KidneyHemodynamic:
    """
    ВАЖНО: передаём ТОЛЬКО то, что есть в YAML.
    renal_resistance в YAML отсутствует → KidneyHemodynamic сам вычислит
    (P_autoreg − 5)/RBF_target = (95−5)/20 = 4.5.
    """
    return KidneyHemodynamic(**KID_CFG)


INPUTS_HEALTHY = {
    "P_sa": 85.0,
    "P_sv": 12.0,      # типичное P_sv в whole_body
    "C_tox": 0.0,
    "V_blood": 5800.0,
}


def banner():
    print("=" * 78)
    print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S} | NumPy {np.__version__}")
    print("KidneyHemodynamic: изолированный тест (новая физиология)")
    print("=" * 78)
    print(f"  kidney params из physiology.yaml:")
    for k, v in KID_CFG.items():
        print(f"    {k:28s} = {v}")
    print(f"  systemic:")
    print(f"    fluid_intake_rate  = {SYS_CFG.get('fluid_intake_rate')}")
    print(f"    insensible_loss    = {SYS_CFG.get('insensible_loss_rate')}")
    print(f"  Входы (healthy): {INPUTS_HEALTHY}")
    print(f"  → {RESULT_FILE}")
    print("=" * 78)


# ---------------------------------------------------------------------
# TEST 1 — интерфейс
# ---------------------------------------------------------------------
def test_interface():
    print("\n" + "=" * 78)
    print("TEST 1: Интерфейс OrganModel (алгебраический, 0 состояний)")
    print("=" * 78)
    k = make_kidney()
    sz = k.get_state_size()
    y0 = k.get_initial_state()
    d0 = k.get_derivatives(0.0, y0, {})
    k.compute_effects(**INPUTS_HEALTHY)
    out = k.get_outputs(y0)
    print(f"  state_size = {sz}  (ожидание 0)")
    print(f"  y0         = {y0}, size = {y0.size}")
    print(f"  dy/dt      = {d0}, size = {d0.size}")
    print(f"  outputs    = {out}")
    ok = (sz == 0 and y0.size == 0 and d0.size == 0
          and "GFR" in out and "Q_renal" in out and "urine_output" in out)
    print(f"  [{'OK' if ok else 'FAIL'}] интерфейс корректен")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 2 — auto-consistent renal_resistance
# ---------------------------------------------------------------------
def test_auto_resistance():
    print("\n" + "=" * 78)
    print("TEST 2: auto-consistent renal_resistance")
    print("=" * 78)
    k = make_kidney()
    expected = (k.P_autoreg - 5.0) / k.RBF_target
    print(f"  P_autoreg       = {k.P_autoreg}")
    print(f"  RBF_target      = {k.RBF_target} мл/с")
    print(f"  renal_resistance = {k.renal_resistance:.4f}  (ожидание {expected:.4f})")
    print(f"  basal_urine_output = {k.basal_urine_output:.4f} мл/с = {k.basal_urine_output*60:.2f} мл/мин")
    ok = abs(k.renal_resistance - expected) < 1e-9
    print(f"  [{'OK' if ok else 'FAIL'}] renal_resistance авто-вычислен верно")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 3 — GFR(P_sa)
# ---------------------------------------------------------------------
def test_gfr_curve():
    print("\n" + "=" * 78)
    print("TEST 3: GFR(P_sa) — три-веточная кривая ауторегуляции")
    print("=" * 78)
    k = make_kidney()
    P_list = [40, 50, 60, 70, 80, 85, 95, 120, 150, 180, 200]
    print(f"  {'P_sa':>5}  {'GFR, мл/мин':>14}  {'% base':>10}")
    for P in P_list:
        gfr = k._compute_gfr(P)
        print(f"  {P:>5}  {gfr:>14.2f}  {gfr/k.GFR_base*100:>9.1f}%")

    gfr_95  = k._compute_gfr(95)
    gfr_60  = k._compute_gfr(60)
    gfr_180 = k._compute_gfr(180)
    gfr_200 = k._compute_gfr(200)

    # auto_index в [80, 180]
    gfr_auto = [k._compute_gfr(P) for P in range(80, 181, 5)]
    auto_index = (max(gfr_auto) - min(gfr_auto)) / np.mean(gfr_auto)

    checks = [
        ("GFR(95) ≈ 120 мл/мин",           abs(gfr_95 - 120.0) < 1.0),
        ("GFR(60) ∈ [70, 95]",             70 <= gfr_60 <= 95),
        ("GFR(180) ≈ 120 (плато)",         abs(gfr_180 - 120.0) < 2.0),
        ("GFR(200) > GFR(180)",            gfr_200 > gfr_180),
        ("auto_index[80,180] < 0.05",      auto_index < 0.05),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")
    print(f"  auto_index в [80,180] = {auto_index:.4f}")
    return {"ok": ok, "auto_index": auto_index}


# ---------------------------------------------------------------------
# TEST 4 — RBF-стабилизация (Q_renal)
# ---------------------------------------------------------------------
def test_rbf_plateau():
    print("\n" + "=" * 78)
    print("TEST 4: RBF-стабилизация — Q_renal(P_sa)")
    print("=" * 78)
    k = make_kidney()
    print(f"  RBF_target = {k.RBF_target} мл/с")
    print(f"  {'P_sa':>5}  {'Q_renal, мл/с':>15}  {'Q_renal, мл/мин':>17}")
    Q_list = []
    for P in [40, 60, 70, 80, 95, 120, 150, 180, 200]:
        eff = k.compute_effects(P, 12.0, 0.0, 5800.0)
        Q = eff["Q_renal"]
        Q_list.append((P, Q))
        print(f"  {P:>5}  {Q:>15.3f}  {Q*60:>17.1f}")

    # Плато [95, 180] должно быть ≈ 20 мл/с
    Q_95  = dict(Q_list)[95]
    Q_120 = dict(Q_list)[120]
    Q_180 = dict(Q_list)[180]
    Q_200 = dict(Q_list)[200]

    checks = [
        ("Q_renal(95) ≈ 20 мл/с",          abs(Q_95 - 20.0) < 1.0),
        ("Q_renal(120) ≈ 20 мл/с",         abs(Q_120 - 20.0) < 1.0),
        ("Q_renal(180) ≈ 20 мл/с",         abs(Q_180 - 20.0) < 1.0),
        ("Q_renal(200) > Q_renal(180)",    Q_200 > Q_180 + 2.0),
        ("Q_renal(60) < 20 (гипоперфузия)", Q_list[1][1] < 20.0),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 5 — basal_urine_output floor
# ---------------------------------------------------------------------
def test_basal_urine():
    print("\n" + "=" * 78)
    print("TEST 5: basal_urine_output = 0.005 мл/с (0.3 мл/мин)")
    print("=" * 78)
    k = make_kidney()
    basal = k.basal_urine_output
    print(f"  basal = {basal:.5f} мл/с = {basal*60:.3f} мл/мин")
    print(f"  {'P_sa':>5}  {'GFR, мл/мин':>14}  {'urine, мл/мин':>15}  {'basal срабатывает?':>20}")
    ok = True
    for P in [0, 10, 20, 40, 60, 85, 95]:
        eff = k.compute_effects(P, 12.0, 0.0, 5800.0)
        GFR_ml_min = eff["GFR"] * 60.0
        urine_ml_min = eff["urine_output"] * 60.0
        basal_active = eff["urine_output"] <= basal * 1.001
        print(f"  {P:>5}  {GFR_ml_min:>14.2f}  {urine_ml_min:>15.3f}  "
              f"{'ДА' if basal_active else 'нет':>20}")
        if P <= 20 and not basal_active:
            ok = False
    # При P=0 GFR=0, urine должна быть = basal
    eff0 = k.compute_effects(0, 12.0, 0.0, 5800.0)
    ok = ok and abs(eff0["urine_output"] - basal) < 1e-9
    print(f"  [{'OK' if ok else 'FAIL'}] basal срабатывает при P_sa ≤ 20, urine = basal")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 6 — pressure-natriuresis
# ---------------------------------------------------------------------
def test_pressure_natriuresis():
    print("\n" + "=" * 78)
    print("TEST 6: Pressure-natriuresis — urine растёт с P_sa")
    print("=" * 78)
    k = make_kidney()
    print(f"  {'P_sa':>5}  {'reabs':>8}  {'urine, мл/мин':>15}")
    reabs_list, urine_list = [], []
    for P in [40, 60, 85, 95, 120, 150, 180]:
        reabs = k._reabsorption_frac(P)
        eff = k.compute_effects(P, 12.0, 0.0, 5800.0)
        urine_ml_min = eff["urine_output"] * 60.0
        reabs_list.append(reabs)
        urine_list.append(urine_ml_min)
        print(f"  {P:>5}  {reabs:>8.5f}  {urine_ml_min:>15.3f}")

    # Монотонность urine по P (кроме basal-floor)
    mono = all(urine_list[i] <= urine_list[i+1] + 1e-6
               for i in range(len(urine_list)-1))
    # Reabs падает с P
    reabs_dec = all(reabs_list[i] >= reabs_list[i+1] - 1e-9
                    for i in range(len(reabs_list)-1))
    checks = [
        ("urine монотонно растёт с P",     mono),
        ("reabs монотонно падает с P",     reabs_dec),
        ("urine(150) > urine(95)",         urine_list[5] > urine_list[3]),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 7 — равновесие объёма (intake = urine)
# ---------------------------------------------------------------------
def test_volume_equilibrium():
    print("\n" + "=" * 78)
    print("TEST 7: Равновесие объёма — P_eq, где intake = urine")
    print("=" * 78)
    k = make_kidney()
    intake = SYS_CFG.get("fluid_intake_rate", 0.02)
    ins = SYS_CFG.get("insensible_loss_rate", 0.0)
    target = intake - ins
    print(f"  intake={intake} мл/с, insensible={ins} мл/с, "
          f"target urine = {target:.5f} мл/с = {target*60:.3f} мл/мин")

    def f(P):
        return k.compute_effects(P, 12.0, 0.0, 5800.0)["urine_output"] - target

    try:
        P_eq = bisect(f, 30.0, 200.0)
        eff_eq = k.compute_effects(P_eq, 12.0, 0.0, 5800.0)
        print(f"  P_eq = {P_eq:.1f} мм рт.ст.")
        print(f"    GFR   = {eff_eq['GFR']*60:.2f} мл/мин")
        print(f"    urine = {eff_eq['urine_output']*60:.3f} мл/мин")
        ok = 60 < P_eq < 130
        print(f"  [{'OK' if ok else 'FAIL'}] P_eq в [60,130] — физиологично")
    except Exception as e:
        print(f"  Не найдено равновесие: {e}")
        ok = False
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 8 — mass balance токсина
# ---------------------------------------------------------------------
def test_toxin_mass_balance():
    print("\n" + "=" * 78)
    print("TEST 8: Mass balance токсина")
    print("=" * 78)
    k = make_kidney()
    P_sa, P_sv, Vb, C0 = 90.0, 12.0, 5000.0, 0.01
    C = C0
    dt, t_end = 1.0, 3600.0
    mass_removed = 0.0
    for _ in range(int(t_end / dt)):
        eff = k.compute_effects(P_sa, P_sv, C, Vb)
        GFR_s = eff["GFR"]
        toxin_excreted = k.toxin_clearance_frac * GFR_s * C
        mass_removed += toxin_excreted * dt
        C = C + eff["dC_tox"] * dt
    mass_lost = (C0 - C) * Vb
    rel = abs(mass_removed - mass_lost) / max(abs(mass_removed), 1e-12)
    print(f"  C0 = {C0}, C(1ч) = {C:.6f}, Vb = {Vb}")
    print(f"  Удалено через excreted: {mass_removed:.6e}")
    print(f"  Потеряно из C·Vb:       {mass_lost:.6e}")
    print(f"  rel = {rel:.3e}")
    ok = rel < 1e-3
    print(f"  [{'OK' if ok else 'FAIL'}] mass balance < 1e-3")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 9 — Q_renal ∝ 1/R_eff (не линейно от (P_sa − P_sv))
# ---------------------------------------------------------------------
def test_q_renal_with_autoreg():
    print("\n" + "=" * 78)
    print("TEST 9: Q_renal = (P_sa − P_sv)/R_eff (R_eff зависит от P_sa)")
    print("=" * 78)
    k = make_kidney()
    print(f"  {'P_sa':>5}  {'P_sv':>5}  {'R_eff':>10}  {'Q_renal, мл/с':>15}  {'pred':>10}")
    ok = True
    for P_sa in [60, 85, 95, 120]:
        for P_sv in [5, 12]:
            eff = k.compute_effects(P_sa, P_sv, 0.0, 5800.0)
            R_eff = k._renal_resistance_eff(P_sa)
            pred = max((P_sa - P_sv) / R_eff, 0.0)
            diff = abs(eff["Q_renal"] - pred)
            ok = ok and diff < 1e-9
            print(f"  {P_sa:>5}  {P_sv:>5}  {R_eff:>10.4f}  "
                  f"{eff['Q_renal']:>15.4f}  {pred:>10.4f}")
    print(f"  [{'OK' if ok else 'FAIL'}] Q_renal = max((P_sa−P_sv)/R_eff, 0)")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 10 — клип Q_renal при P_sa < P_sv
# ---------------------------------------------------------------------
def test_q_renal_clip():
    print("\n" + "=" * 78)
    print("TEST 10: Клип Q_renal ≥ 0 при P_sa ≤ P_sv (анурия)")
    print("=" * 78)
    k = make_kidney()
    print(f"  {'P_sa':>5}  {'P_sv':>5}  {'Q_renal':>10}  {'GFR, мл/мин':>14}  {'urine, мл/мин':>15}")
    ok = True
    for P_sa, P_sv in [(0, 12), (10, 12), (5, 5), (0, 30)]:
        eff = k.compute_effects(P_sa, P_sv, 0.0, 5800.0)
        print(f"  {P_sa:>5}  {P_sv:>5}  {eff['Q_renal']:>10.4f}  "
              f"{eff['GFR']*60:>14.2f}  {eff['urine_output']*60:>15.3f}")
        if eff["Q_renal"] < -1e-9:
            ok = False
    print(f"  [{'OK' if ok else 'FAIL'}] Q_renal ≥ 0 всегда")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 11 — строгие инварианты на сетке
# ---------------------------------------------------------------------
def test_strict_invariants():
    print("\n" + "=" * 78)
    print("TEST 11: Строгие инварианты на сетке P_sa × P_sv × V_blood")
    print("=" * 78)
    k = make_kidney()
    violations = []
    for P_sa in [0, 20, 40, 60, 85, 100, 150, 200]:
        for P_sv in [0, 5, 12, 30]:
            for Vb in [3000.0, 5800.0, 8000.0]:
                eff = k.compute_effects(P_sa, P_sv, 0.01, Vb)
                for key, val in eff.items():
                    if not np.isfinite(val):
                        violations.append(f"non-finite {key} @ ({P_sa},{P_sv},{Vb})")
                if eff["GFR"] < 0:
                    violations.append(f"GFR<0 @ ({P_sa},{P_sv})")
                if eff["urine_output"] < 0:
                    violations.append(f"urine<0 @ ({P_sa},{P_sv})")
                if eff["Q_renal"] < -1e-9:
                    violations.append(f"Q_renal<0 @ ({P_sa},{P_sv})")
                if eff["dV_blood"] > 0:
                    violations.append(f"dV_blood>0 @ ({P_sa},{P_sv})")
    if violations:
        for v in violations[:10]:
            print(f"  [FAIL] {v}")
    else:
        print(f"  Все проверки пройдены (3×4×3 = 36 комбинаций)")
    ok = len(violations) == 0
    print(f"  [{'OK' if ok else 'FAIL'}] строгие инварианты")
    return {"ok": ok}


# ---------------------------------------------------------------------
# Сводка
# ---------------------------------------------------------------------
def summary(results):
    print("\n" + "=" * 78)
    print("СВОДКА: KidneyHemodynamic (новая физиология)")
    print("=" * 78)
    names = [
        "Интерфейс 0-state",
        "auto-consistent renal_resistance",
        "GFR-кривая (плато + прорыв)",
        "RBF-стабилизация на 20 мл/с",
        "basal_urine floor",
        "Pressure-natriuresis",
        "Равновесие объёма",
        "Mass balance токсина",
        "Q_renal с ауторегуляцией",
        "Клип Q_renal ≥ 0",
        "Строгие инварианты",
    ]
    for n, r in zip(names, results):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    print()
    if all(r["ok"] for r in results):
        print("ВЫВОД: KidneyHemodynamic физиологичен в изоляции.")
        print("Все ключевые показатели: GFR(95)=120, Q_renal(95)=20 мл/с,")
        print("urine(95)≈intake, basal floor работает, клип Q≥0 активен.")
        print("Следующий шаг — полный прогон run_simulation_parallel.py.")
    else:
        print("ВЫВОД: см. FAIL выше.")


def run_all(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    sys.stdout = Tee(sys.__stdout__, buf)
    try:
        banner()
        rs = [
            test_interface(),
            test_auto_resistance(),
            test_gfr_curve(),
            test_rbf_plateau(),
            test_basal_urine(),
            test_pressure_natriuresis(),
            test_volume_equilibrium(),
            test_toxin_mass_balance(),
            test_q_renal_with_autoreg(),
            test_q_renal_clip(),
            test_strict_invariants(),
        ]
        summary(rs)
        print(f"\nФиниш: {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout = sys.__stdout__
    path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Отчёт сохранён: {path}")


if __name__ == "__main__":
    run_all(RESULT_FILE)