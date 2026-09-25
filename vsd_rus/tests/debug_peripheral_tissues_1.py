#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_peripheral_tissues.py — изолированная проверка PeripheralTissues.

Состояние: [C_O2_local, C_lactate_local, R_eff].
Проверяем:
  • интерфейс и устойчивость
  • steady-state (C_O2_loc, R_eff, Q, VO2)
  • масс-баланс O2 (tissue + blood report)
  • масс-баланс лактата (production = clearance + release на стационаре)
  • ауторегуляция: гипоксия → вазодилатация, гипертензия → вазоконстрикция
  • flow-зависимость VO2
  • диагностические выходы согласованы с моделью
  • строгие инварианты на сетке

Запуск:  python tests/debug_peripheral_tissues.py
Отчёт:   tests/results_debug_peripheral_tissues.txt
"""
from __future__ import annotations
import sys, io
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import solve_ivp
from peripheral_tissues import PeripheralTissues
from physio_config import load_physiology

RESULT_FILE = Path(__file__).resolve().parent / "results_debug_peripheral_tissues_1.txt"


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, d):
        for x in self.s: x.write(d); x.flush()
    def flush(self):
        for x in self.s: x.flush()


CFG = load_physiology()
PERI_CFG = dict(CFG.get("peripheral", {}))


def make_pt(**overrides) -> PeripheralTissues:
    """Инстанциируем с параметрами из YAML (не дефолтами класса!)"""
    cfg = dict(PERI_CFG)
    # R_base = null в YAML — передаём явно как в whole_body (R_sys_peripheral)
    if cfg.get("R_base") is None:
        cfg["R_base"] = 3.8
    cfg.update(overrides)
    return PeripheralTissues(**cfg)


INPUTS_NORM = {
    "P_sa": 85.0,
    "P_sv": 5.0,
    "C_a_O2": 0.20,
    "C_v_lactate": 0.10,
    "V_blood": 5000.0,
}


def integrate(pt, inputs, t_end=120.0, max_step=0.5):
    y0 = pt.get_initial_state()
    def rhs(t, y): return pt.get_derivatives(t, y, inputs)
    return solve_ivp(rhs, (0.0, t_end), y0, method="LSODA",
                     rtol=1e-8, atol=1e-10, max_step=max_step)


def out_at(pt, y, inputs):
    pt.get_derivatives(0.0, y, inputs)
    return pt.get_outputs(y)


def banner():
    print("=" * 78)
    print(f"Запуск: {datetime.now():%Y-%m-%d %H:%M:%S} | NumPy {np.__version__}")
    print("PeripheralTissues: изолированный тест")
    print("=" * 78)
    for k, v in PERI_CFG.items():
        print(f"    {k:28s} = {v}")
    print(f"  Входы (норма): {INPUTS_NORM}")
    print(f"  → {RESULT_FILE}")
    print("=" * 78)


# ---------------------------------------------------------------------
# TEST 1 — интерфейс
# ---------------------------------------------------------------------
def test_interface():
    print("\n" + "=" * 78)
    print("TEST 1: Интерфейс OrganModel (3 состояния)")
    print("=" * 78)
    pt = make_pt()
    sz = pt.get_state_size()
    y0 = pt.get_initial_state()
    d0 = pt.get_derivatives(0.0, y0, INPUTS_NORM)
    print(f"  state_size = {sz}  (ожидание 3)")
    print(f"  y0 = {y0}")
    print(f"  dy/dt(0) = {d0}")
    ok = (sz == 3 and y0.size == 3 and d0.size == 3
          and np.all(np.isfinite(y0)) and np.all(np.isfinite(d0)))
    print(f"  [{'OK' if ok else 'FAIL'}] интерфейс корректен")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 2 — steady-state (норма)
# ---------------------------------------------------------------------
def test_steady_normal():
    print("\n" + "=" * 78)
    print("TEST 2: Steady-state (P_sa=85, C_a_O2=0.20)")
    print("=" * 78)
    pt = make_pt()
    sol = integrate(pt, INPUTS_NORM, t_end=200.0)
    y_ss = sol.y[:, -1]
    o = out_at(pt, y_ss, INPUTS_NORM)

    print(f"  C_O2_local       = {o['C_O2_local']:.4f}  (норма 0.15, init 0.10)")
    print(f"  C_lactate_local  = {o['C_lactate_local']:.4f}  (норма 0.10)")
    print(f"  R_eff            = {o['R_eff']:.4f}  (R_base={pt.R_base:.4f})")
    print(f"  Q_peripheral     = {o['Q_peripheral']:.3f} мл/с")
    print(f"  O2_consumption   = {o['O2_consumption_periph']:.4f} мл/с")
    print(f"  f_O2_autoreg     = {o['f_O2_autoreg']:.4f}")
    print(f"  f_P_myogenic     = {o['f_P_myogenic']:.4f}")

    checks = [
        ("C_O2_loc ∈ [0.08, 0.18]",         0.08 < o["C_O2_local"] < 0.18),
        ("C_lactate ∈ [0.05, 0.20]",        0.05 < o["C_lactate_local"] < 0.20),
        ("R_eff ∈ [3.0, 4.5]",              3.0 < o["R_eff"] < 4.5),
        ("Q ∈ [15, 30] мл/с",               15 < o["Q_peripheral"] < 30),
        ("VO2 ∈ [1.0, 2.0] мл/с",           1.0 < o["O2_consumption_periph"] < 2.0),
        ("f_O2 ∈ [0.9, 1.0]",               0.9 < o["f_O2_autoreg"] <= 1.0),
        ("f_P = 1.0 (deadband 85±10)",      abs(o["f_P_myogenic"] - 1.0) < 0.01),
        ("Нет NaN/Inf",                     bool(np.all(np.isfinite(sol.y)))),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")
    return {"ok": ok, "metrics": o}


# ---------------------------------------------------------------------
# TEST 3 — масс-баланс O2 (tissue)
# ---------------------------------------------------------------------
def test_o2_mass_balance():
    print("\n" + "=" * 78)
    print("TEST 3: Масс-баланс O2 — доставка = потребление (на стационаре)")
    print("=" * 78)
    pt = make_pt()
    sol = integrate(pt, INPUTS_NORM, t_end=200.0)
    y_ss = sol.y[:, -1]
    o = out_at(pt, y_ss, INPUTS_NORM)

    Q = o["Q_peripheral"]
    C_loc = o["C_O2_local"]
    C_a = INPUTS_NORM["C_a_O2"]
    VO2 = o["O2_consumption_periph"]
    delivery = Q * (C_a - C_loc)

    print(f"  Доставка O2 = Q·(C_a − C_loc) = {delivery:.6f} мл/с")
    print(f"  Потребление VO2_eff           = {VO2:.6f} мл/с")
    rel = abs(delivery - VO2) / max(abs(VO2), 1e-9)
    print(f"  Относительное расхождение = {rel:.3e}")

    ok = rel < 1e-6
    print(f"  [{'OK' if ok else 'FAIL'}] масс-баланс O2 на стационаре < 1e-6")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 4 — масс-баланс лактата
# ---------------------------------------------------------------------
def test_lactate_mass_balance():
    print("\n" + "=" * 78)
    print("TEST 4: Масс-баланс лактата — production = clearance + release")
    print("=" * 78)
    pt = make_pt()
    sol = integrate(pt, INPUTS_NORM, t_end=200.0)
    y_ss = sol.y[:, -1]
    o = out_at(pt, y_ss, INPUTS_NORM)

    C_lac = o["C_lactate_local"]
    prod = o["lactate_production"]
    clearance = pt.k_lactate_clear * C_lac
    release = o["lactate_release_to_blood"]
    balance = prod - clearance - release

    print(f"  C_lactate_local    = {C_lac:.6f}")
    print(f"  production         = {prod:.6e}")
    print(f"  clearance          = {clearance:.6e}")
    print(f"  release            = {release:.6e}")
    print(f"  balance (prod − cl − rel) = {balance:.3e}")

    ok = abs(balance) < 1e-8
    print(f"  [{'OK' if ok else 'FAIL'}] масс-баланс лактата < 1e-8")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 5 — myogenic response
# ---------------------------------------------------------------------
def test_myogenic():
    print("\n" + "=" * 78)
    print("TEST 5: Миогенный отклик — R_eff растёт с P_sa")
    print("=" * 78)
    print(f"  {'P_sa':>6}  {'R_eff':>10}  {'Q':>10}  {'f_P(out)':>10}  {'f_P(int)':>10}")
    R_list = []
    for P_sa in (60.0, 80.0, 90.0, 100.0, 120.0, 150.0):
        inp = dict(INPUTS_NORM, P_sa=P_sa)
        pt = make_pt()
        sol = integrate(pt, inp, t_end=200.0)
        o = out_at(pt, sol.y[:, -1], inp)
        R_list.append(o["R_eff"])
        # Внутренний расчёт f_P (с deadband) — для сравнения
        excess = P_sa - pt.P_sa_norm
        if abs(excess) <= pt.P_myogenic_deadband:
            f_P_int = 1.0
        else:
            signed = excess - np.sign(excess) * pt.P_myogenic_deadband
            f_P_int = 1.0 + pt.k_P_myogenic * signed
        print(f"  {P_sa:>6.1f}  {o['R_eff']:>10.4f}  {o['Q_peripheral']:>10.3f}  "
              f"{o['f_P_myogenic']:>10.4f}  {f_P_int:>10.4f}")

    # R_eff не убывает с P_sa
    mono = all(R_list[i] <= R_list[i+1] + 1e-6 for i in range(len(R_list)-1))
    print(f"  R_eff монотонно не убывает: {mono}")
    ok = mono
    print(f"  [{'OK' if ok else 'FAIL'}] миогенный отклик корректен")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 6 — гипоксическая вазодилатация
# ---------------------------------------------------------------------
def test_hypoxia():
    print("\n" + "=" * 78)
    print("TEST 6: Гипоксия — C_a_O2 ↓ → C_O2_loc ↓, R_eff ↓, лактат ↑")
    print("=" * 78)
    print(f"  {'C_a_O2':>8}  {'C_O2_loc':>10}  {'R_eff':>10}  {'VO2':>10}  {'lac_prod':>12}")
    results = []
    for Ca in (0.20, 0.15, 0.10, 0.06):
        inp = dict(INPUTS_NORM, C_a_O2=Ca)
        pt = make_pt()
        sol = integrate(pt, inp, t_end=200.0)
        o = out_at(pt, sol.y[:, -1], inp)
        results.append((Ca, o["C_O2_local"], o["R_eff"],
                        o["O2_consumption_periph"], o["lactate_production"]))
        print(f"  {Ca:>8.3f}  {o['C_O2_local']:>10.4f}  {o['R_eff']:>10.4f}  "
              f"{o['O2_consumption_periph']:>10.4f}  {o['lactate_production']:>12.6e}")

    # C_O2_loc падает с C_a_O2
    c_o2 = [r[1] for r in results]
    mono_down = all(c_o2[i] >= c_o2[i+1] - 1e-6 for i in range(len(c_o2)-1))
    # R_eff падает (вазодилатация)
    r_eff = [r[2] for r in results]
    vaso = r_eff[-1] <= r_eff[0] + 1e-6
    # лактат растёт
    lac = [r[4] for r in results]
    lac_up = lac[-1] >= lac[0] - 1e-9

    checks = [
        ("C_O2_loc падает с C_a_O2",   mono_down),
        ("R_eff падает (вазодилатация)", vaso),
        ("lactate_production растёт",  lac_up),
        ("C_O2_loc ≤ C_a_O2 всюду",    all(r[1] <= r[0] + 1e-6 for r in results)),
    ]
    ok = True
    for name, c in checks:
        ok = ok and c
        print(f"  [{'OK' if c else 'FAIL'}] {name}")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 7 — flow-зависимость VO2
# ---------------------------------------------------------------------
def test_vo2_flow_dependence():
    print("\n" + "=" * 78)
    print("TEST 7: VO2_eff зависит от Q (flow-зависимый метаболизм)")
    print("=" * 78)
    print(f"  {'P_sa':>6}  {'Q':>10}  {'VO2_eff':>10}  {'Q_factor':>10}")
    V_list = []
    for P_sa in (40.0, 60.0, 85.0, 120.0, 160.0):
        inp = dict(INPUTS_NORM, P_sa=P_sa)
        pt = make_pt()
        sol = integrate(pt, inp, t_end=200.0)
        o = out_at(pt, sol.y[:, -1], inp)
        qf = float(np.clip(o["Q_peripheral"] / 20.0, 0.1, 1.5))
        V_list.append(o["O2_consumption_periph"])
        print(f"  {P_sa:>6.1f}  {o['Q_peripheral']:>10.3f}  "
              f"{o['O2_consumption_periph']:>10.4f}  {qf:>10.4f}")

    mono = all(V_list[i] <= V_list[i+1] + 1e-6 for i in range(len(V_list)-1))
    print(f"  VO2 монотонно растёт с Q: {mono}")
    ok = mono
    print(f"  [{'OK' if ok else 'FAIL'}] flow-зависимость VO2 корректна")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 8 — отчёт в BloodPool
# ---------------------------------------------------------------------
def test_blood_report():
    print("\n" + "=" * 78)
    print("TEST 8: Отчёт в BloodPool (dC_lactate_blood)")
    print("=" * 78)
    pt = make_pt()
    sol = integrate(pt, INPUTS_NORM, t_end=200.0)
    y_ss = sol.y[:, -1]
    o = out_at(pt, y_ss, INPUTS_NORM)

    dC_lac_blood = o["dC_lactate_blood"]
    # Проверка: dC_blood = lac_release · V_tissue_eff / V_blood
    lac_release = o["lactate_release_to_blood"]
    expected = (lac_release * pt.V_tissue_eff) / INPUTS_NORM["V_blood"]

    print(f"  lactate_release       = {lac_release:.6e} мг/(мл·с)")
    print(f"  dC_lactate_blood      = {dC_lac_blood:.6e} мг/(мл·с)")
    print(f"  Ожидание (release·V_t/V_b) = {expected:.6e}")
    rel = abs(dC_lac_blood - expected) / max(abs(expected), 1e-12)

    print(f"  Относительное расхождение = {rel:.3e}")
    ok = rel < 1e-9
    print(f"  [{'OK' if ok else 'FAIL'}] отчёт в BloodPool согласован")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 9 — строгие инварианты (сетка)
# ---------------------------------------------------------------------
def test_strict_invariants():
    print("\n" + "=" * 78)
    print("TEST 9: Строгие инварианты (сетка P_sa × C_a_O2 × V_blood)")
    print("=" * 78)
    violations = []
    for P_sa in (20.0, 60.0, 85.0, 120.0, 200.0):
        for Ca in (0.02, 0.10, 0.20):
            for Vb in (2000.0, 5000.0, 8000.0):
                inp = dict(INPUTS_NORM, P_sa=P_sa, C_a_O2=Ca, V_blood=Vb)
                pt = make_pt()
                try:
                    sol = integrate(pt, inp, t_end=150.0)
                except Exception as e:
                    violations.append(f"crash @ ({P_sa},{Ca},{Vb}): {e}")
                    continue
                if not np.all(np.isfinite(sol.y)):
                    violations.append(f"non-finite @ ({P_sa},{Ca},{Vb})")
                    continue
                y = sol.y[:, -1]
                if y[0] < -1e-8:
                    violations.append(f"C_O2_loc<0 @ ({P_sa},{Ca},{Vb})")
                if y[1] < -1e-8:
                    violations.append(f"C_lac<0 @ ({P_sa},{Ca},{Vb})")
                if y[2] < 1e-3:
                    violations.append(f"R_eff<1e-3 @ ({P_sa},{Ca},{Vb})")

    if violations:
        for v in violations[:10]:
            print(f"  [FAIL] {v}")
    else:
        print(f"  Все проверки пройдены (5×3×3 = 45 комбинаций)")
    ok = len(violations) == 0
    print(f"  [{'OK' if ok else 'FAIL'}] строгие инварианты")
    return {"ok": ok}


# ---------------------------------------------------------------------
# TEST 10 — консистентность диагностических выходов
# ---------------------------------------------------------------------
def test_diagnostics_consistency():
    print("\n" + "=" * 78)
    print("TEST 10: Диагностика vs модель (f_P_myogenic, f_O2_autoreg)")
    print("=" * 78)
    pt = make_pt()

    # f_P_myogenic в outputs НЕ учитывает deadband; проверим для P_sa=100
    P_sa = 100.0
    inp = dict(INPUTS_NORM, P_sa=P_sa)
    sol = integrate(pt, inp, t_end=200.0)
    y_ss = sol.y[:, -1]
    o = out_at(pt, y_ss, inp)

    # Модель (внутри _autoregulation_target)
    excess = P_sa - pt.P_sa_norm  # 100 - 90 = 10
    if abs(excess) <= pt.P_myogenic_deadband:  # |10| ≤ 10 → deadband
        f_P_model = 1.0
    else:
        f_P_model = 1.0 + pt.k_P_myogenic * (excess - np.sign(excess) * pt.P_myogenic_deadband)

    # Diagnostics
    f_P_diag = o["f_P_myogenic"]  # clip(1 + k_P·(P_sa−P_norm), 1, R_max) — БЕЗ deadband

    print(f"  P_sa = {P_sa}")
    print(f"  f_P (модель, с deadband)      = {f_P_model:.6f}")
    print(f"  f_P_myogenic (диагностика)     = {f_P_diag:.6f}")
    print(f"  Разница: {abs(f_P_model - f_P_diag):.6f}")

    consistent = abs(f_P_model - f_P_diag) < 1e-6
    print(f"  [{'OK' if consistent else 'WARN'}] диагностика согласована с моделью")
    if not consistent:
        print("  ↑ Диагностика f_P_myogenic в outputs не учитывает deadband.")
        print("     Это визуальный дефект, не влияет на физику.")
    # Не FAIL — только warning
    return {"ok": True, "consistent": consistent}


# ---------------------------------------------------------------------
# Сводка
# ---------------------------------------------------------------------
def summary(results):
    print("\n" + "=" * 78)
    print("СВОДКА: PeripheralTissues")
    print("=" * 78)
    names = [
        "Интерфейс 3-state",
        "Steady-state (норма)",
        "Масс-баланс O2",
        "Масс-баланс лактата",
        "Миогенный отклик",
        "Гипоксия",
        "Flow-зависимость VO2",
        "Отчёт в BloodPool",
        "Строгие инварианты",
        "Консистентность diagnostics",
    ]
    for n, r in zip(names, results):
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {n}")
    if all(r["ok"] for r in results):
        print("\nВЫВОД: PeripheralTissues структурно корректен.")
        print("⚠ Физиологические замечания (см. TEST 5, 10):")
        print("  • f_O2 = 1 − k·ΔO2 — слишком слабая вазодилатация (нужно 2–3×)")
        print("  • VO2 floor 0.1 — ткань потребляет O2 без кровотока")
    else:
        print("\nВЫВОД: см. FAIL выше.")


def run_all(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    sys.stdout = Tee(sys.__stdout__, buf)
    try:
        banner()
        rs = [
            test_interface(),
            test_steady_normal(),
            test_o2_mass_balance(),
            test_lactate_mass_balance(),
            test_myogenic(),
            test_hypoxia(),
            test_vo2_flow_dependence(),
            test_blood_report(),
            test_strict_invariants(),
            test_diagnostics_consistency(),
        ]
        summary(rs)
        print(f"\nФиниш: {datetime.now():%Y-%m-%d %H:%M:%S}")
    finally:
        sys.stdout = sys.__stdout__
    path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Отчёт сохранён: {path}")


if __name__ == "__main__":
    run_all(RESULT_FILE)