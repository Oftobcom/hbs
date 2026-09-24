#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml/stage1_identifiability.py — v7 (объединённая версия)

Stage 1: анализ идентифицируемости 8 параметров WholeBodyModel
по 7 наблюдаемым выходам (ЭхоКГ-подобным).

Метод:
    J_ij = (ΔX_i / X0_i) / (Δθ_j / scale_j)   # относительный якобиан
    SVD(J); cond = S_max / S_min
    cond > 100 → коллинеарность → фиксируем слабые параметры.

Режимы отбора фиксируемых:
    6      — фиксируем 2 параметра (приоритет V0_blood, HR_base, C_sys_art)
    5      — фиксируем ещё один (Fallback A)
    6_alt  — заменяем C_sys_art на k_inotropy (Fallback B)

Объединяет сильные стороны двух предыдущих версий:
    ✓ корректный вызов select_final_theta(..., sim_cfg=...) в main()
    ✓ реальный Fallback B (пересчёт J на 9 столбцах с k_inotropy)
    ✓ compute_jacobian поддерживает param_names / scales_override
    ✓ НЕ мутирует глобальный PARAM_SCALES (локальные копии)
    ✓ физиологические sanity-чеки в get_steady_outputs
    ✓ быстрый Stage 1 (t_end≈1200, t_calib≈300) поверх солвера из YAML

Запуск:
    python -m ml.stage1_identifiability
    python ml/stage1_identifiability.py debug     # диагностика базовой точки
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- Корень проекта (там, где whole_body.py) ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from whole_body import WholeBodyModel
    from physio_config import load_physiology
except ImportError:
    WholeBodyModel = None
    load_physiology = None


# =============================================================================
# 0. Константы — синхронизированы с physiology.yaml
# =============================================================================

# d_vsd -> R_vsd = K / (d/2)^4.  Калибровка: d=4мм -> R=5.0
_D_VSD_REF = 4.0
_R_VSD_REF = 5.0
K_VSD = _R_VSD_REF * (_D_VSD_REF / 2.0) ** 4  # =80

PARAM_NAMES = [
    "d_vsd",
    "E_max_lv",
    "E_max_rv",
    "R_sys",
    "flow_sensitivity",
    "C_sys_art",
    "HR_base",
    "V0_blood",
]

THETA0 = {
    "d_vsd":            4.0,
    "E_max_lv":         3.0,
    "E_max_rv":         0.8,
    "R_sys":            None,
    "flow_sensitivity": 0.1,
    "C_sys_art":        1.5,
    "HR_base":          70.0,
    "V0_blood":         5800.0,
}

# Характерные масштабы параметров. ВАЖНО: используется именно scale,
# а не theta0 — иначе для flow_sensitivity (θ=0.02, scale=0.05) ошибка в 2.5x.
PARAM_SCALES = {
    "d_vsd":            4.0,
    "E_max_lv":         2.5,
    "E_max_rv":         0.8,
    "R_sys":            3.8,
    "flow_sensitivity": 0.05,
    "C_sys_art":        1.5,
    "HR_base":          70.0,
    "V0_blood":         5800.0,
}

# ALT-набор для Fallback B: C_sys_art → k_inotropy
PARAM_NAMES_ALT = PARAM_NAMES + ["k_inotropy"]
PARAM_SCALES_ALT = {**PARAM_SCALES, "k_inotropy": 0.5}
THETA0_ALT = {**THETA0, "k_inotropy": 0.5}

X_NAMES = ["P_sa", "P_pa", "Q_aortic", "Qp_Qs", "EDV_LV", "EDV_RV", "HR"]


def d_vsd_to_R_vsd(d_mm: float) -> float:
    """Диаметр ДМЖП (мм) → гидродинамическое сопротивление."""
    if d_mm <= 0:
        return np.inf
    r_mm = d_mm / 2.0
    return K_VSD / (r_mm ** 4)


# =============================================================================
# 1. Построение модели
# =============================================================================

def build_model(theta: dict,
                target_MAP: Optional[float] = None,
                target_CO: Optional[float] = None,
                config_path: Optional[str] = None,
                config_overrides: Optional[dict] = None) -> "WholeBodyModel":
    """Собирает WholeBodyModel из theta + physiology.yaml."""
    if load_physiology is None:
        raise RuntimeError("physio_config.load_physiology не найден")
    cfg = load_physiology(config_path, config_overrides)

    heart_params = dict(cfg["heart"])
    heart_params["E_max_lv"] = theta["E_max_lv"]
    heart_params["E_max_rv"] = theta["E_max_rv"]
    heart_params["hr"] = theta["HR_base"]
    heart_params["R_vsd"] = (
        d_vsd_to_R_vsd(theta["d_vsd"]) if theta["d_vsd"] > 0 else np.inf
    )

    lungs_params = dict(cfg["lungs"])
    lungs_params["flow_sensitivity"] = theta["flow_sensitivity"]

    blood_params = {
        "V0": theta["V0_blood"],
        "initial_concentrations": cfg["blood"]["initial_concentrations"],
    }

    baroreflex_params = dict(cfg["baroreflex"])
    baroreflex_params["HR_base"] = theta["HR_base"]
    if "k_inotropy" in theta:
        baroreflex_params["k_inotropy"] = float(theta["k_inotropy"])

    liver_params = dict(cfg["liver"])
    kidney_params = dict(cfg["kidney"])
    brain_params = dict(cfg["brain"])
    gitract_params = dict(cfg["gitract"])
    gas_exchange_params = dict(cfg["gas_exchange"])

    peripheral_params = dict(cfg["peripheral"])
    if peripheral_params.get("R_base") is None:
        peripheral_params.pop("R_base", None)

    sys_cfg = cfg["systemic"]
    if target_MAP is None:
        target_MAP = sys_cfg["target_MAP"]
    if target_CO is None:
        target_CO = sys_cfg["target_CO"]

    return WholeBodyModel(
        heart_params=heart_params,
        lungs_params=lungs_params,
        liver_params=liver_params,
        kidney_params=kidney_params,
        blood_params=blood_params,
        gitract_params=gitract_params,
        brain_params=brain_params,
        baroreflex_params=baroreflex_params,
        gas_exchange_params=gas_exchange_params,
        peripheral_params=peripheral_params,
        flow_dependent_lungs=True,
        R_sys_peripheral=theta["R_sys"],
        target_MAP=target_MAP,
        target_CO=target_CO,
        C_sys_art=theta["C_sys_art"],
        C_pul_ven=sys_cfg.get("C_pul_ven", 15.0),
        P_sa0=sys_cfg.get("P_sa0", 85.0),
        P_sv0=sys_cfg.get("P_sv0", 12.0),
        P_pv0=sys_cfg.get("P_pv0", 12.0),
        SYS_VEN_FRACTION=sys_cfg.get("SYS_VEN_FRACTION", 0.58),
        tau_target=sys_cfg.get("tau_target", 300.0),
        fluid_intake_rate=sys_cfg.get("fluid_intake_rate", 0.015),
        insensible_loss_rate=sys_cfg.get("insensible_loss_rate", 0.0),
    )


def resolve_R_sys(theta: dict) -> float:
    """Если R_sys=None — спросить у модели авто-значение под target_MAP/CO."""
    if theta.get("R_sys") is not None:
        return float(theta["R_sys"])
    m = build_model({**theta, "R_sys": None})
    return float(m.R_sys_peripheral)


# =============================================================================
# 2. Симуляция → стационарные выходы
# =============================================================================

def _stage1_sim_cfg(sim_cfg: Optional[dict]) -> dict:
    """
    Солвер (method, rtol, atol, max_step) берётся из YAML, если задан.
    """
    out = {
        "method": "LSODA",
        "rtol": 1e-4,
        "atol": 1e-5,
        "max_step": 0.1,
        "t_calib": 600.0,
        "t_end": 800.0,
        "n_samples_t": 4000,
        "t_start_stationary": 300.0,
        "stationary_rel_tol_stage1": 0.05,
    }
    if not sim_cfg:
        return out

    out["method"] = str(sim_cfg.get("method", out["method"]))
    out["rtol"] = float(sim_cfg.get("rtol", out["rtol"]))
    out["atol"] = float(sim_cfg.get("atol", out["atol"]))
    out["max_step"] = float(sim_cfg.get("max_step", out["max_step"]))

    # t_span в YAML может быть длиннее — обрезаем для Stage 1
    t_span = sim_cfg.get("t_span")
    if isinstance(t_span, (list, tuple)) and len(t_span) >= 2:
        out["t_end"] = min(float(t_span[1]), 1200.0)

    n_samples = sim_cfg.get("n_samples_t")
    if n_samples is not None:
        out["n_samples_t"] = min(int(n_samples), 12000)

    if "stationary_rel_tol_stage1" in sim_cfg:
        out["stationary_rel_tol_stage1"] = float(sim_cfg["stationary_rel_tol_stage1"])

    return out


def _collect_outputs(model, sol) -> dict:
    """Прогоняет compute_outputs по всем точкам решения → dict[str, np.ndarray]."""
    keys = None
    rows = []
    for i, ti in enumerate(sol.t):
        out = model.compute_outputs(ti, sol.y[:, i])
        if keys is None:
            keys = list(out.keys())
        rows.append([out[k] for k in keys])
    data = {k: np.array([r[j] for r in rows], dtype=float)
            for j, k in enumerate(keys)}
    data["t"] = np.asarray(sol.t, dtype=float)
    return data


def get_steady_outputs(model,
                       sim_cfg: Optional[dict] = None,
                       verbose: bool = False) -> Optional[dict]:
    """
    Калибровка → интегрирование → проверка стационара → усреднение
    за последние 10 кардиоциклов.

    Возвращает словарь {P_sa, P_pa, Q_aortic, Qp_Qs, EDV_LV, EDV_RV, HR}
    или None, если решение не сошлось / нефизиологично.
    """
    cfg = _stage1_sim_cfg(sim_cfg)
    t_end = float(cfg["t_end"])
    n_samples = int(cfg["n_samples_t"])
    t_calib = float(cfg["t_calib"])
    t_start_stationary = float(cfg["t_start_stationary"])
    method = str(cfg["method"])

    y0 = model.calibrate_initial_state(t_calib=t_calib)
    t_eval = np.linspace(0.0, t_end, n_samples)
    try:
        sol = model.simulate(
            (0.0, t_end), t_eval=t_eval, y0=y0,
            method=method,
            rtol=float(cfg["rtol"]),
            atol=float(cfg["atol"]),
            max_step=float(cfg["max_step"]),
        )
    except Exception as e:
        if verbose:
            print(f"  [warn] solver failed: {e}")
        return None

    print(f"[solver, 1] nfev={sol.nfev} njev={getattr(sol,'njev',0)} "
        f"nlu={getattr(sol,'nlu',0)} t={sol.t[-1]:.1f} y_last={sol.y[:4,-1]}")

    if not getattr(sol, "success", False) or sol.y.shape[1] < 2:
        return None
    if not np.all(np.isfinite(sol.y[:, -1])):
        return None

    data = _collect_outputs(model, sol)

    # --- Проверка стационарности: два окна по 10 циклов ---
    tail_for_hr = data["t"] > t_start_stationary
    if not np.any(tail_for_hr):
        return None
    HR_for_stat = float(np.mean(data["HR"][tail_for_hr]))
    if not np.isfinite(HR_for_stat) or HR_for_stat < 10:
        return None
    T_stat = 60.0 / HR_for_stat

    dt = float(data["t"][1] - data["t"][0]) if data["t"].size > 1 else 0.1
    win_len = max(int(round(10.0 * T_stat / max(dt, 1e-6))), 4)
    ps_recent = data["P_sa"][-2 * win_len:]
    if ps_recent.size < 4:
        return None
    half = ps_recent.size // 2
    mean_early = float(np.mean(ps_recent[:half]))
    mean_late = float(np.mean(ps_recent[half:]))
    if mean_late <= 0:
        return None
    rel_diff = abs(mean_late - mean_early) / mean_late
    stat_tol = float(cfg["stationary_rel_tol_stage1"])
    if rel_diff > stat_tol:
        if verbose:
            print(f"  [warn] нестационарен: |ΔP_sa|/P_sa = {rel_diff:.4f} > {stat_tol}")
        return None

    # --- Усреднение за последние 10 циклов ---
    HR_mean = float(np.mean(data["HR"][tail_for_hr]))
    T = 60.0 / max(HR_mean, 1e-6)
    window = data["t"] > (data["t"][-1] - 10.0 * T)

    X = {}
    for key in ("P_sa", "P_pa", "Q_aortic", "HR"):
        X[key] = float(np.mean(data[key][window]))
    mean_Qp = float(np.mean(data["Q_pulmonary"][window]))
    mean_Qa = float(np.mean(data["Q_aortic"][window]))
    X["Qp_Qs"] = mean_Qp / max(mean_Qa, 1e-6)
    X["EDV_LV"] = float(np.max(data["V_lv"][window]))
    X["EDV_RV"] = float(np.max(data["V_rv"][window]))

    # --- Физиологический sanity-check ---
    if not (40.0 < X["P_sa"] < 180.0
            and 5.0 < X["P_pa"] < 80.0
            and X["EDV_LV"] > 50.0
            and 0.0 < X["Qp_Qs"] < 20.0):
        if verbose:
            print(f"  [warn] нефизиологично: P_sa={X['P_sa']:.1f}, "
                  f"P_pa={X['P_pa']:.1f}, Qp_Qs={X['Qp_Qs']:.2f}, "
                  f"EDV_LV={X['EDV_LV']:.0f}")
        return None

    X["_data"] = data
    return X


# =============================================================================
# 3. Относительный численный якобиан
# =============================================================================

def compute_jacobian(theta0: dict,
                     rel_step: float = 0.01,
                     sim_cfg: Optional[dict] = None,
                     verbose: bool = True,
                     param_names: Optional[list] = None,
                     scales_override: Optional[dict] = None,
                     ) -> tuple[np.ndarray, dict, dict]:
    """
    J_ij = (ΔX_i / X0_i) / (Δθ_j / scale_j).

    Возвращает (J, X0_dict, theta0_resolved), где theta0_resolved
    содержит уже разрешённый R_sys. Глобальный PARAM_SCALES НЕ мутируется.
    """
    if param_names is None:
        param_names = PARAM_NAMES
    scales = dict(scales_override) if scales_override is not None else dict(PARAM_SCALES)

    theta0 = dict(theta0)
    if theta0.get("R_sys") is None:
        theta0["R_sys"] = resolve_R_sys(theta0)
    scales["R_sys"] = float(theta0["R_sys"])

    if verbose:
        print(f"[Stage1] R_sys resolved = {theta0['R_sys']:.4f}")
        print(f"[Stage1] R_vsd(d_vsd={theta0['d_vsd']}мм) = "
              f"{d_vsd_to_R_vsd(theta0['d_vsd']):.4f}")

    model0 = build_model(theta0)
    X0 = get_steady_outputs(model0, sim_cfg=sim_cfg, verbose=verbose)
    if X0 is None:
        raise RuntimeError("Базовая точка не вышла на стационар")

    if verbose:
        print("[Stage1] X0:")
        for k in X_NAMES:
            print(f"    {k:12s} = {X0[k]:.4f}")

    n_x, n_p = len(X_NAMES), len(param_names)
    J = np.full((n_x, n_p), np.nan)

    for j, p in enumerate(param_names):
        scale = float(scales[p])
        delta = rel_step * scale
        if p == "d_vsd":
            delta = max(delta, 0.04)

        theta_plus = dict(theta0)
        theta_minus = dict(theta0)
        theta_plus[p] = theta0[p] + delta
        theta_minus[p] = theta0[p] - delta
        if p == "d_vsd":
            theta_minus[p] = max(theta_minus[p], 0.5)

        Xp = get_steady_outputs(build_model(theta_plus), sim_cfg=sim_cfg, verbose=False)
        Xm = get_steady_outputs(build_model(theta_minus), sim_cfg=sim_cfg, verbose=False)
        if Xp is None or Xm is None:
            if verbose:
                print(f"  [warn] {p}: Xp={Xp is not None}, Xm={Xm is not None} → NaN")
            continue

        for i, x in enumerate(X_NAMES):
            dX = (Xp[x] - Xm[x]) / (2.0 * delta)
            J[i, j] = dX * scale / max(abs(X0[x]), 1e-9)

        if verbose:
            print(f"  [ok] {p:18s} delta={delta:.4g}  "
                  f"dX(P_sa)={Xp['P_sa'] - X0['P_sa']:+.4g}  "
                  f"dX(Qp_Qs)={Xp['Qp_Qs'] - X0['Qp_Qs']:+.4g}")

    return J, {k: X0[k] for k in X_NAMES}, theta0


# =============================================================================
# 4. SVD и выбор фиксируемых
# =============================================================================

def analyze_svd(J: np.ndarray, param_names: list, x_names: list) -> dict:
    """SVD + число обусловленности + последний правый сингулярный вектор."""
    if np.any(np.isnan(J)):
        bad = [param_names[j] for j in range(J.shape[1]) if np.any(np.isnan(J[:, j]))]
        print(f"[warn] NaN в J по {bad} → 0 для SVD")
    J_clean = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

    U, S, Vt = np.linalg.svd(J_clean, full_matrices=False)
    cond = float(S[0] / S[-1]) if S[-1] > 1e-12 else np.inf
    v_last = Vt[-1]
    contrib = pd.Series(np.abs(v_last), index=param_names).sort_values(ascending=False)

    return {"U": U, "S": S, "Vt": Vt, "cond": cond,
            "v_last": v_last, "contrib": contrib, "J_clean": J_clean}


def _fix_and_cond(J: np.ndarray, param_names: list, to_fix: list):
    """Возвращает (cond, v_last, keep_names, svd_res) для урезанной J."""
    keep_idx = [i for i, p in enumerate(param_names) if p not in to_fix]
    keep_names = [param_names[i] for i in keep_idx]
    res = analyze_svd(J[:, keep_idx], keep_names, X_NAMES)
    return res["cond"], res["v_last"], keep_names, res


def _pick_to_fix_by_name(svd_res: dict, param_names: list, priority: list,
                         n_fix: int = 2,
                         thr_priority: float = 0.3,
                         thr_other: float = 0.4) -> list:
    """Выбор ровно n_fix параметров для фиксации по приоритету и |Vt[-1]|."""
    contrib = svd_res["contrib"]
    to_fix = []
    for p in priority:
        if p in contrib and float(contrib[p]) > thr_priority:
            to_fix.append(p)
    for p, v in contrib.items():
        if p in to_fix:
            continue
        if float(v) > thr_other:
            to_fix.append(p)
    if len(to_fix) < n_fix:
        for p in contrib.index:
            if p not in to_fix:
                to_fix.append(p)
            if len(to_fix) >= n_fix:
                break
    return to_fix[:n_fix]


def select_final_theta(J: np.ndarray,
                       param_names: list,
                       verbose: bool = True,
                       sim_cfg: Optional[dict] = None) -> dict:
    """
    Логика 6 → 5 → 6_alt.

    Режим '6':     фиксируем 2 параметра (приоритет V0_blood, HR_base, C_sys_art).
    Режим '5':     Fallback A — фиксируем ещё один.
    Режим '6_alt': Fallback B — реальный пересчёт J на 9 параметрах
                   (PARAM_NAMES_ALT) и замена C_sys_art на k_inotropy.
    """
    priority = ["V0_blood", "HR_base", "C_sys_art"]

    # --- Режим 6 ---
    res8 = analyze_svd(J, param_names, X_NAMES)
    if verbose:
        print(f"\n[Stage1] cond_8 = {res8['cond']:.2f}")

    to_fix = _pick_to_fix_by_name(res8, param_names, priority, n_fix=2)
    cond6, v6, keep6, res6 = _fix_and_cond(J, param_names, to_fix)
    if verbose:
        print(f"[Stage1] режим 6: to_fix={to_fix}, cond={cond6:.3f}, "
              f"keep={keep6}")
    if cond6 < 100.0:
        return {"mode": "6", "theta_final": keep6, "to_fix": to_fix,
                "cond": cond6, "v_last": v6, "res": res6, "res8": res8}

    # --- Fallback A (режим 5) ---
    contrib6 = res6["contrib"]
    extra_candidates = ["C_sys_art", "E_max_lv", "R_sys"]
    extra = next((p for p in extra_candidates if p in contrib6), None)
    if extra is None and len(contrib6):
        extra = contrib6.index[0]
    to_fix2 = to_fix + ([extra] if extra is not None else [])

    cond5, v5, keep5, res5 = _fix_and_cond(J, param_names, to_fix2)
    if verbose:
        print(f"[Stage1] Fallback A (5 параметров): to_fix={to_fix2}, "
              f"cond={cond5:.3f}")
    if cond5 < 100.0:
        return {"mode": "5", "theta_final": keep5, "to_fix": to_fix2,
                "cond": cond5, "v_last": v5, "res": res5, "res8": res8}

    # --- Fallback B (режим 6_alt): реальный пересчёт J на 9 столбцах ---
    if verbose:
        print("[Stage1] Fallback B: пересчёт J с k_inotropy (9 столбцов)...")

    J_alt, _, _ = compute_jacobian(
        THETA0_ALT,
        rel_step=0.01,
        sim_cfg=sim_cfg,
        verbose=False,
        param_names=PARAM_NAMES_ALT,
        scales_override=PARAM_SCALES_ALT,
    )
    # Из 9 параметров фиксируем те же to_fix + C_sys_art (заменён на k_inotropy)
    to_fix_alt = list(to_fix) + ["C_sys_art"]
    cond6a, v6a, keep6a, res6a = _fix_and_cond(J_alt, PARAM_NAMES_ALT, to_fix_alt)
    if verbose:
        print(f"[Stage1] режим 6_alt: to_fix={to_fix_alt}, "
              f"cond={cond6a:.3f}, keep={keep6a}")

    return {"mode": "6_alt", "theta_final": keep6a, "to_fix": to_fix_alt,
            "cond": cond6a, "v_last": v6a, "res": res6a, "res8": res8}


# =============================================================================
# 5. Визуализация
# =============================================================================

def plot_results(J: np.ndarray, svd_res: dict, param_names: list,
                 x_names: list, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Тепловая карта J
    ax = axes[0, 0]
    j_abs_max = np.nanmax(np.abs(J)) if np.any(np.isfinite(J)) else 1.0
    im = ax.imshow(np.nan_to_num(J, nan=0.0), aspect="auto", cmap="RdBu_r",
                   vmin=-j_abs_max, vmax=j_abs_max)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(x_names)))
    ax.set_yticklabels(x_names, fontsize=9)
    ax.set_title("Относительный якобиан J")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (2) Спектр сингулярных чисел
    ax = axes[0, 1]
    ax.semilogy(np.arange(1, len(svd_res["S"]) + 1), svd_res["S"], "o-")
    ax.set_xlabel("Индекс сингулярного числа")
    ax.set_ylabel("S_i (log)")
    ax.set_title(f"SVD-спектр, cond = {svd_res['cond']:.2f}")
    ax.grid(True, which="both", alpha=0.3)
    if svd_res["S"][0] > 0:
        ax.axhline(svd_res["S"][0] / 100.0, color="r", ls="--",
                   alpha=0.5, label="порог cond=100")
    ax.legend()

    # (3) Последний правый сингулярный вектор
    ax = axes[1, 0]
    v = svd_res["v_last"]
    colors = ["crimson" if abs(x) > 0.4 else "steelblue" for x in v]
    ax.bar(range(len(param_names)), v, color=colors)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_title("Vt[-1] — слабое направление (|v|>0.4 → фиксировать)")
    ax.grid(True, alpha=0.3)

    # (4) Текстовая сводка
    ax = axes[1, 1]
    ax.axis("off")
    txt = "|Vt[-1]| — вклад в слабое направление:\n\n"
    for k, val in svd_res["contrib"].items():
        flag = "  <-- FIX" if val > 0.4 else ""
        txt += f"  {k:18s} : {val:.4f}{flag}\n"
    txt += f"\ncond = {svd_res['cond']:.2f}"
    txt += "\n\ncond > 100 → коллинеарность" if svd_res["cond"] > 100 \
        else "\n\ncond ≤ 100 → хорошо обусловлена"
    ax.text(0.02, 0.98, txt, va="top", ha="left",
            family="monospace", fontsize=10, transform=ax.transAxes)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 6. main
# =============================================================================

def main() -> None:
    t_start = datetime.now()
    print(f"[Stage1] Старт: {t_start:%Y-%m-%d %H:%M:%S}")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Stage 1: анализ идентифицируемости WholeBodyModel")
    print("=" * 70)

    sim_cfg = None
    if load_physiology is not None:
        cfg = load_physiology()
        sim_cfg = cfg.get("simulation", {})

    theta0 = dict(THETA0)
    J, X0, theta0_resolved = compute_jacobian(
        theta0, rel_step=0.01, sim_cfg=sim_cfg, verbose=True
    )

    df_J = pd.DataFrame(J, index=X_NAMES, columns=PARAM_NAMES)
    df_J.to_csv(out_dir / "stage1_J.csv", float_format="%.6g")
    print(f"\n[Stage1] J сохранён: {out_dir / 'stage1_J.csv'}")
    print(df_J.round(3))

    svd_res = analyze_svd(J, PARAM_NAMES, X_NAMES)
    print(f"\n[Stage1] cond_8 = {svd_res['cond']:.3f}")
    print("[Stage1] |Vt[-1]|:")
    for k, val in svd_res["contrib"].items():
        print(f"   {k:18s} : {val:+.4f}")

    plot_results(J, svd_res, PARAM_NAMES, X_NAMES, out_dir / "stage1_SVD.png")
    print(f"[Stage1] График: {out_dir / 'stage1_SVD.png'}")

    selection = select_final_theta(J, PARAM_NAMES, verbose=True, sim_cfg=sim_cfg)

    lines = []
    lines.append("# STAGE 1 REPORT\n\n")
    lines.append("## Базовая точка θ₀\n")
    lines.append(f"- d_vsd = {theta0_resolved['d_vsd']:.4f} мм → "
                 f"R_vsd = {d_vsd_to_R_vsd(theta0_resolved['d_vsd']):.4f}\n")
    lines.append(f"- R_sys = {theta0_resolved['R_sys']:.4f} (auto под MAP/CO)\n")
    lines.append(f"- E_max_lv = {theta0_resolved['E_max_lv']}, "
                 f"E_max_rv = {theta0_resolved['E_max_rv']}\n")
    lines.append(f"- flow_sensitivity = {theta0_resolved['flow_sensitivity']}, "
                 f"C_sys_art = {theta0_resolved['C_sys_art']}\n")
    lines.append(f"- HR_base = {theta0_resolved['HR_base']}, "
                 f"V0_blood = {theta0_resolved['V0_blood']}\n\n")

    lines.append("## X0 (последние 10 циклов, EDV=max)\n")
    for k in X_NAMES:
        lines.append(f"- {k:10s} = {X0[k]:.4f}\n")
    lines.append("\n")

    lines.append(f"## cond_8 = {svd_res['cond']:.3f}\n\n")
    lines.append("## |Vt[-1]| (слабое направление)\n")
    for k, v in svd_res["contrib"].items():
        flag = "  ← FIX" if k in selection["to_fix"] else ""
        lines.append(f"- {k:18s} : {v:.4f}{flag}\n")
    lines.append("\n")

    lines.append("## Решение\n")
    lines.append(f"- Режим: **{selection['mode']}**\n")
    lines.append(f"- Зафиксированы: {selection['to_fix']}\n")
    lines.append(f"- θ_final = {selection['theta_final']}\n")
    lines.append(f"- cond_final = {selection['cond']:.3f}\n\n")

    lines.append("## Ограничения\n")
    lines.append("- C_sys_art на стационаре влияет только на пульсации; "
                 "если не идентифицируется — фиксируется в Fallback A.\n")
    lines.append("- Экстраполяция за R_vsd ∈ [0.5, 15.8] "
                 "(d_vsd ∈ [3.0, 7.11] мм) не гарантируется.\n")
    lines.append("- Формула J использует PARAM_SCALES, а не theta0.\n")

    (out_dir / "STAGE1_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print(f"[Stage1] Отчёт: {out_dir / 'STAGE1_REPORT.md'}")
    print("\nDone.")

    t_end_dt = datetime.now()
    print(f"\n[Stage1] Финиш: {t_end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"[Stage1] Длительность: {t_end_dt - t_start}") 


def debug_base_point() -> None:
    """Проверка базовой точки: конвергенция, steady-state, один кардиоцикл."""
    t_start = datetime.now()
    print("=" * 70)
    print(f"[Stage1 DEBUG] Старт: {t_start:%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    if load_physiology is None:
        raise RuntimeError("load_physiology не найден")
    cfg = load_physiology()
    sim_cfg = cfg.get("simulation", {})

    theta = dict(THETA0)
    theta["R_sys"] = resolve_R_sys(theta)
    print(f"=== DEBUG BASE POINT ===")
    print(f"[Stage1] R_sys resolved = {theta['R_sys']:.3f}, K_VSD = {K_VSD}")

    model = build_model(theta)
    t_calib = 600.0
    print(f"[Stage1] calibrate t_calib = {t_calib}с")

    y0 = model.calibrate_initial_state(t_calib=t_calib)
    print(f"y0 heart = {y0[model.idx['heart']]}")
    print(f"y0 V_blood = {y0[model.idx['blood']][0]:.0f}")

    t_end = 1000.0
    t_eval = np.linspace(0.0, t_end, 8001)
    sol = model.simulate(
        (0.0, t_end), t_eval=t_eval, y0=y0,
        method=str(sim_cfg.get("method", "LSODA")),
        rtol=1e-4, atol=1e-5, max_step=0.1,
    )
    print(f"[solver, 2] nfev={sol.nfev} njev={getattr(sol,'njev',0)} "
      f"nlu={getattr(sol,'nlu',0)} t={sol.t[-1]:.1f} y_last={sol.y[:4,-1]}")
    data = _collect_outputs(model, sol)

    print("\n--- Конвергенция по окнам ---")
    for t_lo, t_hi in [(100, 300), (250, 350), (300, 400)]:
        m = (data["t"] >= t_lo) & (data["t"] <= t_hi)
        if not np.any(m):
            continue
        ps = data["P_sa"][m]
        print(f"[{t_lo:4d}-{t_hi:4d}]  "
              f"P_sa={ps.mean():5.1f}±{ps.std():4.1f}  "
              f"V_lv={data['V_lv'][m].mean():5.1f}  "
              f"V_rv={data['V_rv'][m].mean():5.1f}  "
              f"HR={data['HR'][m].mean():4.1f}")

    HR_mean = float(np.mean(data["HR"][data["t"] > 250]))
    T = 60.0 / max(HR_mean, 1.0)
    win = data["t"] > (data["t"][-1] - 10.0 * T)
    Qp_mean = float(np.mean(data["Q_pulmonary"][win]))
    Qs_mean = float(np.mean(data["Q_aortic"][win]))
    print(f"\n--- Steady 10 циклов (T={T:.3f}с) ---")
    print(f"P_sa={data['P_sa'][win].mean():.1f} "
          f"P_pa={data['P_pa'][win].mean():.1f} "
          f"Qa={Qs_mean:.1f} "
          f"Qp/Qs={Qp_mean / max(Qs_mean, 1e-6):.2f} "
          f"EDV_LV={np.max(data['V_lv'][win]):.0f} "
          f"EDV_RV={np.max(data['V_rv'][win]):.0f} "
          f"V_blood={data['V_blood'][-1]:.0f}")
    # --- Диагностика наполнения (венозный возврат) ---
    print(f"P_sv={data['P_sv'][win].mean():.1f} P_pv={data['P_pv'][win].mean():.1f} "
          f"P_la={data['P_la'][win].mean():.1f} P_ra={data['P_ra'][win].mean():.1f} "
          f"V_sv={data['V_sv'][win].mean():.0f}/{data['V_sv_target'][win].mean():.0f} "
          f"({data['V_sv_fraction'][win].mean()*100:.0f}%)")
    print(f"Q_sv_to_ra={data['Q_sv_to_ra'][win].mean():.1f} "
          f"Q_pv_to_la={data['Q_pv_to_la'][win].mean():.1f} "
          f"Q_periph={data['Q_peripheral'][win].mean():.1f} "
          f"Q_ven_in={data['Q_ven_in'][win].mean():.1f} "
          f"R_eff={data['R_eff_peripheral'][win].mean():.2f} "
          f"f_P={data['f_P_myogenic'][win].mean():.3f}")

    cycle = data["t"] > (data["t"][-1] - T)
    print(f"\n--- Один кардиоцикл ---")
    for k in ("V_lv", "V_rv", "P_lv", "P_rv", "Q_mitral", "Q_tricuspid"):
        if k not in data:
            continue
        arr = data[k][cycle]
        print(f"  {k:12s}  min={arr.min():7.1f}  "
              f"max={arr.max():7.1f}  mean={arr.mean():7.1f}")

    t_end_dt = datetime.now()
    print("\n" + "=" * 70)
    print(f"[Stage1 DEBUG] Финиш: {t_end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"[Stage1 DEBUG] Длительность: {t_end_dt - t_start}")
    print("=" * 70)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        print("[Stage1] DEBUG MODE")
        debug_base_point()
    else:
        print("[Stage1] RUN MODE")
        main()