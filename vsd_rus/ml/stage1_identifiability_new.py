#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml/stage1_identifiability.py — Stage 1: скрининг идентифицируемых параметров.

Источники параметров (единственные):
    config/physiology.yaml      — модель (значения θ₀, X0, солвер).
    config/identifiability.yaml — методология Stage 1 (шкалы, границы,
                                  пороги, целевое число оставляемых параметров).

Алгоритм:
    1. J_ij = (ΔX_i / X0_i) / (Δθ_j / s_j), центрированная разность,
       столбцы = vary_params + alt_params.
    2. SVD(J), cond = S_max / S_min.
    3. Если cond > cond_threshold — жадное обратное исключение:
       фиксируем параметр с максимальным |Vt[-1]|, пересчитываем SVD,
       повторяем до cond < threshold или len(keep) == target_n_keep.
    4. Приоритетные параметры (priority_to_fix) фиксируются первыми.

Запуск:
    python -m ml.stage1_identifiability                # verbose=1
    python -m ml.stage1_identifiability -v 2           # verbose=2
    python -m ml.stage1_identifiability -v 0           # тихо
    python -m ml.stage1_identifiability debug          # диагностика
    python -m ml.stage1_identifiability --config path
    python -m ml.stage1_identifiability --physio path
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning,   module="scipy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whole_body import WholeBodyModel                         # noqa: E402
from sim_builder import build_model_from_params               # noqa: E402
from ml.identifiability_config import (                       # noqa: E402
    load_identifiability_config,
    IdentifiabilityConfig,
)


# =============================================================================
# Tee-логгер
# =============================================================================

class Tee:
    def __init__(self, *streams):
        if not streams:
            raise ValueError("Tee: нужен хотя бы один поток")
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        if "\n" in data:
            self.flush()

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def fileno(self):
        return self.streams[0].fileno()


# =============================================================================
# 1. θ → модель
# =============================================================================

def d_vsd_to_R_vsd(d_mm: float, k_vsd: float) -> float:
    if d_mm <= 0:
        return float("inf")
    return k_vsd / ((d_mm / 2.0) ** 4)


def _apply_theta_to_cfg(physio: dict, theta: dict, ident_cfg) -> dict:
    """
    Накладывает θ на merged-physiology cfg.

    d_vsd → heart.R_vsd (через Пуазейля с k_vsd из YAML).
    HR_base → heart.hr И baroreflex.HR_base.
    Остальные — по identifiability.theta_sources.

    Универсально: работает для любого набора vary_params при условии,
    что theta_sources в YAML покрывает их.
    """
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in physio.items()}

    if theta.get("d_vsd") is not None:
        cfg["heart"]["R_vsd"] = (
            d_vsd_to_R_vsd(theta["d_vsd"], ident_cfg.k_vsd)
            if theta["d_vsd"] > 0 else float("inf")
        )

    if theta.get("HR_base") is not None:
        cfg["heart"]["hr"] = float(theta["HR_base"])
        cfg["baroreflex"]["HR_base"] = float(theta["HR_base"])

    for p, src in ident_cfg.ident["theta_sources"].items():
        if src.get("from") == "geometry":
            continue
        if p == "HR_base":
            continue
        v = theta.get(p)
        if v is None:
            continue
        cfg[src["section"]][src["key"]] = v

    return cfg


def build_model(theta: dict, ident_cfg: IdentifiabilityConfig) -> WholeBodyModel:
    cfg = _apply_theta_to_cfg(ident_cfg.physiology, theta, ident_cfg)
    return build_model_from_params(cfg)


def resolve_R_sys(theta: dict, ident_cfg: IdentifiabilityConfig) -> float:
    if theta.get("R_sys") is not None:
        return float(theta["R_sys"])
    return float(build_model(theta, ident_cfg).R_sys_peripheral)


# =============================================================================
# 2. Симуляция → стационарные выходы
# =============================================================================

_STAGE1_REQUIRED_KEYS = (
    "method", "rtol", "atol", "max_step",
    "t_calib", "t_end", "n_samples_t",
    "t_start_stationary", "stationary_rel_tol",
)


def _resolve_stage1_sim_cfg(ident_cfg: IdentifiabilityConfig,
                            verbose: int = 0) -> dict:
    s = dict(ident_cfg.sim_cfg)
    missing = [k for k in _STAGE1_REQUIRED_KEYS if k not in s]
    if missing:
        raise ValueError(
            "physiology.simulation ∪ identifiability.simulation "
            f"не содержат обязательных ключей Stage 1: {missing}."
        )
    hl = ident_cfg.hard_limits
    if "t_end_max" in hl and s["t_end"] > hl["t_end_max"]:
        if verbose >= 1:
            print(f"  [warn] Stage1: t_end {s['t_end']} → {hl['t_end_max']}")
        s["t_end"] = float(hl["t_end_max"])
    if "n_samples_max" in hl and s["n_samples_t"] > hl["n_samples_max"]:
        if verbose >= 1:
            print(f"  [warn] Stage1: n_samples_t {s['n_samples_t']} → "
                  f"{hl['n_samples_max']}")
        s["n_samples_t"] = int(hl["n_samples_max"])
    return s


def _collect_outputs(model, sol) -> dict:
    keys = None
    outputs = []
    for i, ti in enumerate(sol.t):
        out = model.compute_outputs(ti, sol.y[:, i])
        if keys is None:
            keys = list(out.keys())
        outputs.append(out)
    data = {k: np.fromiter((o[k] for o in outputs),
                           dtype=float, count=len(outputs))
            for k in keys}
    data["t"] = np.asarray(sol.t, dtype=float)
    return data


def _passes_sanity(X: dict, ident_cfg: IdentifiabilityConfig) -> tuple[bool, str]:
    sanity = ident_cfg.ident.get("sanity", {})
    if not sanity:
        return True, "no sanity section"
    for name, (lo, hi) in sanity.items():
        v = X.get(name)
        if v is None:
            continue
        if not (float(lo) <= float(v) <= float(hi)):
            return False, f"{name}={v:.3f} вне [{lo}, {hi}]"
    return True, "ok"


def get_steady_outputs(model, ident_cfg: IdentifiabilityConfig,
                       verbose: int = 0) -> Optional[dict]:
    """
    Калибровка → интегрирование → проверка стационара → усреднение
    за последние 10 циклов; EDV — за последний цикл.
    """
    verbose = int(verbose)
    cfg = _resolve_stage1_sim_cfg(ident_cfg, verbose=verbose)
    t_end = float(cfg["t_end"])
    n_samples = int(cfg["n_samples_t"])
    t_calib = float(cfg["t_calib"])
    t_start_stationary = float(cfg["t_start_stationary"])
    method = str(cfg["method"])

    y0 = model.calibrate_initial_state(t_calib=t_calib)

    if verbose >= 1:
        try:
            print(f"[CALIB] t_calib={t_calib:.1f}s "
                  f"y0 heart={y0[model.idx['heart']]} "
                  f"V_blood={y0[model.idx['blood']][0]:.1f} "
                  f"P_sa0={y0[model.idx['sys_art']][0]:.2f}")
        except Exception as e:
            print(f"[CALIB] y0 diag fail: {e}")

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
        if verbose >= 1:
            print(f"  [warn] solver failed: {e}")
        return None

    if verbose >= 1:
        print(f"[solver] nfev={sol.nfev} njev={getattr(sol, 'njev', 0)} "
              f"t={sol.t[-1]:.1f} y_last heart={sol.y[:4, -1]} "
              f"success={getattr(sol, 'success', False)}")

    if not getattr(sol, "success", False) or sol.y.shape[1] < 2:
        return None
    if not np.all(np.isfinite(sol.y[:, -1])):
        return None

    data = _collect_outputs(model, sol)

    if verbose >= 1:
        print("\n--- CONVERGENCE WINDOWS ---")
        for t_lo, t_hi in [(100, 300), (250, 350), (300, 400),
                           (400, 600), (600, 800)]:
            m = (data["t"] >= t_lo) & (data["t"] <= t_hi)
            if not np.any(m):
                continue
            print(f"[{t_lo:4d}-{t_hi:4d}] "
                  f"P_sa={data['P_sa'][m].mean():6.2f}±"
                  f"{data['P_sa'][m].std():5.2f} "
                  f"V_lv={data['V_lv'][m].mean():6.1f} "
                  f"HR={data['HR'][m].mean():5.1f} "
                  f"P_sv={data['P_sv'][m].mean():5.2f} "
                  f"Q_periph={data['Q_peripheral'][m].mean():6.2f} "
                  f"R_eff={data['R_eff_peripheral'][m].mean():5.3f}")

    # --- Стационарность: две волны по 10 циклов ---
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
    stat_tol = float(cfg["stationary_rel_tol"])
    if verbose >= 1:
        print(f"[STATIONARITY] early={mean_early:.2f} late={mean_late:.2f} "
              f"rel={rel_diff:.4f} tol={stat_tol}")
    if rel_diff > stat_tol:
        if verbose >= 1:
            print(f"  [warn] нестационарен: |ΔP_sa|/P_sa = "
                  f"{rel_diff:.4f} > {stat_tol}")
        return None

    HR_mean = float(np.mean(data["HR"][tail_for_hr]))
    T = 60.0 / max(HR_mean, 1e-6)
    window_avg = data["t"] > (data["t"][-1] - 10.0 * T)
    window_edv = data["t"] > (data["t"][-1] - T)

    X = {}
    for key in ("P_sa", "P_pa", "Q_aortic", "HR"):
        X[key] = float(np.mean(data[key][window_avg]))
    mean_Qp = float(np.mean(data["Q_pulmonary"][window_avg]))
    mean_Qa = float(np.mean(data["Q_aortic"][window_avg]))
    X["Qp_Qs"] = mean_Qp / max(mean_Qa, 1e-6)
    X["EDV_LV"] = float(np.max(data["V_lv"][window_edv]))
    X["EDV_RV"] = float(np.max(data["V_rv"][window_edv]))

    if verbose >= 1:
        print(f"\n[X0 STEADY] P_sa={X['P_sa']:.2f} P_pa={X['P_pa']:.2f} "
              f"Qa={mean_Qa:.2f} Qp={mean_Qp:.2f} Qp/Qs={X['Qp_Qs']:.3f} "
              f"EDV_LV={X['EDV_LV']:.1f} EDV_RV={X['EDV_RV']:.1f} "
              f"HR={X['HR']:.2f}")

    ok, reason = _passes_sanity(X, ident_cfg)
    if not ok:
        if verbose >= 1:
            print(f"  [warn] нефизиологично: {reason}")
        return None

    X["_data"] = data
    return X


# =============================================================================
# 3. Численный якобиан
# =============================================================================

def compute_jacobian(theta0: dict, ident_cfg: IdentifiabilityConfig,
                     verbose: int = 1) -> tuple[np.ndarray, dict, dict]:
    """
    J_ij = (ΔX_i / X0_i) / (Δθ_j / s_j).

    Столбцы = vary_params + alt_params (alt может быть пустым).
    Для каждого параметра: центрированная разность с шагом,
    клипнутым по param_bounds.
    """
    verbose = int(verbose)
    param_names = list(ident_cfg.vary_params) + list(ident_cfg.alt_params)
    scales = dict(ident_cfg.param_scales)
    bounds = dict(ident_cfg.param_bounds)
    x_names = list(ident_cfg.x_names)
    x_scale = dict(ident_cfg.x_typical_scale)
    rel_step = ident_cfg.rel_step
    d_vsd_min_delta = ident_cfg.d_vsd_min_delta

    # --- Полнота конфигурации ---
    for p in param_names:
        if p not in scales:
            raise ValueError(
                f"identifiability.param_scales: нет ключа {p!r}. "
                f"Добавьте в config/identifiability.yaml."
            )
        if p not in bounds:
            raise ValueError(
                f"identifiability.param_bounds: нет ключа {p!r}. "
                f"Добавьте в config/identifiability.yaml."
            )
    for x in x_names:
        if x not in x_scale:
            raise ValueError(
                f"identifiability.x_typical_scale: нет ключа {x!r}."
            )

    theta0 = dict(theta0)
    if theta0.get("R_sys") is None:
        theta0["R_sys"] = resolve_R_sys(theta0, ident_cfg)
    if scales.get("R_sys") is None:
        scales["R_sys"] = float(theta0["R_sys"])

    if verbose >= 2:
        print("=" * 70 + "\n[Stage1] THETA0 RESOLVED")
        for k in param_names:
            print(f"  {k:28s} = {theta0.get(k)}  scale={scales.get(k)}")
        print("=" * 70)
    elif verbose >= 1:
        print(f"[Stage1] R_sys resolved = {theta0['R_sys']:.4f}")
        print(f"[Stage1] R_vsd(d_vsd={theta0['d_vsd']}мм) = "
              f"{d_vsd_to_R_vsd(theta0['d_vsd'], ident_cfg.k_vsd):.4f}")

    X0 = get_steady_outputs(build_model(theta0, ident_cfg),
                            ident_cfg, verbose=verbose)
    if X0 is None:
        raise RuntimeError("Базовая точка не вышла на стационар")

    if verbose >= 1:
        print("\n[Stage1] X0 BASELINE:")
        for k in x_names:
            print(f"    {k:12s} = {X0[k]:.6f}")

    n_x, n_p = len(x_names), len(param_names)
    J = np.full((n_x, n_p), np.nan)

    for j, p in enumerate(param_names):
        scale = float(scales[p])
        delta = rel_step * scale
        if p == "d_vsd":
            delta = max(delta, d_vsd_min_delta)

        theta_plus = dict(theta0)
        theta_minus = dict(theta0)
        theta_plus[p] = theta0[p] + delta
        theta_minus[p] = theta0[p] - delta

        lo, hi = bounds[p]
        theta_plus[p] = min(theta_plus[p], hi)
        theta_minus[p] = max(theta_minus[p], lo)
        if theta_plus[p] <= theta_minus[p]:
            if verbose >= 1:
                print(f"  [warn] {p}: пертурбация вырождена в "
                      f"[{lo}, {hi}] → столбец NaN")
            continue

        if verbose >= 2:
            print(f"\n--- J param {j:2d} {p:28s} scale={scale:.4g} "
                  f"delta={delta:.4g} → plus={theta_plus[p]:.4g} "
                  f"minus={theta_minus[p]:.4g}")

        Xp = get_steady_outputs(build_model(theta_plus, ident_cfg),
                                ident_cfg, verbose=0)
        Xm = get_steady_outputs(build_model(theta_minus, ident_cfg),
                                ident_cfg, verbose=0)
        if Xp is None or Xm is None:
            if verbose >= 1:
                print(f"  [warn] {p}: Xp={Xp is not None}, "
                      f"Xm={Xm is not None} → столбец NaN")
            continue

        for i, x in enumerate(x_names):
            dX = (Xp[x] - Xm[x]) / (2.0 * delta)
            denom = max(abs(X0[x]), x_scale[x])
            J[i, j] = dX * scale / denom

        if verbose >= 2:
            for x in x_names:
                print(f"    dX {x:10s}: X0={X0[x]:8.3f} Xp={Xp[x]:8.3f} "
                      f"Xm={Xm[x]:8.3f} dX={Xp[x] - Xm[x]:+8.3f} "
                      f"J={J[x_names.index(x), j]:+8.4f}")
        elif verbose >= 1:
            print(f"  [ok] {p:28s} delta={delta:.4g}  "
                  f"dX(P_sa)={Xp['P_sa'] - X0['P_sa']:+.4g}  "
                  f"dX(Qp_Qs)={Xp['Qp_Qs'] - X0['Qp_Qs']:+.4g}")

    return J, {k: X0[k] for k in x_names}, theta0


# =============================================================================
# 4. SVD и жадное обратное исключение
# =============================================================================

def analyze_svd(J: np.ndarray, param_names: list, x_names: list,
                verbose: int = 0) -> dict:
    verbose = int(verbose)

    if np.any(np.isnan(J)):
        bad = [param_names[j] for j in range(J.shape[1])
               if np.any(np.isnan(J[:, j]))]
        print(f"[warn] NaN в J по {bad} → 0 для SVD")

    J_clean = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
    U, S, Vt = np.linalg.svd(J_clean, full_matrices=False)
    cond = float(S[0] / S[-1]) if S[-1] > 1e-12 else float("inf")
    v_last = Vt[-1]
    contrib = pd.Series(np.abs(v_last), index=param_names)\
                .sort_values(ascending=False)

    if verbose >= 1:
        print(f"\n=== SVD ({len(param_names)} params) ===")
        print(f"cond = {cond:.4f}  S_max={S[0]:.4f}  S_min={S[-1]:.6f}")
        print("|Vt[-1]| (top-10):")
        for k, v in contrib.head(10).items():
            print(f"  {k:28s} : {v:.6f}")

    return {"U": U, "S": S, "Vt": Vt, "cond": cond,
            "v_last": v_last, "contrib": contrib, "J_clean": J_clean}


def _svd_on_subset(J: np.ndarray, param_names: list, keep: list,
                   x_names: list, verbose: int = 0) -> dict:
    """SVD по подмножеству столбцов (сохраняет исходный порядок param_names)."""
    keep_idx = [param_names.index(p) for p in keep]
    return analyze_svd(J[:, keep_idx], keep, x_names, verbose=verbose)


def select_final_theta(J: np.ndarray, param_names: list,
                       ident_cfg: IdentifiabilityConfig,
                       verbose: int = 1) -> dict:
    """
    Жадное обратное исключение.

    1. forced_fix = priority_to_fix — фиксируются безусловно.
    2. keep = param_names \\ forced_fix.
    3. Пока cond(keep) > cond_threshold и len(keep) > target_n_keep:
         фиксируем параметр с максимальным |Vt[-1]|,
         убираем из keep, добавляем в to_fix.
    4. Если cond(keep) < threshold раньше — останавливаемся.
       Если достигли target_n_keep, а cond всё ещё > threshold —
       возвращаем с пометкой status='target_reached_but_ill_conditioned'.

    Возвращает dict с полной историей шагов.
    """
    x_names = list(ident_cfg.x_names)
    cond_thr = ident_cfg.cond_threshold
    target_keep = ident_cfg.target_n_keep
    forced_fix = [p for p in ident_cfg.priority_to_fix if p in param_names]

    # Стартовое состояние
    keep = [p for p in param_names if p not in forced_fix]
    to_fix = list(forced_fix)
    history = []

    # Первая диагностика
    res_full = analyze_svd(J, param_names, x_names, verbose=0)
    if verbose >= 1:
        print(f"\n[Stage1] cond_full({len(param_names)}) = "
              f"{res_full['cond']:.2e}")

    # Итеративное исключение
    while True:
        res_keep = _svd_on_subset(J, param_names, keep, x_names, verbose=0)
        cond_keep = res_keep["cond"]
        history.append({
            "step": len(history),
            "n_keep": len(keep),
            "n_fix": len(to_fix),
            "cond": cond_keep,
            "just_fixed": history[-1]["just_fixed"] if history else None,
        })
        if verbose >= 1:
            print(f"[Stage1] step {len(history) - 1:2d}: "
                  f"keep={len(keep):2d}  cond={cond_keep:.3e}")

        if cond_keep < cond_thr:
            if verbose >= 1:
                print(f"[Stage1] cond < {cond_thr} достигнут при "
                      f"len(keep)={len(keep)}")
            status = "converged"
            break

        if len(keep) <= target_keep:
            if verbose >= 1:
                print(f"[Stage1] достигнут target_n_keep={target_keep}, "
                      f"но cond={cond_keep:.3e} > {cond_thr}")
            status = "target_reached_but_ill_conditioned"
            break

        # Фиксируем самый вкладной в слабое направление
        worst = res_keep["contrib"].index[0]
        keep.remove(worst)
        to_fix.append(worst)
        if verbose >= 1:
            print(f"           → fix {worst!r} "
                  f"(|Vt[-1]|={res_keep['contrib'][worst]:.4f})")
        history.append({"just_fixed": worst})

    # Финальный SVD
    res_final = _svd_on_subset(J, param_names, keep, x_names, verbose=0)

    return {
        "status":          status,
        "theta_final":     keep,
        "to_fix":          to_fix,
        "cond":            res_final["cond"],
        "v_last":          res_final["v_last"],
        "res":             res_final,
        "res_all":         res_full,
        "history":         history,
        "target_n_keep":   target_keep,
        "forced_fix":      forced_fix,
    }


# =============================================================================
# 5. Визуализация
# =============================================================================

def plot_results(J: np.ndarray, svd_res: dict, param_names: list,
                 x_names: list, out_path: Path) -> None:
    """2×2 dashboard: heatmap J, SVD-спектр, Vt[-1], текстовая сводка."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    j_abs_max = np.nanmax(np.abs(J)) if np.any(np.isfinite(J)) else 1.0
    im = ax.imshow(np.nan_to_num(J, nan=0.0), aspect="auto", cmap="RdBu_r",
                   vmin=-j_abs_max, vmax=j_abs_max)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(x_names)))
    ax.set_yticklabels(x_names, fontsize=9)
    ax.set_title("Относительный якобиан J")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    S = svd_res["S"]
    ax.semilogy(np.arange(1, len(S) + 1), np.maximum(S, 1e-16), "o-")
    ax.set_xlabel("Индекс сингулярного числа")
    ax.set_ylabel("S_i (log)")
    ax.set_title(f"SVD-спектр, cond = {svd_res['cond']:.2e}")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1, 0]
    v = svd_res["v_last"]
    colors = ["crimson" if abs(x) > 0.4 else "steelblue" for x in v]
    ax.barh(range(len(param_names)), v, color=colors)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names, fontsize=8)
    ax.axvline(0.0, color="k", lw=0.5)
    ax.set_title("Vt[-1] — слабое направление (|v|>0.4 → фиксировать)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    txt = "|Vt[-1]| — вклад в слабое направление:\n\n"
    for k, val in svd_res["contrib"].items():
        flag = "  <-- FIX" if val > 0.4 else ""
        txt += f"  {k:28s} : {val:.4f}{flag}\n"
    txt += f"\ncond = {svd_res['cond']:.3e}"
    ax.text(0.02, 0.98, txt, va="top", ha="left",
            family="monospace", fontsize=8, transform=ax.transAxes)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 6. Self-test: θ₀ ↔ модель (покрывает все theta_sources)
# =============================================================================

def _self_test_theta0(ident_cfg: IdentifiabilityConfig) -> None:
    """
    Проверка, что θ₀ из YAML действительно попадает в модель.
    Универсально: проходит по всем theta_sources и сравнивает с атрибутом
    модели, соответствующим (section, key).
    """
    t0 = dict(ident_cfg.theta0)
    if t0.get("R_sys") is None:
        t0["R_sys"] = resolve_R_sys(t0, ident_cfg)
    m = build_model(t0, ident_cfg)

    # Карта (section, key) → атрибут построенной модели.
    # Только для ключей, которые реально доступны как m.<organ>.<attr>.
    # Если ключа в карте нет — молча пропускаем.
    section_to_organ = {
        "heart":      "heart",
        "lungs":      "lungs",
        "baroreflex": "baroreflex",
        "blood":      "blood",
        "systemic":   None,       # системные собираются в WindkesselVessel
    }
    key_to_attr = {
        ("heart",      "E_max_lv"):    ("E_max_base", "LV"),
        ("heart",      "E_max_rv"):    ("E_max_base", "RV"),
        ("heart",      "E_min_lv"):    ("E_min",      "LV"),
        ("heart",      "E_min_rv"):    ("E_min",      "RV"),
        ("heart",      "hr"):          ("hr_base",     None),
        ("heart",      "R_vsd"):       ("R_vsd",       None),
        ("baroreflex", "HR_base"):     ("HR_base",     None),
        ("baroreflex", "P_set"):       ("P_set",       None),
        ("baroreflex", "k_hr"):        ("k_hr",        None),
        ("baroreflex", "k_inotropy"):  ("k_inotropy",  None),
        ("lungs",      "flow_sensitivity"): ("flow_sensitivity", None),
        ("blood",      "V0"):          ("V0",          None),
    }

    checks = []
    for p, src in ident_cfg.ident["theta_sources"].items():
        if src.get("from") == "geometry":
            continue
        section, key = src["section"], src["key"]
        organ_name = section_to_organ.get(section)
        if organ_name is None:
            continue
        attr_map = key_to_attr.get((section, key))
        if attr_map is None:
            continue
        organ = getattr(m, organ_name, None)
        if organ is None:
            continue
        attr, sub = attr_map
        obj = getattr(organ, attr, None)
        if obj is None:
            continue
        actual = obj[sub] if sub is not None else obj
        expected = t0.get(p)
        if expected is None:
            continue
        checks.append((f"{p} ({section}.{key})", actual, expected))

    if not checks:
        print("[Stage1] self-test: нечего проверять "
              "(key_to_attr не покрывает theta_sources).")

    for name, actual, expected in checks:
        if not np.isclose(float(actual), float(expected), rtol=1e-6):
            raise RuntimeError(
                f"Stage1 self-test: {name}: модель={actual} ≠ θ₀={expected}. "
                f"Проверьте physiology.yaml и theta_sources."
            )

    # Дополнительно: R_sys — через m.R_sys_peripheral
    if t0.get("R_sys") is not None:
        if not np.isclose(float(m.R_sys_peripheral), float(t0["R_sys"]),
                          rtol=1e-6):
            raise RuntimeError(
                f"Stage1 self-test: R_sys: модель={m.R_sys_peripheral} "
                f"≠ θ₀={t0['R_sys']}."
            )


# =============================================================================
# 7. main
# =============================================================================

def _main_impl(verbose: int, out_dir: Path, t_start: datetime,
               ident_cfg: IdentifiabilityConfig) -> None:
    print(f"[Stage1] Старт: {t_start:%Y-%m-%d %H:%M:%S} | verbose={verbose}")
    print("=" * 70)
    print("Stage 1: скрининг идентифицируемых параметров WholeBodyModel")
    print(f"  vary_params: {len(ident_cfg.vary_params)}")
    print(f"  alt_params:  {len(ident_cfg.alt_params)}")
    print(f"  x_names:     {len(ident_cfg.x_names)}")
    print(f"  target_n_keep: {ident_cfg.target_n_keep}")
    print("=" * 70)

    if not ident_cfg.ident.get("sanity"):
        print("[warn] identifiability.sanity не задан — "
              "физиологический sanity-check отключён.")

    _self_test_theta0(ident_cfg)
    print("[Stage1] self-test θ₀ ↔ модель: OK")

    theta0 = dict(ident_cfg.theta0)
    J, X0, theta0_resolved = compute_jacobian(theta0, ident_cfg,
                                              verbose=verbose)

    all_param_names = (list(ident_cfg.vary_params)
                       + list(ident_cfg.alt_params))
    df_J = pd.DataFrame(J, index=ident_cfg.x_names, columns=all_param_names)
    df_J.to_csv(out_dir / "stage1_J.csv", float_format="%.6g")
    print(f"\n[Stage1] J сохранён: {out_dir / 'stage1_J.csv'}")
    print(df_J.round(3))

    svd_res = analyze_svd(J, all_param_names, list(ident_cfg.x_names),
                          verbose=verbose)
    plot_results(J, svd_res, all_param_names, list(ident_cfg.x_names),
                 out_dir / "stage1_SVD.png")
    print(f"[Stage1] График: {out_dir / 'stage1_SVD.png'}")

    _sel_verbose = 1 if verbose >= 2 else 0
    selection = select_final_theta(J, all_param_names, ident_cfg,
                                   verbose=_sel_verbose)
    if verbose >= 1:
        print(f"\n[Stage1] Режим: {selection['status']} | "
              f"θ_final={selection['theta_final']} | "
              f"cond={selection['cond']:.3e}")

    # --- Report ---
    lines = ["# STAGE 1 REPORT\n\n"]
    lines.append(f"**Status:** `{selection['status']}`\n")
    lines.append(f"**Target n_keep:** {selection['target_n_keep']}\n")
    lines.append(f"**Actual n_keep:** {len(selection['theta_final'])}\n")
    lines.append(f"**cond_final:** {selection['cond']:.3e}\n\n")

    lines.append("## Базовая точка θ₀\n")
    lines.append(f"- d_vsd = {theta0_resolved['d_vsd']:.4f} мм → "
                 f"R_vsd = "
                 f"{d_vsd_to_R_vsd(theta0_resolved['d_vsd'], ident_cfg.k_vsd):.4f}\n")
    for p in all_param_names:
        v = theta0_resolved.get(p)
        if v is not None:
            lines.append(f"- {p:28s} = {v}\n")
    lines.append("\n")

    lines.append("## YAML sources (θ → physiology.yaml)\n")
    for p, src in ident_cfg.ident["theta_sources"].items():
        if src.get("from") == "geometry":
            lines.append(f"- {p:28s} ← identifiability.vsd_geometry.d_vsd_ref_mm\n")
        else:
            lines.append(f"- {p:28s} ← physiology.{src['section']}.{src['key']}\n")
    lines.append("\n")

    lines.append("## PARAM_SCALES\n")
    for k, v in ident_cfg.param_scales.items():
        vv = v if v is not None else f"auto → {theta0_resolved.get(k)}"
        lines.append(f"- {k:28s} = {vv}\n")
    lines.append("\n")

    lines.append("## X0 (mean по 10 циклам; EDV — max за последний цикл)\n")
    for k in ident_cfg.x_names:
        lines.append(f"- {k:10s} = {X0[k]:.4f}\n")
    lines.append("\n")

    lines.append(f"## cond_full({len(all_param_names)}) = "
                 f"{svd_res['cond']:.3e}\n\n")

    lines.append("## История жадного исключения\n\n")
    lines.append("| step | n_keep | just_fixed | cond |\n")
    lines.append("|---|---|---|---|\n")
    for h in selection["history"]:
        just = h.get("just_fixed") or "—"
        lines.append(f"| {h.get('step', '')} | {h.get('n_keep', '')} | "
                     f"{just} | {h.get('cond', 0):.3e} |\n")
    lines.append("\n")

    lines.append("## Решение\n")
    lines.append(f"- Зафиксированы (forced): {selection['forced_fix']}\n")
    lines.append(f"- Зафиксированы (greedy): "
                 f"{[p for p in selection['to_fix'] if p not in selection['forced_fix']]}\n")
    lines.append(f"- θ_final: {selection['theta_final']}\n")
    lines.append(f"- cond_final: {selection['cond']:.3e}\n\n")

    lines.append("## Ограничения\n")
    lines.append("- При n_x < n_θ SVD коллапсирует: cond_full может быть "
                 "порядка 1e10+, это нормальный сигнал, а не ошибка.\n")
    lines.append("- Параметры, физически неактивные на baseline "
                 "(например, `pressure_sensitivity` при "
                 "`pressure_remodel=false`), имеют нулевой столбец J "
                 "и фиксируются первыми — по причине неактивности, "
                 "а не слабой идентифицируемости.\n")
    lines.append("- θ_final должен совпадать с "
                 "`dataset_generator.VARY_PARAMS`.\n")

    (out_dir / "STAGE1_REPORT.md").write_text("".join(lines),
                                              encoding="utf-8")
    print(f"[Stage1] Отчёт: {out_dir / 'STAGE1_REPORT.md'}")

    t_end_dt = datetime.now()
    print(f"\n[Stage1] Финиш: {t_end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"[Stage1] Длительность: {t_end_dt - t_start}")


def main(verbose: int = 1,
         ident_path: Optional[str] = None,
         physio_path: Optional[str] = None) -> None:
    verbose = int(verbose)
    ident_cfg = load_identifiability_config(ident_path, physio_path)
    t_start = datetime.now()
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stage1_verbose.log"

    with open(log_path, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            _main_impl(verbose, out_dir, t_start, ident_cfg)
            print(f"[Stage1] Полный лог: {log_path}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


# =============================================================================
# 8. debug_base_point
# =============================================================================

def _debug_impl(verbose: int, t_start: datetime,
                ident_cfg: IdentifiabilityConfig) -> None:
    print("=" * 70)
    print(f"[Stage1 DEBUG] Старт: {t_start:%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    cfg = _resolve_stage1_sim_cfg(ident_cfg, verbose=verbose)

    theta = dict(ident_cfg.theta0)
    theta["R_sys"] = resolve_R_sys(theta, ident_cfg)
    print(f"=== DEBUG BASE POINT ===")
    print(f"[Stage1] R_sys resolved = {theta['R_sys']:.3f}, "
          f"K_VSD = {ident_cfg.k_vsd}")

    model = build_model(theta, ident_cfg)
    t_calib = float(cfg["t_calib"])
    print(f"[Stage1] calibrate t_calib = {t_calib}с")

    y0 = model.calibrate_initial_state(t_calib=t_calib)
    print(f"y0 heart = {y0[model.idx['heart']]}")
    print(f"y0 V_blood = {y0[model.idx['blood']][0]:.0f}")

    t_end = float(cfg["t_end"])
    n_samples = int(cfg["n_samples_t"])
    t_eval = np.linspace(0.0, t_end, n_samples)
    sol = model.simulate(
        (0.0, t_end), t_eval=t_eval, y0=y0,
        method=str(cfg["method"]),
        rtol=float(cfg["rtol"]), atol=float(cfg["atol"]),
        max_step=float(cfg["max_step"]),
    )
    print(f"[solver] nfev={sol.nfev} njev={getattr(sol, 'njev', 0)} "
          f"nlu={getattr(sol, 'nlu', 0)} t={sol.t[-1]:.1f} "
          f"y_last={sol.y[:4, -1]}")
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

    t_start_stat = float(cfg["t_start_stationary"])
    HR_mean = float(np.mean(data["HR"][data["t"] > t_start_stat]))
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


def debug_base_point(verbose: int = 1,
                     ident_path: Optional[str] = None,
                     physio_path: Optional[str] = None) -> None:
    verbose = int(verbose)
    ident_cfg = load_identifiability_config(ident_path, physio_path)
    t_start = datetime.now()
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stage1_debug.log"

    with open(log_path, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            _debug_impl(verbose, t_start, ident_cfg)
            print(f"[Stage1 DEBUG] Полный лог: {log_path}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: скрининг идентифицируемых параметров",
    )
    parser.add_argument(
        "mode", nargs="?", default="run", choices=["run", "debug"],
        help="run (default) — полный анализ; debug — диагностика",
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=1, choices=[0, 1, 2],
        help="0=тихо, 1=базовые принты (default), 2=детально",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Путь к identifiability.yaml.",
    )
    parser.add_argument(
        "--physio", type=str, default=None,
        help="Путь к physiology.yaml.",
    )
    args = parser.parse_args()

    if args.mode == "debug":
        print(f"[Stage1] DEBUG MODE (verbose={args.verbose})")
        debug_base_point(verbose=args.verbose,
                         ident_path=args.config,
                         physio_path=args.physio)
    else:
        print(f"[Stage1] RUN MODE (verbose={args.verbose})")
        main(verbose=args.verbose,
             ident_path=args.config,
             physio_path=args.physio)