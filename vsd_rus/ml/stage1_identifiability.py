#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml/stage1_identifiability.py

Stage 1 из ML_Tasks.txt: анализ идентифицируемости 8 параметров
модели WholeBodyModel по 7 наблюдаемым (ЭхоКГ-подобным) выходам.

Метод:
    J_ij = (dX_i / X0_i) / (dtheta_j / scale_j)   # относительный якобиан
    SVD(J); cond = S_max / S_min
    Если cond > 100 - коллинеарность, фиксируем слабые параметры.

Запуск:
    python -m ml.stage1_identifiability
или:
    python ml/stage1_identifiability.py
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- Путь к корню проекта (там, где whole_body.py) ------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whole_body import WholeBodyModel  # noqa: E402


# =============================================================================
# 0. Константы и маппинг параметров
# =============================================================================

# d_vsd -> R_vsd = k / r^4,   r = d/2
# Калибруем так, чтобы d_vsd = 4 мм давало R_vsd = 5.0 (как в run_simulation.py)
_D_VSD_REF = 4.0
_R_VSD_REF = 5.0
K_VSD = _R_VSD_REF * (_D_VSD_REF / 2.0) ** 4


def d_vsd_to_R_vsd(d_mm: float) -> float:
    """Диаметр ДМЖП (мм) -> гидродинамическое сопротивление (усл. ед.)."""
    if d_mm <= 0:
        return np.inf
    r_mm = d_mm / 2.0
    return K_VSD / (r_mm ** 4)


# Порядок параметров фиксирован - он же порядок столбцов J
PARAM_NAMES = [
    "d_vsd",            # -> R_vsd
    "E_max_lv",         # -> heart_params['E_max_lv']
    "E_max_rv",         # -> heart_params['E_max_rv']
    "R_sys",            # -> R_sys_peripheral
    "flow_sensitivity", # -> lungs_params['flow_sensitivity']
    "C_sys_art",        # -> C_sys_art
    "HR_base",          # -> baroreflex_params['HR_base'] и heart_params['hr']
    "V0_blood",         # -> blood_params['V0']
]

# Базовая точка "здорового взрослого", но с МАЛЫМ ДМЖП (d_vsd=4 мм),
# чтобы dR_vsd/dd_vsd != 0.
THETA0 = {
    "d_vsd":            4.0,     # мм
    "E_max_lv":         2.5,
    "E_max_rv":         0.8,
    "R_sys":            None,    # посчитается под MAP=85, CO=83
    "flow_sensitivity": 0.02,
    "C_sys_art":        2.0,
    "HR_base":          70.0,
    "V0_blood":         5000.0,
}

# Масштабы для относительного якобиана (характерные величины параметров)
PARAM_SCALES = {
    "d_vsd":            4.0,     # мм
    "E_max_lv":         2.5,
    "E_max_rv":         0.8,
    "R_sys":            1.0,
    "flow_sensitivity": 0.05,
    "C_sys_art":        2.0,
    "HR_base":          70.0,
    "V0_blood":         5000.0,
}

# Наблюдаемые выходы (ЭхоКГ-подобные)
X_NAMES = ["P_sa", "P_pa", "Q_aortic", "Qp_Qs", "EDV_LV", "EDV_RV", "HR"]

# Начальные концентрации веществ (как в run_simulation.py)
INITIAL_CONC = {
    "tox": 0.0,
    "bilirubin": 0.5,
    "ammonia": 0.3,
    "albumin": 4.5,
    "glucose": 5.0,
    "oxygen": 0.15,
}

# =============================================================================
# 1. Построение модели из θ
# =============================================================================

def build_model(theta: dict,
                target_MAP: float = 85.0,
                target_CO: float = 83.0) -> WholeBodyModel:
    """Собирает WholeBodyModel из словаря θ (см. THETA0)."""
    R_vsd = d_vsd_to_R_vsd(theta["d_vsd"]) if theta["d_vsd"] > 0 else np.inf

    heart_params = {
        "E_max_lv": theta["E_max_lv"],
        "E_max_rv": theta["E_max_rv"],
        "hr":       theta["HR_base"],
        "R_vsd":    R_vsd,
    }
    lungs_params = {
        "flow_dependent_resistance": True,
        "flow_sensitivity":         theta["flow_sensitivity"],
    }
    blood_params = {
        "V0": theta["V0_blood"],
        "initial_concentrations": INITIAL_CONC,
    }
    baroreflex_params = {
        "P_set":   90.0,
        "HR_base": theta["HR_base"],
        "gain":    0.01,
        "tau":     2.0,
    }

    model = WholeBodyModel(
        heart_params=heart_params,
        lungs_params=lungs_params,
        blood_params=blood_params,
        baroreflex_params=baroreflex_params,
        vsd_resistance=R_vsd,
        flow_dependent_lungs=True,
        R_sys_peripheral=theta["R_sys"],
        target_MAP=target_MAP,
        target_CO=target_CO,
        C_sys_art=theta["C_sys_art"],
        C_sys_ven=12.0,
        C_pul_ven=5.0,
    )
    return model


def resolve_R_sys(theta: dict) -> float:
    """Если R_sys=None, спросить у модели, какое значение она посчитала."""
    if theta["R_sys"] is not None:
        return float(theta["R_sys"])
    m = build_model({**theta, "R_sys": None})
    return float(m.R_sys_peripheral)


# =============================================================================
# 2. Симуляция и стационарные выходы
# =============================================================================

def _collect_outputs(model: WholeBodyModel, sol) -> dict:
    """Прогоняет compute_outputs по всем точкам решения."""
    keys = None
    rows = []
    for i, ti in enumerate(sol.t):
        out = model.compute_outputs(ti, sol.y[:, i])
        if keys is None:
            keys = list(out.keys())
        rows.append([out[k] for k in keys])
    data = {k: np.array([r[j] for r in rows]) for j, k in enumerate(keys)}
    data["t"] = sol.t
    return data


def get_steady_outputs(model: WholeBodyModel,
                       t_span=(0.0, 400.0),
                       n_samples: int = 8000,
                       t_start_stationary: float = 300.0,
                       verbose: bool = False) -> dict | None:
    """
    Калибрует y0, интегрирует BDF, проверяет стационар
    и возвращает усреднённые за 10 циклов выходы.
    Возвращает None, если решение не сошлось.
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_samples)
    y0 = model.calibrate_initial_state(t_calib=10.0)
    sol = model.simulate(t_span, t_eval, y0=y0, method="BDF", rtol=1e-6)
    # sol = model.simulate(t_span, t_eval, method='RK45', rtol=1e-5, atol=1e-7)

    if sol.y.shape[1] < 2 or not np.all(np.isfinite(sol.y[:, -1])):
        if verbose:
            print("  [warn] решение не конечно")
        return None

    data = _collect_outputs(model, sol)

    # Проверка стационарности (P_sa за последние 50 с)
    last50 = data["t"] > (t_span[1] - 50.0)
    ps_tail = data["P_sa"][last50]
    if ps_tail.size == 0 or np.mean(ps_tail) <= 0:
        return None
    if np.std(ps_tail) / np.mean(ps_tail) > 0.02:
        if verbose:
            print(f"  [warn] нестационарен: std/mean(P_sa)={np.std(ps_tail)/np.mean(ps_tail):.3f}")
        return None

    # Усредняем за последние 10 кардиоциклов
    tail = data["t"] > t_start_stationary
    HR_mean = float(np.mean(data["HR"][tail])) if np.any(tail) else 70.0
    T = 60.0 / max(HR_mean, 1e-6)
    window = data["t"] > (data["t"][-1] - 10.0 * T)

    X = {}
    for key in ["P_sa", "P_pa", "Q_aortic", "Qp_Qs", "HR"]:
        X[key] = float(np.mean(data[key][window]))
    X["EDV_LV"] = float(np.max(data["V_lv"][window]))
    X["EDV_RV"] = float(np.max(data["V_rv"][window]))
    return X


# =============================================================================
# 3. Относительный численный якобиан
# =============================================================================

def compute_jacobian(theta0: dict,
                     rel_step: float = 0.01,
                     verbose: bool = True
                     ) -> tuple[np.ndarray, dict, dict]:
    """
    J_ij = (dX_i / X0_i) / (dtheta_j / scale_j)

    Возвращает (J, X0, theta0_resolved), где theta0_resolved - это θ с
    уже разрешённым R_sys.
    """
    # Разрешаем R_sys один раз
    theta0 = dict(theta0)
    if theta0["R_sys"] is None:
        theta0["R_sys"] = resolve_R_sys(theta0)

    if verbose:
        print(f"[Stage1] R_sys resolved = {theta0['R_sys']:.4f}")
        print(f"[Stage1] R_vsd(d_vsd={theta0['d_vsd']}мм) = {d_vsd_to_R_vsd(theta0['d_vsd']):.4f}")

    model0 = build_model(theta0)
    X0 = get_steady_outputs(model0, verbose=verbose)
    if X0 is None:
        raise RuntimeError("Базовая точка не вышла на стационар")

    if verbose:
        print("[Stage1] X0:")
        for k in X_NAMES:
            print(f"    {k:14s} = {X0[k]:.4f}")

    nX = len(X_NAMES)
    nP = len(PARAM_NAMES)
    J = np.full((nX, nP), np.nan)

    for j, p in enumerate(PARAM_NAMES):
        theta_p = dict(theta0)
        delta = PARAM_SCALES[p] * rel_step
        theta_p[p] = theta0[p] + delta

        # Отдельно пересчитываем R_vsd для d_vsd
        if p == "d_vsd":
            pass  # build_model сам вызовет d_vsd_to_R_vsd

        model_p = build_model(theta_p)
        Xp = get_steady_outputs(model_p, verbose=False)
        if Xp is None:
            if verbose:
                print(f"  [warn] сдвиг {p} не сошёлся, столбец NaN")
            continue

        for i, x in enumerate(X_NAMES):
            dX = (Xp[x] - X0[x]) / delta
            J[i, j] = dX * PARAM_SCALES[p] / max(abs(X0[x]), 1e-9)

        if verbose:
            print(f"  [ok] {p:18s} delta={delta:.4g}  "
                  f"dX(P_sa)={(Xp['P_sa']-X0['P_sa']):+.4g}  "
                  f"dX(Qp_Qs)={(Xp['Qp_Qs']-X0['Qp_Qs']):+.4g}")

    return J, X0, theta0


# =============================================================================
# 4. SVD-анализ
# =============================================================================

def analyze_svd(J: np.ndarray, param_names: list[str], x_names: list[str]) -> dict:
    """SVD + число обусловленности + последний правый сингулярный вектор."""
    # NaN-колонки -> нули, чтобы SVD не падал
    J_clean = np.nan_to_num(J, nan=0.0)

    U, S, Vt = np.linalg.svd(J_clean, full_matrices=False)
    cond = float(S[0] / S[-1]) if S[-1] > 1e-12 else np.inf

    # Последний правый сингулярный вектор - самое слабое направление
    v_last = Vt[-1]

    # Вклад каждого параметра в слабое направление
    contrib = pd.Series(np.abs(v_last), index=param_names).sort_values(ascending=False)

    return {
        "U": U,
        "S": S,
        "Vt": Vt,
        "cond": cond,
        "v_last": v_last,
        "contrib": contrib,
        "J_clean": J_clean,
    }


def fix_and_recompute(J: np.ndarray,
                      param_names: list[str],
                      x_names: list[str],
                      to_fix: list[str],
                      verbose: bool = True) -> dict:
    """Фиксирует параметры и пересчитывает SVD для урезанной матрицы."""
    keep_idx = [i for i, p in enumerate(param_names) if p not in to_fix]
    J_red = J[:, keep_idx]
    keep_names = [param_names[i] for i in keep_idx]
    res = analyze_svd(J_red, keep_names, x_names)
    if verbose:
        print(f"[Stage1] Зафиксировали: {to_fix}")
        print(f"[Stage1] cond({len(keep_names)} параметров) = {res['cond']:.3f}")
        print(f"[Stage1] слабое направление: "
              f"{dict(zip(keep_names, np.round(res['v_last'], 3)))}")
    return res


# =============================================================================
# 5. Визуализация
# =============================================================================

def plot_results(J: np.ndarray,
                 svd_res: dict,
                 param_names: list[str],
                 x_names: list[str],
                 out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Тепловая карта J
    ax = axes[0, 0]
    im = ax.imshow(np.nan_to_num(J, nan=0.0), aspect="auto", cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(J)) if np.any(np.isfinite(J)) else -1,
                   vmax=+np.nanmax(np.abs(J)) if np.any(np.isfinite(J)) else 1)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(x_names)))
    ax.set_yticklabels(x_names, fontsize=9)
    ax.set_title("Относительный якобиан J (нормированный)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (2) Спектр сингулярных чисел
    ax = axes[0, 1]
    ax.semilogy(np.arange(1, len(svd_res["S"]) + 1), svd_res["S"], "o-")
    ax.set_xlabel("Индекс сингулярного числа")
    ax.set_ylabel("S_i (log)")
    ax.set_title(f"SVD-спектр,  cond = {svd_res['cond']:.2f}")
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(svd_res["S"][0] / 100.0, color="r", ls="--", alpha=0.5,
               label="порог cond=100")
    ax.legend()

    # (3) Последний правый сингулярный вектор Vt[-1]
    ax = axes[1, 0]
    v = svd_res["v_last"]
    colors = ["crimson" if abs(x) > 0.4 else "steelblue" for x in v]
    ax.bar(range(len(param_names)), v, color=colors)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_title("Vt[-1] - самое слабое направление\n(|v|>0.4 -> фиксировать)")
    ax.grid(True, alpha=0.3)

    # (4) Таблица вкладов
    ax = axes[1, 1]
    ax.axis("off")
    contrib = svd_res["contrib"]
    txt = "Вклад параметров в слабое направление |Vt[-1]|:\n\n"
    for k, val in contrib.items():
        flag = "  <-- FIX" if val > 0.4 else ""
        txt += f"  {k:18s} : {val:.4f}{flag}\n"
    txt += f"\ncond = {svd_res['cond']:.2f}"
    if svd_res["cond"] > 100:
        txt += "\n\nВЫВОД: cond > 100 -> есть коллинеарность\n" \
               "Зафиксируйте параметры с |Vt[-1]|>0.4 и пересчитайте."
    else:
        txt += "\n\nВЫВОД: cond <= 100 -> матрица хорошо обусловлена."
    ax.text(0.02, 0.98, txt, va="top", ha="left", family="monospace",
            fontsize=10, transform=ax.transAxes)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 6. main
# =============================================================================

def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Stage 1: анализ идентифицируемости WholeBodyModel")
    print("=" * 70)

    theta0 = dict(THETA0)
    J, X0, theta0_resolved = compute_jacobian(theta0, rel_step=0.01, verbose=True)

    # Сохраняем якобиан
    df_J = pd.DataFrame(J, index=X_NAMES, columns=PARAM_NAMES)
    df_J.to_csv(out_dir / "stage1_J.csv", float_format="%.6g")
    print(f"\n[Stage1] J сохранён: {out_dir / 'stage1_J.csv'}")
    print(df_J.round(3))

    # SVD
    svd_res = analyze_svd(J, PARAM_NAMES, X_NAMES)
    print(f"\n[Stage1] cond_8 = {svd_res['cond']:.3f}")
    print("[Stage1] Vt[-1]:")
    for k, val in svd_res["contrib"].items():
        print(f"   {k:18s} : {val:+.4f}")

    # Графики
    plot_results(J, svd_res, PARAM_NAMES, X_NAMES,
                 out_dir / "stage1_SVD.png")
    print(f"[Stage1] Графики сохранены: {out_dir / 'stage1_SVD.png'}")

    # Автоматическая рекомендация: фиксируем параметры с |Vt[-1]| > 0.4
    to_fix = [p for p, v in svd_res["contrib"].items() if abs(v) > 0.4]
    # Не более 2 параметров за один раз
    to_fix = to_fix[:2]

    report_lines = []
    report_lines.append("# STAGE 1 REPORT\n")
    report_lines.append(f"- X0 = {json.dumps({k: round(v, 4) for k, v in X0.items()}, indent=2)}\n")
    report_lines.append(f"- cond_8 = {svd_res['cond']:.3f}\n")
    report_lines.append("- |Vt[-1]| (вклад в слабое направление):\n")
    for k, v in svd_res["contrib"].items():
        report_lines.append(f"    - {k:18s} : {v:.4f}\n")

    if to_fix:
        svd_red = fix_and_recompute(J, PARAM_NAMES, X_NAMES, to_fix, verbose=True)
        theta_final = [p for p in PARAM_NAMES if p not in to_fix]
        report_lines.append(f"\n- Зафиксированы: {to_fix}\n")
        report_lines.append(f"- cond_6 = {svd_red['cond']:.3f}\n")
        report_lines.append(f"- θ_final = {theta_final}\n")
        if svd_red["cond"] <= 50:
            report_lines.append("\n**Вывод:** cond_6 <= 50, набор θ_final "
                                "пригоден для Stage 2 (Sobol по 6 параметрам).\n")
        else:
            report_lines.append("\n**Вывод:** cond_6 всё ещё велик, "
                                "попробуйте зафиксировать ещё 1 параметр.\n")
    else:
        report_lines.append("\nНи один параметр не имеет |Vt[-1]| > 0.4 — "
                            "все 8 параметров вносят вклад.\n")

    (out_dir / "STAGE1_REPORT.md").write_text("".join(report_lines),
                                              encoding="utf-8")
    print(f"[Stage1] Отчёт сохранён: {out_dir / 'STAGE1_REPORT.md'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
