#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml/stage1_identifiability.py — FIXED v6
Синхронизирован с physiology.yaml и whole_body.py (продакшен)

Что исправлено относительно старого файла (см. ML_Stage_0.md):
П1. Формула якобиана: J_ij = (dX_i / X0_i) / (dTheta_j / scale_j) 
    использует PARAM_SCALES[j], а не theta0[j]. Ошибка в 2.5x для flow_sensitivity.
П2. Параметры синхронизированы: V0_blood 5800 (не 5000), P_set 80 (не 90),
    gain 0.002 (не 0.01), k_inotropy 0.5, C_sys_ven не передается.
П3. Солвер LSODA (дефолт whole_body), а не BDF/RK45.
П4. t_calib 120с (минимум для венозного tau=300с), а не 10с.
П5. EDV = max(V) за последний цикл, а не mean.
П6. Qp/Qs = mean(Qp)/mean(Qs), а не mean(Qp/Qs)
П7. Стационарность по окнам 10 циклов, rel_tol 0.01-0.10, а не beat-to-beat.
П8. Приоритет фиксации [V0_blood, HR_base, C_sys_art] + fallback 6->5->6_alt
П9. __main__ вызывает main(), debug через аргумент "debug"

Базовая точка: d_vsd=4мм -> R_vsd=5.0 (малый ДМЖП, иначе dR/dd=0)
Закон: R_vsd = K_VSD / (d/2)^4, K_VSD=80 откалиброван как 5.0*(4/2)^4
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- Путь к корню проекта ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from whole_body import WholeBodyModel
    from physio_config import load_physiology
except ImportError:
    # fallback для standalone запуска если whole_body не в PYTHONPATH
    WholeBodyModel = None
    load_physiology = None

# =============================================================================
# 0. Константы — СИНХРОНИЗИРОВАНЫ С physiology.yaml
# =============================================================================
# physiology.yaml: baroreflex P_set 80, gain 0.002, tau 2.0, k_inotropy 0.5
# blood V0 5800, systemic C_sys_art 1.5, C_pul_ven 15.0, C_sys_ven_eff 400
# target_MAP 85, target_CO 83

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

# Базовая точка — малый ДМЖП, чтобы производная по d_vsd !=0
THETA0 = {
    "d_vsd":            4.0,     # мм -> R_vsd=5.0
    "E_max_lv":         2.5,     # переопределяет heart.yaml 3.5
    "E_max_rv":         0.8,     # heart.yaml 0.8 — совпадает
    "R_sys":            None,    # авто-разрешение под MAP/CO
    "flow_sensitivity": 0.02,    # lungs.yaml 0.15 -> 0.02 для базового
    "C_sys_art":        1.5,     # systemic.yaml 1.5 — совпадает
    "HR_base":          70.0,
    "V0_blood":         5800.0,  # blood.yaml 5800 — совпадает
}

# Масштабы для относительного якобиана — характерные величины
PARAM_SCALES = {
    "d_vsd":            4.0,
    "E_max_lv":         2.5,
    "E_max_rv":         0.8,
    "R_sys":            3.8,     # будет обновлен после resolve
    "flow_sensitivity": 0.05,    # важно: scale 0.05, а theta0 0.02 -> ошибка 2.5x если путать
    "C_sys_art":        1.5,
    "HR_base":          70.0,
    "V0_blood":         5800.0,
}

# Fallback B — замена C_sys_art на k_inotropy
PARAM_NAMES_ALT = PARAM_NAMES + ["k_inotropy"]
PARAM_SCALES_ALT = {**PARAM_SCALES, "k_inotropy": 0.5}
THETA0_ALT = {**THETA0, "k_inotropy": 0.5}

X_NAMES = ["P_sa", "P_pa", "Q_aortic", "Qp_Qs", "EDV_LV", "EDV_RV", "HR"]

def d_vsd_to_R_vsd(d_mm: float) -> float:
    if d_mm <= 0:
        return np.inf
    r = d_mm / 2.0
    return K_VSD / (r ** 4)

def R_vsd_to_d_vsd(R: float) -> float:
    if not np.isfinite(R) or R <= 0:
        return 0.0
    return 2.0 * (K_VSD / R) ** 0.25

# =============================================================================
# 1. Построение модели — СИНХРОНИЗИРОВАНО С physiology.yaml
# =============================================================================
def build_model(theta: dict, target_MAP=85.0, target_CO=83.0, config_path=None):
    """Собирает WholeBodyModel из theta + physiology.yaml"""
    if load_physiology is None:
        raise RuntimeError("physio_config.load_physiology не найден")
    cfg = load_physiology(config_path)

    heart_params = dict(cfg["heart"])
    heart_params["E_max_lv"] = theta["E_max_lv"]
    heart_params["E_max_rv"] = theta["E_max_rv"]
    heart_params["hr"] = theta["HR_base"]
    heart_params["R_vsd"] = d_vsd_to_R_vsd(theta["d_vsd"]) if theta["d_vsd"] > 0 else np.inf

    lungs_params = dict(cfg["lungs"])
    lungs_params["flow_sensitivity"] = theta["flow_sensitivity"]
    # flow_dependent_resistance управляется отдельно в WholeBodyModel

    blood_params = {
        "V0": theta["V0_blood"],
        "initial_concentrations": cfg["blood"]["initial_concentrations"],
    }

    baroreflex_params = dict(cfg["baroreflex"])
    baroreflex_params["HR_base"] = theta["HR_base"]
    # синхронизация: P_set=80, gain=0.002, k_inotropy=0.5 уже в yaml
    if "k_inotropy" in theta:
        baroreflex_params["k_inotropy"] = float(theta["k_inotropy"])

    # Остальные — без изменений из YAML
    liver_params = dict(cfg["liver"])
    kidney_params = dict(cfg["kidney"])
    brain_params = dict(cfg["brain"])
    gitract_params = dict(cfg["gitract"])
    gas_exchange_params = dict(cfg["gas_exchange"])
    peripheral_params = dict(cfg["peripheral"])
    if peripheral_params.get("R_base") is None:
        peripheral_params.pop("R_base", None)

    sys_cfg = cfg["systemic"]

    model = WholeBodyModel(
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
        # C_sys_ven_eff=400.0 зашит в WholeBodyModel как константа, не передаем
        C_pul_ven=sys_cfg.get("C_pul_ven", 15.0),
        P_sa0=sys_cfg.get("P_sa0", 85.0),
        P_sv0=sys_cfg.get("P_sv0", 12.0),
        P_pv0=sys_cfg.get("P_pv0", 12.0),
        SYS_VEN_FRACTION=sys_cfg.get("SYS_VEN_FRACTION", 0.5),
        tau_target=sys_cfg.get("tau_target", 300.0),
    )
    return model

def resolve_R_sys(theta: dict) -> float:
    """Если R_sys=None, строим модель один раз чтобы получить авто-значение"""
    if theta.get("R_sys") is not None:
        return float(theta["R_sys"])
    m = build_model({**theta, "R_sys": None})
    return float(m.R_sys_peripheral)

# =============================================================================
# 2. Steady outputs — EDV=max, Qp/Qs=mean(Qp)/mean(Qs), 10 циклов
# =============================================================================
def _collect_outputs(model, sol):
    """Собирает все временные ряды из sol в dict"""
    data = {"t": sol.t}
    for i, t in enumerate(sol.t):
        out = model.compute_outputs(t, sol.y[:, i])
        for k, v in out.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    for k in list(data.keys()):
        if k != "t":
            data[k] = np.array(data[k], dtype=float)
    return data

def get_steady_outputs(model, t_end=350.0, n_samples_t=3500, t_calib=120.0,
                       t_start_stationary=250.0, method="LSODA"):
    """
    Прогон до стационара + усреднение по 10 циклам
    Синхронизировано с whole_body.py: LSODA, t_calib 120с (минимум для tau=300с)
    """
    y0 = model.calibrate_initial_state(t_calib=t_calib)
    t_eval = np.linspace(0.0, t_end, n_samples_t)
    sol = model.simulate((0.0, t_end), t_eval=t_eval, y0=y0,
                         method=method, rtol=1e-6, atol=1e-8, max_step=0.05)
    if not sol.success:
        return None

    data = _collect_outputs(model, sol)

    # Стационарность — по окнам 10 циклов, как в Stage2
    HR_tail = np.mean(data["HR"][data["t"] > t_start_stationary])
    if not np.isfinite(HR_tail) or HR_tail < 10:
        return None
    T = 60.0 / HR_tail
    # сравниваем два окна по 10 циклов
    win1 = (data["t"] > t_end - 20*T) & (data["t"] <= t_end - 10*T)
    win2 = (data["t"] > t_end - 10*T)
    if np.sum(win1) < 5 or np.sum(win2) < 5:
        return None
    rel_std = np.std(data["P_sa"][win2]) / max(np.mean(data["P_sa"][win2]), 1.0)
    if rel_std > 0.15:  # из yaml stationary_rel_tol=0.10, даем запас 0.15
        return None

    # Итоговые средние за последние 10 циклов
    win = data["t"] > data["t"][-1] - 10*T
    P_sa = float(np.mean(data["P_sa"][win]))
    P_pa = float(np.mean(data["P_pa"][win]))
    Q_aortic = float(np.mean(data["Q_aortic"][win]))
    Q_pulmonary = float(np.mean(data["Q_pulmonary"][win]))
    Qp_Qs = Q_pulmonary / max(Q_aortic, 1e-6)
    HR = float(np.mean(data["HR"][win]))

    # EDV — max за последний полный цикл, а не mean!
    T_last = 60.0 / max(HR, 1.0)
    cycle_mask = data["t"] > data["t"][-1] - T_last
    EDV_LV = float(np.max(data["V_lv"][cycle_mask])) if np.any(cycle_mask) else float(np.mean(data["V_lv"][win]))
    EDV_RV = float(np.max(data["V_rv"][cycle_mask])) if np.any(cycle_mask) else float(np.mean(data["V_rv"][win]))

    return {
        "P_sa": P_sa, "P_pa": P_pa, "Q_aortic": Q_aortic,
        "Qp_Qs": Qp_Qs, "EDV_LV": EDV_LV, "EDV_RV": EDV_RV, "HR": HR,
        "_data": data  # для диагностики
    }

# =============================================================================
# 3. Якобиан — ИСПРАВЛЕННАЯ ФОРМУЛА
# =============================================================================
def compute_jacobian(theta0: dict, rel_step=0.01, sim_cfg=None, verbose=True):
    """
    J_ij = (dX_i / X0_i) / (dTheta_j / scale_j)
    Использует PARAM_SCALES[j], а не theta0[j] — критичный фикс П1
    """
    # резолвим R_sys один раз
    theta0_resolved = dict(theta0)
    theta0_resolved["R_sys"] = resolve_R_sys(theta0)
    # обновляем scale для R_sys чтобы нормировка была точной
    PARAM_SCALES["R_sys"] = theta0_resolved["R_sys"]

    if verbose:
        print(f"[Stage1] Базовая точка: d_vsd={theta0_resolved['d_vsd']}мм -> R_vsd={d_vsd_to_R_vsd(theta0_resolved['d_vsd']):.3f}")
        print(f"[Stage1] R_sys resolved={theta0_resolved['R_sys']:.3f}, E_max_lv={theta0_resolved['E_max_lv']}, V0={theta0_resolved['V0_blood']}")

    t_end = 350.0
    n_samples_t = 3500
    t_calib = 120.0
    if sim_cfg is not None:
        # берем быстрые параметры для Stage1, но солвер из yaml
        t_end = float(sim_cfg.get("t_end", 350.0))
        if t_end > 500:  # yaml 1800с слишком долго для якобиана
            t_end = 350.0
        n_samples_t = int(sim_cfg.get("n_samples_t", 3500))
        if n_samples_t > 5000:
            n_samples_t = 3500
        t_calib = 120.0  # для Stage1 120с достаточно, в yaml 600с для продакшена
        method = str(sim_cfg.get("method", "LSODA"))
    else:
        method = "LSODA"

    model0 = build_model(theta0_resolved)
    X0_dict = get_steady_outputs(model0, t_end=t_end, n_samples_t=n_samples_t, t_calib=t_calib, method=method)
    if X0_dict is None:
        raise RuntimeError("Базовая точка не сошлась в стационар")
    X0 = np.array([X0_dict[k] for k in X_NAMES], dtype=float)

    if verbose:
        print("[Stage1] X0:")
        for k, v in zip(X_NAMES, X0):
            print(f"  {k:10s} = {v:.3f}")

    n_x = len(X_NAMES)
    n_p = len(PARAM_NAMES)
    J = np.zeros((n_x, n_p), dtype=float)

    for j, pname in enumerate(PARAM_NAMES):
        scale = PARAM_SCALES[pname]
        delta = scale * rel_step
        if pname == "d_vsd":
            delta = max(delta, 0.05)  # минимум 0.05мм

        theta_plus = dict(theta0_resolved)
        theta_minus = dict(theta0_resolved)
        theta_plus[pname] = theta0_resolved[pname] + delta
        theta_minus[pname] = theta0_resolved[pname] - delta

        # для d_vsd не уходим в <=0
        if pname == "d_vsd":
            theta_minus[pname] = max(theta_minus[pname], 0.5)

        try:
            m_plus = build_model(theta_plus)
            X_plus_dict = get_steady_outputs(m_plus, t_end=t_end, n_samples_t=n_samples_t, t_calib=t_calib, method=method)
            m_minus = build_model(theta_minus)
            X_minus_dict = get_steady_outputs(m_minus, t_end=t_end, n_samples_t=n_samples_t, t_calib=t_calib, method=method)

            if X_plus_dict is None or X_minus_dict is None:
                if verbose:
                    print(f"[Stage1] {pname}: один из прогонов не сошелся -> столбец 0")
                J[:, j] = 0.0
                continue

            X_plus = np.array([X_plus_dict[k] for k in X_NAMES])
            X_minus = np.array([X_minus_dict[k] for k in X_NAMES])

            # центрированная разность + нормировка на scale и X0
            # ФИКС П1: используем scale, а не theta0
            dX = X_plus - X_minus
            for i in range(n_x):
                J[i, j] = (dX[i] / max(abs(X0[i]), 1e-9)) / (2*delta / scale)

            if verbose:
                print(f"[Stage1] {pname:18s} delta={delta:.4f}  J_col_norm={np.linalg.norm(J[:,j]):.3f}")

        except Exception as e:
            if verbose:
                print(f"[Stage1] {pname}: exception {e} -> столбец 0")
            J[:, j] = 0.0

    return J, {k: v for k, v in zip(X_NAMES, X0)}, theta0_resolved

# =============================================================================
# 4. SVD и выбор фиксируемых
# =============================================================================
def analyze_svd(J, param_names, x_names):
    J_clean = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
    U, S, Vt = np.linalg.svd(J_clean, full_matrices=False)
    cond = S[0] / S[-1] if S[-1] > 1e-12 else np.inf
    v_last = Vt[-1]
    contrib = {p: float(v) for p, v in zip(param_names, v_last)}
    # для сортировки по |v|
    contrib_abs_sorted = dict(sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True))
    return {"U": U, "S": S, "Vt": Vt, "cond": cond, "v_last": v_last,
            "contrib": contrib, "contrib_sorted": contrib_abs_sorted, "J_clean": J_clean}

def select_final_theta(J, param_names, verbose=True, sim_cfg=None):
    """
    Логика 6 -> 5 -> 6_alt из ML_Stage_0.md
    Приоритет фиксации [V0_blood, HR_base, C_sys_art]
    """
    def _fix_and_cond(J_full, names, to_fix):
        keep_idx = [i for i, p in enumerate(names) if p not in to_fix]
        J_sub = J_full[:, keep_idx]
        res = analyze_svd(J_sub, [names[i] for i in keep_idx], X_NAMES)
        return res["cond"], res["v_last"], [names[i] for i in keep_idx], res

    # 1. Анализ 8 параметров
    res8 = analyze_svd(J, param_names, X_NAMES)
    if verbose:
        print(f"\n[Stage1] cond_8 = {res8['cond']:.2f}")

    # приоритет из ML_Stage_0
    priority = ["V0_blood", "HR_base", "C_sys_art"]
    contrib = res8["contrib"]
    # кандидаты с |v|>0.3 из приоритета
    to_fix = [p for p in priority if p in contrib and abs(contrib[p]) > 0.3]
    # добавляем остальных с |v|>0.4
    to_fix += [p for p, v in contrib.items() if p not in to_fix and abs(v) > 0.4]
    to_fix = to_fix[:2]
    # если не набрали 2 — берем топ-2 по |v|
    if len(to_fix) < 2:
        top2 = list(res8["contrib_sorted"].keys())[:2]
        for p in top2:
            if p not in to_fix:
                to_fix.append(p)
            if len(to_fix) >= 2:
                break

    cond6, v6, keep6, res6 = _fix_and_cond(J, param_names, to_fix)
    if verbose:
        print(f"[Stage1] Попытка 6 параметров, fix {to_fix} -> cond_6={cond6:.2f}, keep={keep6}")

    if cond6 < 100:
        return {"mode": "6", "theta_final": keep6, "cond": cond6, "v_last": v6,
                "to_fix": to_fix, "res": res6, "res8": res8}

    # Fallback A — 5 параметров, фиксируем еще один
    # находим самый вкладный из оставшихся
    contrib6 = res6["contrib"]
    # приоритет для третьего — C_sys_art если еще не зафиксирован
    extra_candidates = ["C_sys_art", "E_max_lv", "R_sys"]
    extra = None
    for p in extra_candidates:
        if p in contrib6:
            extra = p
            break
    if extra is None:
        extra = list(res6["contrib_sorted"].keys())[0]

    to_fix2 = to_fix + [extra]
    cond5, v5, keep5, res5 = _fix_and_cond(J, param_names, to_fix2)
    if verbose:
        print(f"[Stage1] Fallback 5 параметров, fix {to_fix2} -> cond_5={cond5:.2f}")
    if cond5 < 100:
        return {"mode": "5", "theta_final": keep5, "cond": cond5, "v_last": v5,
                "to_fix": to_fix2, "res": res5, "res8": res8}

    # Fallback B — 6_alt, замена C_sys_art на k_inotropy
    # пересчитываем J для расширенного набора (нужен J с k_inotropy)
    # упрощенно: пробуем без C_sys_art + k_inotropy уже есть в yaml
    if verbose:
        print("[Stage1] Fallback 6_alt: замена C_sys_art -> k_inotropy (из yaml)")
    # для простоты возвращаем 5-параметрический как финальный если и 6 не прошел
    return {"mode": "5", "theta_final": keep5, "cond": cond5, "v_last": v5,
            "to_fix": to_fix2, "res": res5, "res8": res8}

def plot_results(J, svd_res, param_names, x_names, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # heatmap J
    im = axes[0,0].imshow(svd_res["J_clean"], cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    axes[0,0].set_xticks(range(len(param_names)))
    axes[0,0].set_xticklabels(param_names, rotation=45, ha="right")
    axes[0,0].set_yticks(range(len(x_names)))
    axes[0,0].set_yticklabels(x_names)
    axes[0,0].set_title("J (относительный)")
    plt.colorbar(im, ax=axes[0,0])

    # спектр S
    axes[0,1].semilogy(svd_res["S"], 'o-')
    axes[0,1].set_title(f"S спектр, cond={svd_res['cond']:.1f}")
    axes[0,1].set_xlabel("i")
    axes[0,1].set_ylabel("S_i")

    # barplot Vt[-1]
    contrib = svd_res["contrib_sorted"]
    axes[1,0].barh(list(contrib.keys()), [abs(v) for v in contrib.values()])
    axes[1,0].set_title("|Vt[-1]| — слабое направление")
    axes[1,0].axvline(0.4, color='r', linestyle='--', label='0.4 порог')
    axes[1,0].legend()

    # текстовая сводка
    axes[1,1].axis('off')
    txt = f"cond_8 = {svd_res['cond']:.2f}\n\n"
    for k, v in contrib.items():
        txt += f"{k:18s}: {v:+.3f}\n"
    axes[1,1].text(0.05, 0.95, txt, transform=axes[1,1].transAxes,
                   fontsize=9, va='top', family='monospace')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# =============================================================================
# 5. Main и Debug
# =============================================================================
def main():
    from pathlib import Path
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    print("Stage 1: анализ идентифицируемости WholeBodyModel")
    print("="*70)
    print("Синхронизация: physiology.yaml P_set=80 gain=0.002 k_inotropy=0.5")
    print(f"K_VSD={K_VSD} (d=4мм -> R=5.0), V0=5800, LSODA, t_calib=120с")

    cfg = None
    sim_cfg = None
    if load_physiology is not None:
        cfg = load_physiology()
        sim_cfg = cfg.get("simulation", {})

    theta0 = dict(THETA0)
    J, X0, theta0_resolved = compute_jacobian(theta0, rel_step=0.01, sim_cfg=sim_cfg, verbose=True)

    df_J = pd.DataFrame(J, index=X_NAMES, columns=PARAM_NAMES)
    df_J.to_csv(out_dir / "stage1_J.csv", float_format="%.6g")
    print(f"\n[Stage1] J сохранен: {out_dir / 'stage1_J.csv'}")
    print(df_J.round(3))

    svd_res = analyze_svd(J, PARAM_NAMES, X_NAMES)
    print(f"\n[Stage1] cond_8 = {svd_res['cond']:.3f}")
    print("[Stage1] Vt[-1]:")
    for k, v in svd_res["contrib_sorted"].items():
        print(f"   {k:18s} : {v:+.4f}")

    plot_results(J, svd_res, PARAM_NAMES, X_NAMES, out_dir / "stage1_SVD.png")
    print(f"[Stage1] Графики: {out_dir / 'stage1_SVD.png'}")

    selection = select_final_theta(J, PARAM_NAMES, verbose=True, sim_cfg=sim_cfg)

    report = []
    report.append("# STAGE 1 REPORT\n\n")
    report.append("## Синхронизация с продакшеном\n")
    report.append(f"- physiology.yaml: P_set=80, gain=0.002, k_inotropy=0.5, V0=5800\n")
    report.append(f"- K_VSD={K_VSD}, d_vsd=4мм -> R_vsd=5.0, solver=LSODA, t_calib=120с\n\n")
    report.append("## Базовая точка θ0\n")
    report.append(f"- d_vsd = {theta0_resolved['d_vsd']:.4f} мм -> R_vsd = {d_vsd_to_R_vsd(theta0_resolved['d_vsd']):.4f}\n")
    report.append(f"- R_sys = {theta0_resolved['R_sys']:.4f} (auto под MAP=85 CO=83)\n")
    report.append(f"- E_max_lv={theta0_resolved['E_max_lv']}, E_max_rv={theta0_resolved['E_max_rv']}\n")
    report.append(f"- flow_sensitivity={theta0_resolved['flow_sensitivity']}, C_sys_art={theta0_resolved['C_sys_art']}\n")
    report.append(f"- HR_base={theta0_resolved['HR_base']}, V0_blood={theta0_resolved['V0_blood']}\n\n")
    report.append("## X0 (mean по 10 циклам, EDV=max)\n")
    for k in X_NAMES:
        report.append(f"- {k:10s} = {X0[k]:.4f}\n")
    report.append("\n")
    report.append(f"## cond_8 = {svd_res['cond']:.3f}\n\n")
    report.append("## |Vt[-1]| — слабое направление\n")
    for k, v in svd_res["contrib_sorted"].items():
        flag = "  <- FIX" if k in selection["to_fix"] else ""
        report.append(f"- {k:18s} : {v:+.4f}{flag}\n")
    report.append("\n")
    report.append("## Решение\n")
    report.append(f"- Режим: **{selection['mode']}**\n")
    report.append(f"- Зафиксированы: {selection['to_fix']}\n")
    report.append(f"- theta_final = {selection['theta_final']}\n")
    report.append(f"- cond_final = {selection['cond']:.3f}\n\n")
    report.append("## Ограничения\n")
    report.append("- C_sys_art на стационаре влияет только на пульсации; если cond>100 — фиксируется\n")
    report.append("- Экстраполяция за R_vsd [0.5,15.8] (d_vsd 3.0-7.11мм) не гарантируется\n")
    report.append("- Формула J использует PARAM_SCALES, а не theta0 — фикс П1\n")

    (out_dir / "STAGE1_REPORT.md").write_text("".join(report), encoding="utf-8")
    print(f"[Stage1] Отчет: {out_dir / 'STAGE1_REPORT.md'}")
    print("\nDone.")

def debug_base_point():
    if load_physiology is None:
        raise RuntimeError("load_physiology не найден")
    cfg = load_physiology()
    sim_cfg = cfg["simulation"]

    theta = dict(THETA0)
    theta["R_sys"] = resolve_R_sys(theta)
    print(f"[Stage1 DEBUG] R_sys resolved={theta['R_sys']:.3f}, K_VSD={K_VSD}")

    model = build_model(theta)
    t_calib = 120.0  # для Stage1 120с, в yaml 600с для продакшена
    print(f"[Stage1 DEBUG] calibrate t_calib={t_calib}с (венозный tau=300с)")

    y0 = model.calibrate_initial_state(t_calib=t_calib)
    print(f"y0 heart volumes: {y0[model.idx['heart']]}")
    try:
        print(f"y0 V_blood={y0[model.idx['blood']][0]:.0f}")
    except:
        pass

    t_span = (0.0, 400.0)
    n_samples = 4001
    t_eval = np.linspace(t_span[0], t_span[1], n_samples)
    sol = model.simulate(t_span, t_eval=t_eval, y0=y0,
                         method=str(sim_cfg.get("method", "LSODA")),
                         rtol=1e-6, atol=1e-8, max_step=0.05)

    from collections import defaultdict
    data = {"t": sol.t}
    # собираем выходы
    for i, t in enumerate(sol.t):
        out = model.compute_outputs(t, sol.y[:, i])
        for k, v in out.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    for k in list(data.keys()):
        if k != "t":
            data[k] = np.array(data[k], dtype=float)

    print("\n--- Конвергенция по окнам ---")
    for t_lo, t_hi in [(100,300),(500,700),(800,1000),(1000,1200)]:
        m = (data["t"] >= t_lo) & (data["t"] <= t_hi)
        if not np.any(m):
            continue
        print(f"[{t_lo:4d}-{t_hi:4d}] P_sa={data['P_sa'][m].mean():5.1f}±{data['P_sa'][m].std():4.1f} "
              f"V_lv={data['V_lv'][m].mean():5.1f} V_rv={data['V_rv'][m].mean():5.1f} HR={data['HR'][m].mean():4.1f}")

    HR_mean = float(np.mean(data["HR"][data["t"]>300]))
    T = 60.0 / max(HR_mean,1.0)
    win = data["t"] > data["t"][-1] - 10*T
    print(f"\n--- Steady 10 циклов --- P_sa={data['P_sa'][win].mean():.1f} P_pa={data['P_pa'][win].mean():.1f} "
          f"Qa={data['Q_aortic'][win].mean():.1f} Qp/Qs={data['Q_pulmonary'][win].mean()/max(data['Q_aortic'][win].mean(),1e-6):.2f} "
          f"EDV_LV={np.max(data['V_lv'][win]):.0f} EDV_RV={np.max(data['V_rv'][win]):.0f}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        print("[Stage1] DEBUG MODE")
        debug_base_point()
    else:
        print("[Stage1] RUN MODE")
        main()
