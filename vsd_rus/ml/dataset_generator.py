#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml/dataset_generator_v5.py

Stage 2: генерация датасета (θ, X_clean, X_noisy) для суррогата и инверсии.

Запуск:
    # smoke-тест (стратифицированно по log R_vsd)
    python -m ml.dataset_generator_v5 --quick

    # полный прогон
    python -m ml.dataset_generator_v5 --n-samples 15000 --n-jobs 8

    # продолжить прерванный прогон (готовые чанки пропускаются)
    python -m ml.dataset_generator_v5 --n-samples 15000 --n-jobs 8

    # принудительно пересчитать всё
    python -m ml.dataset_generator_v5 --n-samples 15000 --no-resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = Path(__file__).resolve().parent

for p in (str(ROOT), str(ML_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ml.stage1_identifiability import (  # type: ignore
        build_model,
        d_vsd_to_R_vsd,
        K_VSD,
    )
except ImportError:
    from stage1_identifiability import (  # type: ignore
        build_model,
        d_vsd_to_R_vsd,
        K_VSD,
    )


# =============================================================================
# 1. Конфигурация
# =============================================================================

VARY_PARAMS: list[str] = [
    "d_vsd",
    "E_max_lv",
    "E_max_rv",
    "R_sys",
    "flow_sensitivity",
    "C_sys_art",
]

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "d_vsd":            (3.0,  18.0),   # мм
    "E_max_lv":         (1.5,   3.5),
    "E_max_rv":         (0.5,   1.5),
    "R_sys":            (0.8,   1.6),
    "flow_sensitivity": (0.02,  0.15),
    "C_sys_art":        (1.2,   3.0),
}

FIXED_THETA: dict[str, float] = {
    "HR_base":  70.0,
    "V0_blood": 5000.0,
}

# Сэмплируем log-uniform по R_vsd, а d_vsd восстанавливаем.
LOG_PARAMS: set[str] = {"d_vsd"}


# =============================================================================
# 2. Утилиты преобразования d_vsd <-> R_vsd
# =============================================================================

def R_vsd_to_d_vsd(R: float) -> float:
    """Обратное к d_vsd_to_R_vsd: d = 2 * (K/R)^(1/4)."""
    if not np.isfinite(R) or R <= 0:
        return 0.0
    return float(2.0 * (K_VSD / R) ** 0.25)


def _sample_vsd_loguniform_vec(u_arr: np.ndarray,
                               d_lo: float,
                               d_hi: float) -> np.ndarray:
    """
    u_arr ∈ [0,1] → d_vsd такой, что R_vsd = d_vsd_to_R_vsd(d_vsd)
    распределён log-uniform на [R(d_hi), R(d_lo)].
    """
    R_lo = d_vsd_to_R_vsd(d_hi)   # маленькое R (большой дефект)
    R_hi = d_vsd_to_R_vsd(d_lo)   # большое R (маленький дефект)
    log_R = np.log(R_lo) + u_arr * (np.log(R_hi) - np.log(R_lo))
    R = np.exp(log_R)
    d = 2.0 * (K_VSD / R) ** 0.25
    return d


# =============================================================================
# 3. SimConfig
# =============================================================================

class SimConfig:
    """Параметры симуляции, фильтрации и шума."""

    def __init__(self, quick: bool = True):
        if quick:
            self.t_end = 180.0
            self.n_samples_t = 2000
            self.t_start_stationary = 120.0
            self.stationary_window = 30.0
            self.stationary_rel_tol = 0.025
        else:
            self.t_end = 400.0
            self.n_samples_t = 8000 # 9000
            self.t_start_stationary = 300.0
            self.stationary_window = 50.0
            self.stationary_rel_tol = 0.015

        # FIX 5 (v5): адаптивное удлинение t_end
        self.adaptive_t_end = True
        self.t_end_max = 600.0
        # FIX 5 (v5): C_sys_ven зафиксирован в build_model (whole_body.py)
        self.C_sys_ven_const = 12.0

        # FIX 8 (v5): держать dt ≈ const при удлинении t_end
        self.scale_n_samples_t_with_t_end = True

        self.n_cycles_avg = 10

        # FIX 2 (v5): технический фильтр по Qp_Qs
        self.P_sa_blowup = 300.0
        self.Qp_Qs_min = 0.05
        self.Qp_Qs_max = 10.0

        # Физиологические пороги — только для флагов
        self.thr_phtn_moderate = 25.0
        self.thr_phtn_severe = 40.0
        self.thr_sys_htn = 140.0
        self.thr_sys_hypo = 70.0
        self.thr_large_shunt = 2.0
        self.thr_dilated_lv = 250.0
        self.thr_dilated_rv = 150.0

        # Шум ЭхоКГ
        self.noise_frac = 0.05
        self.noise_keys = ("Qp_Qs", "P_pa", "EDV_LV", "EDV_RV")

        self.random_seed = 42
        self.chunk_size = 100


# =============================================================================
# 4. Валидация конфига
# =============================================================================

def validate_config() -> None:
    missing = [p for p in VARY_PARAMS if p not in PARAM_BOUNDS]
    if missing:
        raise ValueError(f"Нет границ для VARY_PARAMS: {missing}")
    bad_bounds = [p for p in VARY_PARAMS
                  if PARAM_BOUNDS[p][0] >= PARAM_BOUNDS[p][1]]
    if bad_bounds:
        raise ValueError(f"Некорректные границы (lo >= hi): {bad_bounds}")
    unknown_fixed = [p for p in FIXED_THETA if p in VARY_PARAMS]
    if unknown_fixed:
        raise ValueError(
            f"Параметры одновременно в VARY_PARAMS и FIXED_THETA: {unknown_fixed}"
        )
    bad_log = [p for p in LOG_PARAMS if p not in VARY_PARAMS]
    if bad_log:
        raise ValueError(f"LOG_PARAMS вне VARY_PARAMS: {bad_log}")


# =============================================================================
# 5. Fingerprint конфига
# =============================================================================

def config_fingerprint(cfg: SimConfig) -> str:
    """
    Хэш ключевых полей конфига. Используется в имени чанка и в имени
    итогового parquet, чтобы при изменении конфига старые файлы
    не переиспользовались молча.
    """
    payload = {
        "vary": list(VARY_PARAMS),
        "bounds": {k: list(v) for k, v in PARAM_BOUNDS.items() if k in VARY_PARAMS},
        "fixed": FIXED_THETA,
        "log_params": sorted(LOG_PARAMS),
        "seed": cfg.random_seed,
        "t_end": cfg.t_end,
        "t_end_max": cfg.t_end_max,
        "adaptive_t_end": cfg.adaptive_t_end,
        "C_sys_ven_const": cfg.C_sys_ven_const,
        "scale_n_samples_t": cfg.scale_n_samples_t_with_t_end,
        "n_samples_t": cfg.n_samples_t,
        "t_start_stat": cfg.t_start_stationary,
        "stat_win": cfg.stationary_window,
        "stat_tol": cfg.stationary_rel_tol,
        "n_cycles_avg": cfg.n_cycles_avg,
        "noise_frac": cfg.noise_frac,
        "noise_keys": list(cfg.noise_keys),
        "P_sa_blowup": cfg.P_sa_blowup,
        "Qp_Qs_min": cfg.Qp_Qs_min,
        "Qp_Qs_max": cfg.Qp_Qs_max,
        "chunk_size": cfg.chunk_size,
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:8]


# =============================================================================
# 6. Sobol-сэмплинг с log-uniform для LOG_PARAMS
# =============================================================================

def sample_sobol_all(n_samples: int, vary_params: list[str], seed: int) -> np.ndarray:
    sampler = Sobol(d=len(vary_params), scramble=True, seed=seed)
    units = sampler.random(n=n_samples)          # (n, m) ∈ [0,1)
    samples = np.empty_like(units)

    for j, p in enumerate(vary_params):
        lo, hi = PARAM_BOUNDS[p]
        u_col = units[:, j]
        if p in LOG_PARAMS and p == "d_vsd":
            samples[:, j] = _sample_vsd_loguniform_vec(u_col, lo, hi)
        else:
            samples[:, j] = lo + u_col * (hi - lo)
    return samples


def make_theta(vary_row: np.ndarray) -> dict[str, float]:
    theta = dict(FIXED_THETA)
    for name, val in zip(VARY_PARAMS, vary_row):
        theta[name] = float(val)
    return theta


# =============================================================================
# 7. Фильтры
# =============================================================================

def is_numerically_valid(X: dict[str, float], cfg: SimConfig) -> tuple[bool, str]:
    """Жёсткий технический фильтр. Единственный, который отбрасывает."""
    for k in ("P_sa", "P_pa", "EDV_LV", "EDV_RV", "Qp_Qs"):
        v = X.get(k, np.nan)
        if not np.isfinite(v):
            return False, f"non_finite:{k}"
    if X["EDV_LV"] < 0 or X["EDV_RV"] < 0:
        return False, "negative_volume"
    if X["P_sa"] <= 0 or X["P_sa"] > cfg.P_sa_blowup:
        return False, "blowup_P_sa"
    # FIX 2 (v5): sanity по Qp_Qs
    if X["Qp_Qs"] <= cfg.Qp_Qs_min:
        return False, "Qp_Qs_too_small"
    if X["Qp_Qs"] > cfg.Qp_Qs_max:
        return False, "Qp_Qs_too_large"
    return True, "ok"


def classify_physiological(X: dict[str, float], cfg: SimConfig) -> dict[str, bool]:
    """
    Возвращает набор флагов. Ничего не отбраковывает.
    Stage 4 / Stage 6 сами выберут нужное подмножество.
    """
    Qp_Qs = X["Qp_Qs"]
    flags: dict[str, bool] = {
        # FIX 7 (v5): явные флаги направления сброса
        "is_left_to_right_shunt":  Qp_Qs > 1.0,
        "is_right_to_left_shunt":  Qp_Qs < 1.0,
        "is_pulmonary_htn":        X["P_pa"] > cfg.thr_phtn_moderate,
        "is_severe_phtn":          X["P_pa"] > cfg.thr_phtn_severe,
        "is_systemic_htn":         X["P_sa"] > cfg.thr_sys_htn,
        "is_systemic_hypo":        X["P_sa"] < cfg.thr_sys_hypo,
        "is_large_shunt":          Qp_Qs > cfg.thr_large_shunt,
        "is_dilated_lv":           X["EDV_LV"] > cfg.thr_dilated_lv,
        "is_dilated_rv":           X["EDV_RV"] > cfg.thr_dilated_rv,
    }
    # FIX 7 (v5): Эйзенменгер = право-левый сброс + тяжёлая ЛГ
    flags["is_eisenmenger_like"] = (
        flags["is_right_to_left_shunt"] and flags["is_severe_phtn"]
    )
    any_pathology = any([
        flags["is_pulmonary_htn"], flags["is_systemic_htn"],
        flags["is_systemic_hypo"], flags["is_large_shunt"],
        flags["is_dilated_lv"], flags["is_dilated_rv"],
        flags["is_eisenmenger_like"],
    ])
    flags["is_healthy_range"] = not any_pathology
    return flags


# =============================================================================
# 8. Шум ЭхоКГ
# =============================================================================

def add_echo_noise(X_clean: dict[str, float],
                   cfg: SimConfig,
                   rng: np.random.Generator) -> dict[str, float]:
    X_noisy = dict(X_clean)
    for key in cfg.noise_keys:
        if key not in X_clean:
            continue
        sigma = cfg.noise_frac * abs(X_clean[key])
        X_noisy[key] = float(X_clean[key] + rng.normal(0.0, sigma))
    if "Qp_Qs" in X_noisy:
        X_noisy["Qp_Qs"] = float(np.clip(X_noisy["Qp_Qs"], 0.1, 5.0))
    if "EDV_LV" in X_noisy:
        X_noisy["EDV_LV"] = float(max(X_noisy["EDV_LV"], 20.0))
    if "EDV_RV" in X_noisy:
        X_noisy["EDV_RV"] = float(max(X_noisy["EDV_RV"], 10.0))
    return X_noisy


# =============================================================================
# 9. Свой steady-state wrapper с reason-кодом
# =============================================================================

def _steady_with_reason(model,
                        t_span: tuple[float, float],
                        n_samples: int,
                        t_start_stationary: float,
                        cfg: SimConfig) -> tuple[dict[str, float] | None, str]:
    """
    Возвращает (X, reason). reason == "ok" на успехе.

    Отличия от stage1.get_steady_outputs: возвращает reason-код,
    чтобы отделить "не сошлось по стационару" от "исключение в solve_ivp".
    """
    try:
        y0 = model.calibrate_initial_state(t_calib=10.0)
        t_eval = np.linspace(t_span[0], t_span[1], n_samples)
        sol = model.simulate(t_span, t_eval, y0=y0, method="BDF", rtol=1e-6)
    except Exception as e:
        return None, f"sim_exception:{type(e).__name__}"

    if sol.y.shape[1] < 2:
        return None, "sim_too_few_points"
    if not np.all(np.isfinite(sol.y[:, -1])):
        return None, "sim_non_finite"

    # Собираем выходы
    keys = None
    rows = []
    for i, ti in enumerate(sol.t):
        out = model.compute_outputs(ti, sol.y[:, i])
        if keys is None:
            keys = list(out.keys())
        rows.append([out[k] for k in keys])
    data = {k: np.array([r[j] for r in rows]) for j, k in enumerate(keys)}
    data["t"] = sol.t

    # Стационарность
    tail_mask = data["t"] > (t_span[1] - cfg.stationary_window)
    ps_tail = data["P_sa"][tail_mask]
    if ps_tail.size == 0:
        return None, "stationary_empty_tail"
    ps_mean = float(np.mean(ps_tail))
    if ps_mean <= 0:
        return None, "stationary_nonpositive_P_sa"
    rel_std = float(np.std(ps_tail) / ps_mean)
    if rel_std > cfg.stationary_rel_tol:
        return None, f"stationary_rel_std_{rel_std:.3f}"

    # Усреднение за последние N циклов
    tail = data["t"] > t_start_stationary
    HR_mean = float(np.mean(data["HR"][tail])) if np.any(tail) else 70.0
    T = 60.0 / max(HR_mean, 1e-6)
    window = data["t"] > (data["t"][-1] - cfg.n_cycles_avg * T)

    X: dict[str, float] = {}
    for key in ("P_sa", "P_pa", "Q_aortic", "Qp_Qs", "HR"):
        X[key] = float(np.mean(data[key][window]))
    X["EDV_LV"] = float(np.max(data["V_lv"][window]))
    X["EDV_RV"] = float(np.max(data["V_rv"][window]))

    # FIX 6 (v5): CO = Q_aortic (мл/с), плюс явный вариант в л/мин.
    # Модель оперирует мл/с, поэтому CO в мл/с = Q_aortic.
    # CO_L_min = Q_aortic * 60 / 1000 — на случай клинической интерпретации.
    X["CO"] = X["Q_aortic"]                              # мл/с
    X["CO_L_min"] = X["Q_aortic"] * 60.0 / 1000.0        # л/мин

    return X, "ok"


# =============================================================================
# 10. Адаптивные t_end и n_samples_t
# =============================================================================

def _effective_t_end(theta: dict[str, float], cfg: SimConfig) -> float:
    """
    FIX 5 (v5): удлиняем t_end для "медленных" θ.
    Доминирующая постоянная времени: max(C_sys_art, C_sys_ven_const) * R_sys.
    Reference: C_art=2.0, C_ven=12.0, R_sys=1.2 → tau_ref = 14.4 c.
    """
    if not cfg.adaptive_t_end:
        return cfg.t_end
    R_sys = float(theta.get("R_sys", 1.2))
    C_art = float(theta.get("C_sys_art", 2.0))
    C_ven = float(cfg.C_sys_ven_const)
    tau_est = max(C_art, C_ven) * R_sys
    tau_ref = max(2.0, C_ven) * 1.2
    factor = float(np.clip(tau_est / tau_ref, 1.0, 1.5))
    return min(cfg.t_end * factor, cfg.t_end_max)


def _effective_n_samples_t(t_end_eff: float, cfg: SimConfig) -> int:
    """
    FIX 8 (v5): держим dt ≈ const. Если t_end вырос в 1.3x, то
    n_samples_t тоже растёт в 1.3x — точность не падает.
    """
    if not cfg.scale_n_samples_t_with_t_end:
        return cfg.n_samples_t
    factor = t_end_eff / max(cfg.t_end, 1e-9)
    return max(int(round(cfg.n_samples_t * factor)), 2000)


# =============================================================================
# 11. Один прогон
# =============================================================================

def simulate_one(idx: int, param_row: np.ndarray, cfg: SimConfig) -> dict[str, Any]:
    theta = make_theta(param_row)
    seed = cfg.random_seed + idx
    rng = np.random.default_rng(seed)

    record: dict[str, Any] = {
        "idx": idx,
        "seed": seed,
        **{f"theta_{k}": theta[k] for k in VARY_PARAMS},
        "R_vsd": float(d_vsd_to_R_vsd(theta["d_vsd"])),
        "status": "unknown",
        "reason": "",
    }

    t_end_eff = _effective_t_end(theta, cfg)
    n_samples_t_eff = _effective_n_samples_t(t_end_eff, cfg)
    record["t_end_eff"] = float(t_end_eff)
    record["n_samples_t_eff"] = int(n_samples_t_eff)

    # Сборка модели
    try:
        model = build_model(theta)
    except Exception as e:
        record["status"] = "failed"
        record["reason"] = f"build_exception:{type(e).__name__}"
        return record

    # FIX 3: получаем reason-код
    X_clean, reason = _steady_with_reason(
        model, (0.0, t_end_eff), n_samples_t_eff,
        cfg.t_start_stationary, cfg,
    )

    if X_clean is None:
        record["status"] = "reject"
        record["reason"] = reason
        return record

    # FIX 2: технический фильтр — единственный, который отбрасывает
    ok_num, reason_num = is_numerically_valid(X_clean, cfg)
    if not ok_num:
        record["status"] = "reject"
        record["reason"] = reason_num
        for k, v in X_clean.items():
            record[f"X_clean_{k}"] = v
        return record

    # Физиологические флаги — только колонки
    flags = classify_physiological(X_clean, cfg)

    # Шум ЭхоКГ
    X_noisy = add_echo_noise(X_clean, cfg, rng)

    record["status"] = "ok"
    record["reason"] = "ok"
    for k, v in X_clean.items():
        record[f"X_clean_{k}"] = v
    for k, v in X_noisy.items():
        record[f"X_noisy_{k}"] = v
    for k, v in flags.items():
        record[f"flag_{k}"] = bool(v)

    return record


# =============================================================================
# 12. Основной цикл с чанками и resume
# =============================================================================

def generate_chunked(n_samples: int,
                     cfg: SimConfig,
                     n_jobs: int,
                     out_dir: Path,
                     fp: str,
                     resume: bool = True,
                     verbose: bool = True) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = sample_sobol_all(n_samples, VARY_PARAMS, cfg.random_seed)

    chunk_dfs: list[pd.DataFrame] = []
    t_start_all = time.time()

    for start in range(0, n_samples, cfg.chunk_size):
        end = min(start + cfg.chunk_size, n_samples)
        # FIX 1: fingerprint входит в имя чанка
        chunk_path = out_dir / f"stage2_chunk_{fp}_{start:06d}_{end:06d}.parquet"

        if resume and chunk_path.exists():
            if verbose:
                print(f"[Stage2] [skip] {chunk_path.name}")
            chunk_dfs.append(pd.read_parquet(chunk_path))
            continue

        if verbose:
            print(f"[Stage2] Chunk {start:>6d}-{end:<6d}  ({end - start} точек)")

        chunk_samples = all_samples[start:end]
        t0 = time.time()
        results = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")(
            delayed(simulate_one)(start + i, chunk_samples[i], cfg)
            for i in range(end - start)
        )
        dt = time.time() - t0

        df_chunk = pd.DataFrame(results)
        # Атомарная запись: .tmp → replace
        tmp_path = chunk_path.with_suffix(".parquet.tmp")
        df_chunk.to_parquet(tmp_path, index=False)
        tmp_path.replace(chunk_path)

        if verbose:
            n_ok = int((df_chunk["status"] == "ok").sum())
            print(f"           {dt:6.1f}s  ({dt / (end - start):.2f} с/точку), "
                  f"ok={n_ok}/{len(df_chunk)}  → {chunk_path.name}")

        chunk_dfs.append(df_chunk)

    dt_all = time.time() - t_start_all
    if verbose:
        print(f"[Stage2] Всего чанков: {len(chunk_dfs)}, "
              f"общее время {dt_all / 60:.1f} мин")

    return pd.concat(chunk_dfs, ignore_index=True)


# =============================================================================
# 13. Диагностика (гистограмма R_vsd)
# =============================================================================

def _text_log_histogram(values: np.ndarray, n_bins: int = 10) -> str:
    """Простая текстовая гистограмма в log10-пространстве."""
    v = values[np.isfinite(values) & (values > 0)]
    if v.size == 0:
        return "  (пусто)"
    log_v = np.log10(v)
    hist, edges = np.histogram(log_v, bins=n_bins)
    max_h = max(hist.max(), 1)
    lines = []
    for i in range(n_bins):
        lo = 10 ** edges[i]
        hi = 10 ** edges[i + 1]
        bar = "#" * int(round(40 * hist[i] / max_h))
        lines.append(f"  [{lo:7.4g}, {hi:7.4g}]  {bar} {hist[i]}")
    return "\n".join(lines)


def report_diagnostics(df: pd.DataFrame, cfg: SimConfig,
                       out_dir: Path, fp: str) -> None:
    print("\n" + "=" * 70)
    print("Stage 2 — диагностика")
    print("=" * 70)
    print(f"[Stage2] config fingerprint: {fp}")

    r_min = float(d_vsd_to_R_vsd(PARAM_BOUNDS["d_vsd"][1]))
    r_max = float(d_vsd_to_R_vsd(PARAM_BOUNDS["d_vsd"][0]))
    print(f"\n[Stage2] R_vsd диапазон (из d_vsd ∈ "
          f"[{PARAM_BOUNDS['d_vsd'][0]}, {PARAM_BOUNDS['d_vsd'][1]}] мм):")
    print(f"          R_vsd ∈ [{r_min:.4f}, {r_max:.4f}]  "
          f"(в ML_Tasks.txt ожидалось [0.5, 15])")
    print(f"          sampling: log-uniform по R_vsd для d_vsd "
          f"(см. LOG_PARAMS = {sorted(LOG_PARAMS)})")

    print("\n[Stage2] Статусы:")
    print(df["status"].value_counts().to_string())

    print("\n[Stage2] Причины reject/failed (top-15):")
    bad = df.loc[df["status"] != "ok", "reason"].value_counts().head(15)
    print(bad.to_string() if len(bad) else "  (нет отбракованных)")

    n_ok = int((df["status"] == "ok").sum())
    print(f"\n[Stage2] Валидных точек (status=ok): "
          f"{n_ok}/{len(df)}  ({100 * n_ok / max(len(df), 1):.1f}%)")

    # FIX 7 + FIX 4: гистограмма R_vsd
    if "R_vsd" in df.columns:
        R_all = df["R_vsd"].to_numpy()
        print(f"\n[Stage2] Гистограмма R_vsd (все строки, log10):")
        print(_text_log_histogram(R_all, n_bins=10))

    if n_ok > 0:
        ok_df = df[df["status"] == "ok"]
        flag_cols = [c for c in ok_df.columns if c.startswith("flag_")]
        print("\n[Stage2] Доли флагов среди валидных:")
        for c in flag_cols:
            frac = float(ok_df[c].mean())
            print(f"          {c:30s} {frac * 100:5.1f}%")

        cols = [c for c in df.columns if c.startswith("X_clean_")]
        print("\n[Stage2] X_clean (только ok):")
        print(df.loc[df["status"] == "ok", cols].describe().round(3).to_string())

    # Сохранение отчёта
    report = {
        "config_fingerprint": fp,
        "n_total": int(len(df)),
        "n_ok": n_ok,
        "R_vsd_range_actual": [r_min, r_max],
        "R_vsd_range_planned": [0.5, 15.0],
        "log_params": sorted(LOG_PARAMS),
        "t_end": cfg.t_end,
        "t_end_max": cfg.t_end_max,
        "adaptive_t_end": cfg.adaptive_t_end,
        "C_sys_ven_const": cfg.C_sys_ven_const,
        "scale_n_samples_t": cfg.scale_n_samples_t_with_t_end,
        "status_counts": df["status"].value_counts().to_dict(),
        "reason_counts": df.loc[df["status"] != "ok", "reason"]
                            .value_counts().to_dict(),
    }
    if n_ok > 0:
        ok_df = df[df["status"] == "ok"]
        flag_cols = [c for c in ok_df.columns if c.startswith("flag_")]
        report["flag_fractions"] = {c: float(ok_df[c].mean()) for c in flag_cols}

    out_path = out_dir / f"stage2_diagnostics_{fp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[Stage2] Диагностика сохранена: {out_path}")


# =============================================================================
# 14. Quick stratified (стратификация по log R_vsd)
# =============================================================================

def make_quick_stratified_samples(n_bins: int = 6, per_bin: int = 6) -> np.ndarray:
    """
    Стратифицированно по log(R_vsd): по 6 бинов × 6 точек.
    Покрывает весь диапазон R_vsd, включая крупные дефекты.
    """
    d_lo, d_hi = PARAM_BOUNDS["d_vsd"]
    R_hi = d_vsd_to_R_vsd(d_lo)     # маленькое d → большое R
    R_lo = d_vsd_to_R_vsd(d_hi)     # большое d → маленькое R

    log_edges = np.linspace(np.log(R_lo), np.log(R_hi), n_bins + 1)
    log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])

    rng = np.random.default_rng(0)
    rows = []
    for log_R in log_centers:
        R = float(np.exp(log_R))
        d_val = float(R_vsd_to_d_vsd(R))
        for _ in range(per_bin):
            row = []
            for p in VARY_PARAMS:
                lo, hi = PARAM_BOUNDS[p]
                if p == "d_vsd":
                    row.append(d_val)
                else:
                    row.append(float(rng.uniform(lo, hi)))
            rows.append(row)
    return np.array(rows)


# =============================================================================
# 15. CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: dataset generator v5"
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--quick", action="store_true",
                        help="smoke-test: 36 точек, стратифицированно по log R_vsd")
    parser.add_argument("--no-resume", action="store_true",
                        help="не использовать готовые чанки (пересчитать всё)")
    parser.add_argument("--output", type=str, default=None,
                        help="итоговый parquet (по умолчанию results/stage2_dataset_<fp>_<n>.parquet)")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--n-samples-t", type=int, default=None,
                        help="переопределить число точек t_eval (по умолчанию cfg)")
    args = parser.parse_args()

    validate_config()

    cfg = SimConfig(quick=args.quick)
    cfg.chunk_size = args.chunk_size
    # cfg = SimConfig(quick=True)
    if args.n_samples_t is not None:
        cfg.n_samples_t = args.n_samples_t
        cfg.scale_n_samples_t_with_t_end = False  # ручное значение — не масштабируем

    fp = config_fingerprint(cfg)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем конфиг с отпечатком
    cfg_dump = {
        "fingerprint": fp,
        "vary_params": VARY_PARAMS,
        "param_bounds": {k: list(v) for k, v in PARAM_BOUNDS.items()},
        "fixed_theta": FIXED_THETA,
        "log_params": sorted(LOG_PARAMS),
        "sim_config": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in vars(cfg).items()
        },
    }
    (out_dir / f"stage2_config_{fp}.json").write_text(
        json.dumps(cfg_dump, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 70)
    print("Stage 2 — генерация датасета (θ → X_clean / X_noisy)  [v5]")
    print("=" * 70)
    print(f"fingerprint : {fp}")
    print(f"VARY_PARAMS : {VARY_PARAMS}")
    print(f"LOG_PARAMS  : {sorted(LOG_PARAMS)}")
    print(f"FIXED_THETA : {FIXED_THETA}")
    print(f"t_end       : {cfg.t_end} (max {cfg.t_end_max}, "
          f"adaptive={cfg.adaptive_t_end}, C_ven_const={cfg.C_sys_ven_const})")
    print(f"n_samples_t : {cfg.n_samples_t} "
          f"(scale_with_t_end={cfg.scale_n_samples_t_with_t_end})")
    print(f"chunk_size  : {cfg.chunk_size}")
    print(f"quick       : {args.quick}")

    r_min = d_vsd_to_R_vsd(PARAM_BOUNDS["d_vsd"][1])
    r_max = d_vsd_to_R_vsd(PARAM_BOUNDS["d_vsd"][0])
    print(f"R_vsd range : [{r_min:.4f}, {r_max:.4f}]  "
          f"(ожидание из плана: [0.5, 15])")

    t0 = time.time()

    if args.quick:
        samples = make_quick_stratified_samples(n_bins=6, per_bin=6)
        n_samples = len(samples)
        print(f"[Stage2] QUICK: {n_samples} точек, стратифицированно по log R_vsd")
        results = Parallel(n_jobs=args.n_jobs, verbose=0, backend="loky")(
            delayed(simulate_one)(i, samples[i], cfg) for i in range(n_samples)
        )
        df = pd.DataFrame(results)
    else:
        df = generate_chunked(
            n_samples=args.n_samples,
            cfg=cfg,
            n_jobs=args.n_jobs,
            out_dir=out_dir,
            fp=fp,
            resume=not args.no_resume,
            verbose=True,
        )

    dt = time.time() - t0
    print(f"\n[Stage2] Общее время: {dt / 60:.1f} мин")

    report_diagnostics(df, cfg, out_dir, fp)

    # Финальный parquet
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = out_dir / f"stage2_dataset_{fp}_{len(df)}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n[Stage2] Итоговый датасет: {out_path}  "
          f"({out_path.stat().st_size / 1e6:.1f} МБ)")

    preview_path = out_dir / f"stage2_dataset_preview_{fp}.csv"
    df.head(50).to_csv(preview_path, index=False)
    print(f"[Stage2] Превью (50 строк): {preview_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
