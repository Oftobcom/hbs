#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_vsd_comparison.py
Расширенная визуализация для сравнения гемодинамики при ДМЖП.

Загружает .npz, сгенерированные run_simulation.py, и строит набор
диагностических фигур:

    fig1_hemodynamics_timeseries.png   — временные ряды
    fig2_phase_portraits.png           — фазовые PV-портреты (если есть P_lv/P_rv)
    fig3_bar_comparison.png            — столбчатое сравнение установившихся метрик
    fig4_detailed_cardiac.png          — детальный анализ сердца и регионов
    comprehensive_dashboard.png        — комплексный дашборд 4×4
    schematic_heart_comparison.png     — схематическая диаграмма

Ожидаемые колонки в .npz (обязательные помечены *):
    t*, P_sa*, P_pa*, P_sv, P_pv,
    Q_aortic*, Q_pulmonary*, Q_vsd*, Qp_Qs*,
    V_lv*, V_rv*, V_blood*, HR*,
    GFR, Q_brain, O2_consumption,
    SaO2, shunt_fraction_R2L, P_a_O2, P_v_O2, C_a_O2, C_v_O2,
    P_lv, P_rv, P_la, P_ra  (опционально — нужны для PV-петель)

Совместим с matplotlib 3.4+ (стиль задаётся безопасно).
"""

from __future__ import annotations

import glob
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from scipy.stats import linregress
from utils import safe_savgol_filter as safe_savgol
from utils import subsample, steady_mask
from physio_config import load_all_patients

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Стиль и палитра
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    """Безопасная настройка стиля (не падает, если стиль недоступен)."""
    for style in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "ggplot", "default"):
        try:
            plt.style.use(style)
            break
        except (OSError, ValueError):
            continue
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.autolayout": False,
    })

_setup_style()
_PATIENTS = load_all_patients()
COLORS: Dict[str, str] = {p['label']: p['color'] for p in _PATIENTS.values()}
SCENARIO_ORDER = [p['label'] for p in
                  sorted(_PATIENTS.values(), key=lambda c: int(c['order']))]

# ===========================================================================
# Утилиты
# ===========================================================================

def _color(name: str) -> str:
    return COLORS.get(name, "gray")


def _short(s: str, n: int = 22) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def has_field(data: dict, *keys: str) -> bool:
    """True, если все ключи присутствуют и непусты."""
    for k in keys:
        if k not in data:
            return False
        arr = data[k]
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return False
    return True


def steady_mean_std(data: dict, key: str, scale: float = 1.0
                    ) -> Optional[Tuple[float, float]]:
    """Среднее ± std по установившемуся окну. None — если данных нет."""
    if key not in data:
        return None
    m = steady_mask(data) & np.isfinite(np.asarray(data[key]))
    vals = np.asarray(data[key])[m]
    if vals.size == 0:
        return None
    return float(np.mean(vals) * scale), float(np.std(vals) * scale)


# ===========================================================================
# Загрузка
# ===========================================================================

def load_all_results(pattern: str = "vsd_results_*.npz"
                     ) -> Optional[Dict[str, dict]]:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"❌ Не найдено файлов по маске {pattern!r}.")
        print("   Сначала запустите run_simulation.py.")
        return None

    print(f"Найдено файлов: {len(files)}")
    results: Dict[str, dict] = {}

    for path in files:
        try:
            with np.load(path, allow_pickle=True) as npz:
                # label из метаданных npz; fallback — имя файла
                label = str(npz['label']) if 'label' in npz.files else path
                payload = {k: np.array(npz[k]) for k in npz.files
                           if k not in ('label', 'id', 'description')}
            results[label] = payload
            print(f"✓ {path:40s} → {label}")
        except Exception as exc:
            print(f"✗ Ошибка загрузки {path}: {exc}")

    if not results:
        print("❌ Не удалось загрузить ни одного файла.")
        return None

    # Упорядочим по SCENARIO_ORDER, остальное — в конец
    ordered: Dict[str, dict] = {}
    for key in SCENARIO_ORDER:
        if key in results:
            ordered[key] = results.pop(key)
    ordered.update(results)

    print(f"\nЗагружено сценариев: {list(ordered.keys())}")
    return ordered


# ===========================================================================
# Рис. 1 — Временные ряды
# ===========================================================================

def plot_hemodynamic_timeseries(results: Dict[str, dict]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Гемодинамика при ДМЖП: временные ряды",
                 fontsize=14, fontweight="bold")

    metrics = [
        ("P_sa",        "Системное АД (мм рт. ст.)",  axes[0, 0]),
        ("P_pa",        "Лёгочное АД (мм рт. ст.)",   axes[0, 1]),
        ("Q_aortic",    "Системный выброс (мл/с)",    axes[0, 2]),
        ("Q_pulmonary", "Лёгочный кровоток (мл/с)",   axes[1, 0]),
        ("Qp_Qs",       "Qp / Qs",                    axes[1, 1]),
        ("Q_vsd",       "Шунт VSD (мл/с)",            axes[1, 2]),
        ("V_lv",        "Объём ЛЖ (мл)",              axes[2, 0]),
        ("V_rv",        "Объём ПЖ (мл)",              axes[2, 1]),
        ("SaO2",        "SaO₂ (%)",                   axes[2, 2]),
    ]

    for metric, ylabel, ax in metrics:
        for scenario, data in results.items():
            if not has_field(data, "t", metric):
                continue
            t, y = subsample(data, metric)
            if metric == "SaO2":
                y = y * 100.0
            mask = np.isfinite(y)
            if not np.any(mask):
                continue
            # Для Qp/Qs и SaO2 наложим лёгкое сглаживание
            if metric in ("Qp_Qs", "SaO2") and mask.sum() > 100:
                y_plot = np.array(y, dtype=float)
                y_plot[mask] = safe_savgol(y_plot[mask], 101, 3)
            else:
                y_plot = y
            ax.plot(t[mask], y_plot[mask],
                    color=_color(scenario), lw=1.5, label=scenario)

        ax.set_xlabel("Время (с)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if metric == "SaO2":
            ax.axhline(90, color="orange", ls=":", alpha=0.6)
        if metric == "Qp_Qs":
            ax.axhline(1.0, color="black", ls="--", alpha=0.4)

    # Одна общая легенда
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right",
           bbox_to_anchor=(0.99, 0.95), fontsize=8, ncol=2)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig("fig1_hemodynamics_timeseries.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig1_hemodynamics_timeseries.png")


# ===========================================================================
# Рис. 2 — Фазовые PV-портреты
# ===========================================================================

def _last_cardiac_cycle(data: dict, n_cycles: float = 2.0) -> slice:
    """Возвращает slice на последние ~n_cycles кардиоциклов."""
    t = np.asarray(data.get("t", []))
    if t.size < 4:
        return slice(0, t.size)
    hr = float(np.mean(data["HR"][-min(50, t.size):])) if "HR" in data else 70.0
    hr = max(hr, 20.0)
    T = 60.0 / hr
    t_start = t[-1] - n_cycles * T
    idx0 = int(np.searchsorted(t, t_start))
    return slice(idx0, t.size)


def plot_phase_portraits(results: Dict[str, dict]) -> None:
    have_real = any(has_field(d, "V_lv", "P_lv", "V_rv", "P_rv")
                    for d in results.values())
    if not have_real:
        print("  ⚠ P_lv/P_rv отсутствуют — PV-петли пропущены.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Фазовые портреты желудочков (последние ~2 цикла)",
                 fontsize=13, fontweight="bold")

    for scenario, data in results.items():
        if not has_field(data, "V_lv", "P_lv", "V_rv", "P_rv"):
            continue
        sl = _last_cardiac_cycle(data, n_cycles=2.0)
        V_lv = np.asarray(data["V_lv"])[sl]
        P_lv = np.asarray(data["P_lv"])[sl]
        V_rv = np.asarray(data["V_rv"])[sl]
        P_rv = np.asarray(data["P_rv"])[sl]

        mask_lv = np.isfinite(V_lv) & np.isfinite(P_lv)
        mask_rv = np.isfinite(V_rv) & np.isfinite(P_rv)
        if np.any(mask_lv):
            axes[0].plot(V_lv[mask_lv], P_lv[mask_lv],
                         color=_color(scenario), lw=1.6, alpha=0.9,
                         label=scenario)
        if np.any(mask_rv):
            axes[1].plot(V_rv[mask_rv], P_rv[mask_rv],
                         color=_color(scenario), lw=1.6, alpha=0.9,
                         label=scenario)

    axes[0].set_xlabel("Объём ЛЖ (мл)")
    axes[0].set_ylabel("Давление ЛЖ (мм рт. ст.)")
    axes[0].set_title("Левый желудочек")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")

    axes[1].set_xlabel("Объём ПЖ (мл)")
    axes[1].set_ylabel("Давление ПЖ (мм рт. ст.)")
    axes[1].set_title("Правый желудочек")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig("fig2_phase_portraits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig2_phase_portraits.png")


# ===========================================================================
# Рис. 3 — Столбчатое сравнение
# ===========================================================================

def plot_bar_comparison(results: Dict[str, dict]) -> None:
    metrics = [
        ("P_sa",     "Системное АД (мм рт. ст.)", 1.0),
        ("P_pa",     "Лёгочное АД (мм рт. ст.)",  1.0),
        ("Q_aortic", "Системный выброс (мл/с)",   1.0),
        ("Qp_Qs",    "Qp / Qs",                   1.0),
        ("V_rv",     "Объём ПЖ (мл)",             1.0),
        ("V_blood",  "Объём крови (мл)",          1.0),
        ("GFR",      "СКФ (мл/с)",                1.0),
        ("SaO2",     "SaO₂ (%)",                  100.0),
    ]
    scenarios = list(results.keys())
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle("Сравнение установившихся показателей",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ax, (metric, ylabel, scale) in zip(axes, metrics):
        means, stds = [], []
        for sc in scenarios:
            r = steady_mean_std(results[sc], metric, scale)
            if r is None:
                means.append(0.0)
                stds.append(0.0)
            else:
                means.append(r[0])
                stds.append(r[1])

        bars = ax.bar(scenarios, means,
                      color=[_color(s) for s in scenarios],
                      alpha=0.75, edgecolor="black", linewidth=1)
        ax.errorbar(scenarios, means, yerr=stds, fmt="none",
                    ecolor="black", capsize=4, capthick=1)
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=40, labelsize=7)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")
        ax.grid(True, alpha=0.3, axis="y")

        ymax = max([abs(m) for m in means] + [1e-6])
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.02,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig("fig3_bar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig3_bar_comparison.png")


# ===========================================================================
# Рис. 4 — Детальный анализ
# ===========================================================================

def plot_cardiovascular_parameters(results: Dict[str, dict]) -> None:
    scenarios = list(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Детальный анализ сердечно-сосудистых параметров",
                 fontsize=14, fontweight="bold")

    # --- 1. Qp/Qs во времени ---
    ax = axes[0, 0]
    for sc, data in results.items():
        if not has_field(data, "t", "Qp_Qs"):
            continue
        t, y = subsample(data, "Qp_Qs", 4000)
        mask = np.isfinite(y)
        if not np.any(mask):
            continue
        y_plot = np.array(y, dtype=float)
        if mask.sum() > 100:
            y_plot[mask] = safe_savgol(y_plot[mask], 101, 3)
        ax.plot(t[mask], y_plot[mask], color=_color(sc), lw=1.6, label=sc)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="Норма (1.0)")
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Qp / Qs")
    ax.set_title("Соотношение Qp/Qs")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- 2. Объём ЛЖ ---
    ax = axes[0, 1]
    for sc, data in results.items():
        if not has_field(data, "t", "V_lv"):
            continue
        r = steady_mean_std(data, "V_lv")
        if r is not None:
            ax.axhline(r[0], color=_color(sc), ls="--", alpha=0.7,
                        label=f"{_short(sc, 14)}: {r[0]:.0f} мл")
        t, y = subsample(data, "V_lv", 4000)
        mask = np.isfinite(y)
        if np.any(mask):
            ax.plot(t[mask], y[mask], color=_color(sc), lw=0.8, alpha=0.4)
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Объём ЛЖ (мл)")
    ax.set_title("Объём левого желудочка")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- 3. Объём ПЖ ---
    ax = axes[0, 2]
    for sc, data in results.items():
        if not has_field(data, "t", "V_rv"):
            continue
        r = steady_mean_std(data, "V_rv")
        if r is not None:
            ax.axhline(r[0], color=_color(sc), ls="--", alpha=0.7,
                        label=f"{_short(sc, 14)}: {r[0]:.0f} мл")
        t, y = subsample(data, "V_rv", 4000)
        mask = np.isfinite(y)
        if np.any(mask):
            ax.plot(t[mask], y[mask], color=_color(sc), lw=0.8, alpha=0.4)
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Объём ПЖ (мл)")
    ax.set_title("Объём правого желудочка")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- 4. Корреляция Q_vsd ↔ Qp/Qs ---
    ax = axes[1, 0]
    xs, ys = [], []
    for sc, data in results.items():
        if not has_field(data, "Q_vsd", "Qp_Qs"):
            continue
        mask = steady_mask(data) & np.isfinite(data["Q_vsd"]) \
            & np.isfinite(data["Qp_Qs"])
        if not np.any(mask):
            continue
        q_vsd = np.asarray(data["Q_vsd"])[mask]
        q_ratio = np.asarray(data["Qp_Qs"])[mask]
        step = max(1, q_vsd.size // 500)
        ax.scatter(q_vsd[::step], q_ratio[::step],
                   c=_color(sc), s=10, alpha=0.5, label=sc)
        if abs(np.mean(q_vsd)) > 1.0:
            xs.append(float(np.mean(q_vsd)))
            ys.append(float(np.mean(q_ratio)))
    if len(xs) > 1:
        slope, intercept, r_val, _, _ = linregress(xs, ys)
        x_line = np.linspace(min(xs), max(xs), 50)
        ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.6,
                label=f"R² = {r_val ** 2:.3f}")
    ax.axvline(0, color="black", ls="--", alpha=0.5)
    ax.set_xlabel("Шунт VSD (мл/с)")
    ax.set_ylabel("Qp / Qs")
    ax.set_title("Корреляция: шунт ↔ Qp/Qs")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- 5. Сравнение объёмов желудочков ---
    ax = axes[1, 1]
    chambers = [("V_lv", "ЛЖ"), ("V_rv", "ПЖ")]
    x = np.arange(len(chambers))
    width = 0.8 / max(len(scenarios), 1)
    for i, sc in enumerate(scenarios):
        vals = []
        for key, _ in chambers:
            r = steady_mean_std(results[sc], key)
            vals.append(r[0] if r is not None else 0.0)
        offset = (i - (len(scenarios) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=sc,
               color=_color(sc), alpha=0.75)
    ax.set_ylabel("Объём (мл)")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in chambers])
    ax.set_title("Сравнение объёмов желудочков")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # --- 6. Региональное распределение кровотока ---
    ax = axes[1, 2]
    organs = ["Мозг", "Почки", "Печень", "ЖКТ"]
    organ_keys = ["Q_brain", "Q_renal", "Q_liver_out", "Q_gitract_out"]
    for sc, data in results.items():
        flows = []
        for key in organ_keys:
            r = steady_mean_std(data, key)
            flows.append(r[0] if r is not None else 0.0)
        ax.plot(organs, flows, "o-", color=_color(sc),
                lw=2, markersize=8, label=sc)
    ax.set_ylabel("Кровоток (мл/с)")
    ax.set_title("Региональное распределение кровотока")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig("fig4_detailed_cardiac.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig4_detailed_cardiac.png")


# ===========================================================================
# Рис. 5 — Комплексный дашборд
# ===========================================================================

def _plot_series(ax, results, key, ylabel, title,
                 scale=1.0, smooth=False, hlines=()):
    """Универсальный помощник для временного ряда на оси."""
    for sc, data in results.items():
        if not has_field(data, "t", key):
            continue
        t, y = subsample(data, key, 4000)
        y = y * scale
        mask = np.isfinite(y)
        if not np.any(mask):
            continue
        y_plot = np.array(y, dtype=float)
        if smooth and mask.sum() > 100:
            y_plot[mask] = safe_savgol(y_plot[mask], 101, 3)
        ax.plot(t[mask], y_plot[mask], color=_color(sc), lw=1.6, label=sc)
    for y0, col, ls in hlines:
        ax.axhline(y0, color=col, ls=ls, alpha=0.5)
    ax.set_xlabel("Время (с)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_bars(ax, results, keys, labels, ylabel, title):
    """Группированные столбцы по нескольким метрикам."""
    scenarios = list(results.keys())
    x = np.arange(len(keys))
    width = 0.8 / max(len(scenarios), 1)
    for i, sc in enumerate(scenarios):
        vals = []
        for key in keys:
            r = steady_mean_std(results[sc], key)
            vals.append(r[0] if r is not None else 0.0)
        offset = (i - (len(scenarios) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=sc,
               color=_color(sc), alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


def plot_comprehensive_dashboard(results: Dict[str, dict]) -> None:
    scenarios = list(results.keys())
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Qp/Qs
    ax1 = fig.add_subplot(gs[0, :2])
    _plot_series(ax1, results, "Qp_Qs", "Qp / Qs",
                 "Соотношение лёгочного и системного кровотока",
                 smooth=True,
                 hlines=[(1.0, "black", "--"),
                         (1.5, "orange", ":"),
                         (2.0, "red", ":")])

    # 2. Давления
    ax2 = fig.add_subplot(gs[0, 2])
    _plot_bars(ax2, results,
               ["P_sa", "P_pa"], ["Сист. АД", "Лёг. АД"],
               "Давление (мм рт. ст.)", "Сравнение давлений")

    # 3. SaO₂
    ax3 = fig.add_subplot(gs[0, 3])
    _plot_series(ax3, results, "SaO2", "SaO₂ (%)",
                 "Артериальная сатурация O₂",
                 scale=100.0, smooth=True,
                 hlines=[(90, "orange", ":")])
    ax3.set_ylim(60, 100)

    # 4. Объёмы желудочков
    ax4 = fig.add_subplot(gs[1, 0])
    _plot_bars(ax4, results,
               ["V_lv", "V_rv"], ["ЛЖ", "ПЖ"],
               "Объём (мл)", "Объёмы желудочков")

    # 5. Региональные кровотоки
    ax5 = fig.add_subplot(gs[1, 1])
    organs = ["Мозг", "Почки", "Печень", "ЖКТ"]
    organ_keys = ["Q_brain", "Q_renal", "Q_liver_out", "Q_gitract_out"]
    for sc, data in results.items():
        flows = []
        for key in organ_keys:
            r = steady_mean_std(data, key)
            flows.append(r[0] if r is not None else 0.0)
        ax5.plot(organs, flows, "o-", color=_color(sc),
                 lw=2, markersize=8, label=sc)
    ax5.set_ylabel("Кровоток (мл/с)")
    ax5.set_title("Региональное распределение кровотока",
                  fontsize=11, fontweight="bold")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    # 6. Объём крови
    ax6 = fig.add_subplot(gs[1, 2])
    _plot_series(ax6, results, "V_blood", "Объём крови (мл)",
                 "Объём циркулирующей крови")

    # 7. Фракция R→L
    ax7 = fig.add_subplot(gs[1, 3])
    _plot_series(ax7, results, "shunt_fraction_R2L", "R→L шунт (%)",
                 "Фракция право-левого шунта",
                 scale=100.0,
                 hlines=[(10, "orange", ":")])

    # 8. Qp/Qs vs Q_vsd
    ax8 = fig.add_subplot(gs[2, 0])
    xs, ys = [], []
    for sc, data in results.items():
        if not has_field(data, "Q_vsd", "Qp_Qs"):
            continue
        r_vsd = steady_mean_std(data, "Q_vsd")
        r_qp = steady_mean_std(data, "Qp_Qs")
        if r_vsd is None or r_qp is None:
            continue
        if abs(r_vsd[0]) > 1.0:
            xs.append(r_vsd[0])
            ys.append(r_qp[0])
            ax8.scatter(r_vsd[0], r_qp[0], s=120,
                        c=_color(sc), edgecolor="black",
                        linewidth=1.5, label=sc)
    if len(xs) > 1:
        slope, intercept, r_val, _, _ = linregress(xs, ys)
        x_line = np.linspace(min(xs), max(xs), 50)
        ax8.plot(x_line, slope * x_line + intercept,
                 "k--", alpha=0.6, label=f"R² = {r_val ** 2:.3f}")
    ax8.axvline(0, color="black", ls="--", alpha=0.5)
    ax8.set_xlabel("Шунт VSD (мл/с)")
    ax8.set_ylabel("Qp / Qs")
    ax8.set_title("Корреляция: шунт → Qp/Qs",
                  fontsize=11, fontweight="bold")
    ax8.legend(fontsize=7)
    ax8.grid(True, alpha=0.3)

    # 9. SaO2 vs Qp/Qs
    ax9 = fig.add_subplot(gs[2, 1])
    for sc, data in results.items():
        if not has_field(data, "SaO2", "Qp_Qs"):
            continue
        mask = steady_mask(data) & np.isfinite(data["SaO2"]) \
            & np.isfinite(data["Qp_Qs"])
        if not np.any(mask):
            continue
        q = np.asarray(data["Qp_Qs"])[mask]
        s = np.asarray(data["SaO2"])[mask] * 100.0
        step = max(1, q.size // 500)
        ax9.scatter(q[::step], s[::step], c=_color(sc),
                    s=10, alpha=0.5, label=sc)
    ax9.axhline(90, color="orange", ls=":", alpha=0.5)
    ax9.axvline(1.0, color="black", ls="--", alpha=0.5)
    ax9.set_xlabel("Qp / Qs")
    ax9.set_ylabel("SaO₂ (%)")
    ax9.set_title("SaO₂ vs Qp/Qs (Эйзенменгер)",
                  fontsize=11, fontweight="bold")
    ax9.legend(fontsize=7)
    ax9.grid(True, alpha=0.3)

    # 10. P_aO2
    ax10 = fig.add_subplot(gs[2, 2])
    _plot_series(ax10, results, "P_a_O2", "P_aO₂ (мм рт. ст.)",
                 "Парциальное давление O₂ в артерии")

    # 11. GFR
    ax11 = fig.add_subplot(gs[2, 3])
    _plot_series(ax11, results, "GFR", "СКФ (мл/с)", "Функция почек")

    # 12. Статистическая таблица
    ax12 = fig.add_subplot(gs[3, :])
    ax12.axis("off")

    summary_metrics = [
        ("Qp_Qs",              "Qp/Qs",              "{:.2f} ± {:.2f}", 1.0),
        ("SaO2",               "SaO₂, %",            "{:.1f} ± {:.2f}", 100.0),
        ("P_sa",               "АД сист., мм рт. ст.", "{:.0f} ± {:.0f}", 1.0),
        ("P_pa",               "АД лёг., мм рт. ст.",  "{:.0f} ± {:.0f}", 1.0),
        ("V_lv",               "Объём ЛЖ, мл",       "{:.0f} ± {:.0f}", 1.0),
        ("V_rv",               "Объём ПЖ, мл",       "{:.0f} ± {:.0f}", 1.0),
        ("Q_aortic",           "Сист. выброс, мл/с", "{:.1f} ± {:.1f}", 1.0),
        ("shunt_fraction_R2L", "R→L шунт, %",        "{:.1f} ± {:.2f}", 100.0),
        ("V_blood",            "Объём крови, мл",    "{:.0f} ± {:.0f}", 1.0),
        ("GFR",                "СКФ, мл/с",          "{:.2f} ± {:.2f}", 1.0),
        ("O2_consumption",  "CMRO₂, мл O₂/с",   "{:.3f} ± {:.3f}", 1.0),
        ("C_a_O2",          "C_aO₂, мл/мл",      "{:.3f} ± {:.3f}", 1.0),
        ("Q_brain",         "Q_br, мл/с",        "{:.2f} ± {:.2f}", 1.0),
    ]

    header = ["Показатель"] + [_short(s, 22) for s in scenarios]

    table_rows = [header]
    for key, label, fmt, scale in summary_metrics:
        row = [label]
        for sc in scenarios:
            r = steady_mean_std(results[sc], key, scale)
            row.append(fmt.format(*r) if r is not None else "N/A")
        table_rows.append(row)

    table = ax12.table(cellText=table_rows, cellLoc="center",
                       loc="center",
                       colWidths=[0.26] + [0.74 / len(scenarios)] * len(scenarios))
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    for i in range(len(table_rows)):
        try:
            table[(i, 0)].set_facecolor("#f0f0f0")
        except KeyError:
            pass

    ax12.set_title("Сводка установившихся значений (t > 0.75·t_end)",
                   fontsize=10, fontweight="bold")

    fig.suptitle("ДАШБОРД: Сравнение гемодинамики при ДМЖП\n"
                 "Здоровый человек vs пациенты с дефектом МЖП",
                 fontsize=14, fontweight="bold")

    plt.savefig("comprehensive_dashboard.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ comprehensive_dashboard.png")


# ===========================================================================
# Рис. 6 — Схематическая диаграмма
# ===========================================================================

def _draw_heart_schematic(ax, title, title_color,
                          lv_radius=2.0, rv_radius=2.0,
                          shunt_direction=None,
                          qs_label="Qs = Qp", qs_color="green") -> None:
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", color=title_color)

    # Камеры
    for (cx, cy, r, face) in [
        (4, 6, lv_radius, "#e74c3c"),   # ЛЖ
        (8, 6, rv_radius, "#3498db"),   # ПЖ
        (3.5, 9, 1.3, "#e74c3c"),       # ЛП
        (8.5, 9, 1.3, "#3498db"),       # ПП
    ]:
        ax.add_patch(Circle((cx, cy), r, facecolor=face,
                            edgecolor="black", alpha=0.7))

    # Магистральные сосуды
    ax.plot([4, 4], [6, 3], "r-", lw=3)
    ax.plot([8, 8], [6, 3], "b-", lw=3)
    ax.plot([4, 4], [9, 10.3], "r-", lw=3)
    ax.plot([8, 8], [9, 10.3], "b-", lw=3)

    # Стрелки направления кровотока
    for xy, xytext, color in [
        ((4, 3), (4, 4), "red"),
        ((8, 3), (8, 4), "blue"),
        ((4, 10.3), (4, 9), "red"),
        ((8, 10.3), (8, 9), "blue"),
    ]:
        ax.annotate("", xy=xy, xytext=xytext,
                    arrowprops=dict(arrowstyle="->", lw=2, color=color))

    # Шунт
    if shunt_direction == "L2R":
        ax.annotate("", xy=(6.1, 6), xytext=(5.8, 5.2),
                    arrowprops=dict(arrowstyle="->", lw=2, color="purple"))
    elif shunt_direction == "R2L":
        ax.annotate("", xy=(5.8, 5.2), xytext=(6.1, 6),
                    arrowprops=dict(arrowstyle="->", lw=2, color="purple"))

    # Подписи
    ax.text(4, 1.5, "Аорта", ha="center", fontsize=10, fontweight="bold")
    ax.text(8, 1.5, "Лёгочная артерия", ha="center",
            fontsize=10, fontweight="bold")
    ax.text(2, 6, "ЛЖ", ha="center", fontsize=10, fontweight="bold")
    ax.text(10, 6, "ПЖ", ha="center", fontsize=10, fontweight="bold")
    ax.text(2.5, 9, "ЛП", ha="center", fontsize=9)
    ax.text(9.5, 9, "ПП", ha="center", fontsize=9)
    ax.text(6, 11, qs_label, ha="center", fontsize=11,
            fontweight="bold", color=qs_color)


def plot_schematic_heart_comparison() -> None:
    """
    Схематическое сравнение гемодинамики для двух крайних фенотипов:
    «Здоровый» и «Эйзенменгер, декомпенсированный».

    Промежуточные сценарии (малый / большой ДМЖП, компенсированный
    Эйзенменгер) на этой схеме не показываются — она предназначена
    для наглядной демонстрации двух полюсов клинического спектра:
    нормы без шунта и запущенного право-левого шунта с цианозом.

    Цвета панелей подтягиваются из COLORS, если соответствующие метки
    присутствуют в текущем наборе пациентов; иначе используются
    разумные дефолты, чтобы функция работала и без физио-конфига.
    """
    # --- Цвета: пробуем взять из общей палитры, иначе — дефолты ---
    healthy_color = COLORS.get("Здоровый", "#2ecc71")
    eisenmenger_color = (
        COLORS.get("Эйзенменгер, декомпенсированный")
        or COLORS.get("Эйзенменгер декомпенс. (R=0.5)")
        or COLORS.get("Эйзенменгер (R=0.7)")
        or "#5b2c6f"   # тёмно-фиолетовый — согласован с общей палитрой
    )

    # --- Фигура: 2 панели вместо 3 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Схематическое сравнение гемодинамики:\n"
        "Здоровый vs Эйзенменгер (декомпенсированный)",
        fontsize=14, fontweight="bold",
    )

    # --- Панель 0: здоровый ---
    _draw_heart_schematic(
        axes[0],
        "Здоровое сердце (норма)",
        healthy_color,
        lv_radius=2.0, rv_radius=2.0,
        shunt_direction=None,
        qs_label="Qs = Qp  (шунт отсутствует)",
        qs_color=healthy_color,
    )

    # --- Панель 1: декомпенсированный Эйзенменгер ---
    _draw_heart_schematic(
        axes[1],
        "Эйзенменгер, декомпенсированный",
        eisenmenger_color,
        lv_radius=2.3, rv_radius=2.8,   # ПЖ дилатирован — визуальный акцент
        shunt_direction="R2L",
        qs_label="Qs > Qp  (R→L, цианоз)",
        qs_color=eisenmenger_color,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig("schematic_heart_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ schematic_heart_comparison.png")


# ===========================================================================
# Статистическая сводка в консоль
# ===========================================================================

def print_statistical_summary(results: Dict[str, dict]) -> None:
    scenarios = list(results.keys())

    print("\n" + "=" * 100)
    print("СТАТИСТИЧЕСКАЯ СВОДКА (установившийся режим, t > 0.75·t_end)")
    print("=" * 100)

    metrics = [
        # --- Гемодинамика: давления и потоки ---
        ("P_sa",               "Системное АД",              "мм рт. ст.", "{:.1f} ± {:.1f}", 1.0),
        ("P_pa",               "Лёгочное АД",               "мм рт. ст.", "{:.1f} ± {:.1f}", 1.0),
        ("Q_aortic",           "Системный выброс",          "мл/с",       "{:.1f} ± {:.1f}", 1.0),
        ("Q_pulmonary",        "Лёгочный кровоток",         "мл/с",       "{:.1f} ± {:.1f}", 1.0),
        ("Qp_Qs",              "Qp / Qs",                   "",           "{:.2f} ± {:.2f}", 1.0),
        ("Q_vsd",              "Шунт VSD",                  "мл/с",       "{:+.1f} ± {:.1f}", 1.0),
        # --- Оксигенация ---
        ("SaO2",               "SaO₂",                      "%",          "{:.1f} ± {:.2f}", 100.0),
        ("shunt_fraction_R2L", "R→L шунт",                  "%",          "{:.1f} ± {:.2f}", 100.0),
        # --- Объёмы ---
        ("V_lv",               "Объём ЛЖ",                  "мл",         "{:.1f} ± {:.1f}", 1.0),
        ("V_rv",               "Объём ПЖ",                  "мл",         "{:.1f} ± {:.1f}", 1.0),
        ("V_blood",            "Объём крови",               "мл",         "{:.0f} ± {:.0f}", 1.0),
        # --- Регионарные функции ---
        ("GFR",                "СКФ",                       "мл/с",       "{:.2f} ± {:.2f}", 1.0),
        ("Q_brain",            "Мозговой кровоток",         "мл/с",       "{:.2f} ± {:.2f}", 1.0),
        # --- Потребление O₂: мозг / периферия / интеграл ---
        ("O2_consumption",        "Потребление O₂ мозгом",      "мл O₂/с", "{:.3f} ± {:.3f}", 1.0),
        ("O2_consumption_periph", "Потребление O₂ периферией",  "мл O₂/с", "{:.3f} ± {:.3f}", 1.0),
        ("O2_uptake",             "Поглощение O₂ лёгкими",      "мл O₂/с", "{:.3f} ± {:.3f}", 1.0),
    ]

    header = f"{'Показатель':<32}"
    for s in scenarios:
        header += f"{_short(s, 20):>22}" 
    print(header)
    print("-" * len(header))

    for key, label, unit, fmt, scale in metrics:
        line = f"{(label + ' [' + unit + ']'):<32}"
        for sc in scenarios:
            r = steady_mean_std(results[sc], key, scale)
            if r is None:
                line += f"{'N/A':>22}"
            else:
                line += f"{fmt.format(*r):>22}"
        print(line)

    print("=" * 100)

    print("\n🔍 КЛЮЧЕВЫЕ НАХОДКИ:")
    for sc in scenarios:
        data = results[sc]
        r_sao2 = steady_mean_std(data, "SaO2", 100.0)
        r_qp   = steady_mean_std(data, "Qp_Qs")
        r_ppa  = steady_mean_std(data, "P_pa")
        r_r2l  = steady_mean_std(data, "shunt_fraction_R2L", 100.0)
        if r_sao2 is None or r_qp is None:
            continue
        sao2, qp_qs = r_sao2[0], r_qp[0]
        ppa = r_ppa[0] if r_ppa else float("nan")
        r2l = r_r2l[0] if r_r2l else 0.0

        tags = []
        if sao2 < 90:
            tags.append("гипоксемия")
        if qp_qs > 2.0:
            tags.append("большой шунт")
        if qp_qs < 1.0:
            tags.append("R→L")
        if ppa > 40:
            tags.append("тяжёлая ЛГ")
        note = f"  [{', '.join(tags)}]" if tags else ""

        print(f"  • {sc:<24}: Qp/Qs={qp_qs:.2f}, "
              f"SaO₂={sao2:.1f}%, P_pa={ppa:.1f} мм рт. ст., "
              f"R→L={r2l:.1f}%{note}")


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    print("=" * 70)
    print("ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ ГЕМОДИНАМИКИ: ЗДОРОВЫЙ vs ДМЖП")
    print("=" * 70)

    results = load_all_results()
    if not results:
        print("\n⚠ Не удалось загрузить данные.")
        print("   Сначала запустите run_simulation.py")
        return

    print("\n📊 Генерация графиков...")
    plot_hemodynamic_timeseries(results)
    plot_phase_portraits(results)
    plot_bar_comparison(results)
    plot_cardiovascular_parameters(results)
    plot_comprehensive_dashboard(results)
    plot_schematic_heart_comparison()

    print_statistical_summary(results)

    print("\n✅ Готово. Файлы:")
    for f in ("fig1_hemodynamics_timeseries.png",
              "fig2_phase_portraits.png",
              "fig3_bar_comparison.png",
              "fig4_detailed_cardiac.png",
              "comprehensive_dashboard.png",
              "schematic_heart_comparison.png"):
        print(f"   - {f}")


if __name__ == "__main__":
    main()