#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_liver_effect.py
Влияние ДМЖП / Эйзенменгера на печень по результатам vsd_results_*.npz.

Фигуры:
    fig_liver_timeseries.png   — временные ряды (Q_liver, лактат, альбумин, SaO2)
    fig_liver_bars.png         — столбчатое сравнение steady-state
    fig_liver_scatter.png      — корреляции (лактат↔SaO2, Q_liver↔P_sa, …)
    fig_liver_dashboard.png    — сводный дашборд 2×3 + таблица

Запуск:
    python plot_liver_effect.py
    python plot_liver_effect.py --pattern "vsd_results_*.npz"
"""

from __future__ import annotations

import argparse
import glob
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress

from utils import (
    subsample,
    steady_mean,
    steady_mean_std,
    qp_qs_steady,
    STEADY_FRAC_DEFAULT,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Стиль / палитра (без жёсткой зависимости от physio_config)
# ---------------------------------------------------------------------------

def _setup_style() -> None:
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
        "legend.fontsize": 8,
        "figure.autolayout": False,
    })


_setup_style()

_FALLBACK_COLORS = {
    "Здоровый": "#2ecc71",
    "Малый ДМЖП (R=5.0)": "#3498db",
    "Большой ДМЖП (R=1.0)": "#e67e22",
    "Эйзенменгер, компенсированный": "#9b59b6",
    "Эйзенменгер, декомпенсированный": "#5b2c6f",
}
_FALLBACK_ORDER = list(_FALLBACK_COLORS.keys())

COLORS: Dict[str, str] = dict(_FALLBACK_COLORS)
SCENARIO_ORDER: List[str] = list(_FALLBACK_ORDER)

try:
    from physio_config import load_all_patients, load_physiology
    _PHYSIOLOGY = load_physiology()
    _PATIENTS = load_all_patients(base_physiology=_PHYSIOLOGY)
    COLORS = {p["label"]: p["color"] for p in _PATIENTS.values()}
    SCENARIO_ORDER = [
        p["label"] for p in sorted(_PATIENTS.values(), key=lambda c: int(c["order"]))
    ]
except Exception:
    pass  # работаем с fallback


def _color(name: str) -> str:
    return COLORS.get(name, "gray")


def _short(s: str, n: int = 22) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def has_field(data: dict, *keys: str) -> bool:
    for k in keys:
        if k not in data:
            return False
        arr = data[k]
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------

def load_all_results(pattern: str = "vsd_results_*.npz"
                     ) -> Optional[Dict[str, dict]]:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"❌ Не найдено файлов по маске {pattern!r}.")
        print("   Сначала запустите run_simulation_parallel.py.")
        return None

    print(f"Найдено файлов: {len(files)}")
    results: Dict[str, dict] = {}

    for path in files:
        try:
            with np.load(path, allow_pickle=True) as npz:
                label = str(npz["label"]) if "label" in npz.files else path
                payload = {
                    k: np.array(npz[k])
                    for k in npz.files
                    if k not in ("label", "id", "description")
                }
            results[label] = payload
            print(f"✓ {path:40s} → {label}")
        except Exception as exc:
            print(f"✗ Ошибка загрузки {path}: {exc}")

    if not results:
        print("❌ Не удалось загрузить ни одного файла.")
        return None

    ordered: Dict[str, dict] = {}
    for key in SCENARIO_ORDER:
        if key in results:
            ordered[key] = results.pop(key)
    ordered.update(results)

    print(f"\nЗагружено сценариев: {list(ordered.keys())}")
    return ordered


# ---------------------------------------------------------------------------
# Печёночные метрики
# ---------------------------------------------------------------------------

LIVER_SERIES = [
    ("Q_liver_out",      "Q_liver (мл/с)",        1.0,   []),
    ("C_lactate_blood",  "Лактат крови (мл/мл)",  1.0,   []),
    ("C_albumin_blood",  "Альбумин крови",        1.0,   []),
    ("SaO2",             "SaO₂ (%)",              100.0, [(90, "orange", ":", "SaO₂=90%")]),
]

LIVER_BAR_METRICS = [
    ("Q_liver_out",     "Q_liver (мл/с)",     1.0),
    ("C_lactate_blood", "Лактат крови",       1.0),
    ("C_albumin_blood", "Альбумин",           1.0),
    ("dC_lactate_liver","dC_lactate_liver",   1.0),
    ("P_sa",            "P_sa (мм рт. ст.)",   1.0),
    ("SaO2",            "SaO₂ (%)",           100.0),
]


# ---------------------------------------------------------------------------
# Рис. 1 — временные ряды
# ---------------------------------------------------------------------------

def plot_liver_timeseries(results: Dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Влияние ДМЖП / Эйзенменгера на печень: временные ряды",
                 fontsize=14, fontweight="bold")

    for ax, (key, ylabel, scale, hlines) in zip(axes.flat, LIVER_SERIES):
        for sc, data in results.items():
            if not has_field(data, "t", key):
                continue
            t, y = subsample(data, key)
            y = y * scale
            m = np.isfinite(y)
            if not np.any(m):
                continue
            ax.plot(t[m], y[m], color=_color(sc), lw=1.5, label=sc)
        ax.set_xlabel("Время (с)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        for y0, col, ls, lbl in hlines:
            ax.axhline(y0, color=col, ls=ls, alpha=0.5, label=lbl)
        ax.set_title(key, fontsize=11)

    handles, labels, seen = [], [], set()
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li in seen:
                continue
            seen.add(li)
            handles.append(hi)
            labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="upper right",
                   bbox_to_anchor=(0.99, 0.97), fontsize=8, ncol=2)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig("fig_liver_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_liver_timeseries.png")


# ---------------------------------------------------------------------------
# Рис. 2 — столбцы steady-state
# ---------------------------------------------------------------------------

def plot_liver_bars(results: Dict[str, dict]) -> None:
    scenarios = list(results.keys())
    n = len(LIVER_BAR_METRICS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))
    fig.suptitle(
        f"Печень: установившиеся значения (t > {STEADY_FRAC_DEFAULT:.2f}·t_end)",
        fontsize=14, fontweight="bold",
    )
    axes = np.atleast_1d(axes).flatten()

    for ax, (key, ylabel, scale) in zip(axes, LIVER_BAR_METRICS):
        means, stds = [], []
        for sc in scenarios:
            r = steady_mean_std(results[sc], key, scale)
            if r is None:
                means.append(0.0)
                stds.append(0.0)
            else:
                means.append(r[0])
                stds.append(r[1])

        bars = ax.bar(
            scenarios, means,
            color=[_color(s) for s in scenarios],
            alpha=0.75, edgecolor="black", linewidth=1,
        )
        ax.errorbar(scenarios, means, yerr=stds, fmt="none",
                    ecolor="black", capsize=4, capthick=1)
        ax.set_ylabel(ylabel)
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")
        ax.grid(True, alpha=0.3, axis="y")

        ymax = max([abs(m) for m in means] + [1e-9])
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.02,
                f"{m:.3g}", ha="center", va="bottom", fontsize=8,
            )

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig("fig_liver_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_liver_bars.png")


# ---------------------------------------------------------------------------
# Рис. 3 — scatter-корреляции
# ---------------------------------------------------------------------------

def _steady_pair(data: dict, kx: str, ky: str,
                 sx: float = 1.0, sy: float = 1.0
                 ) -> Optional[Tuple[float, float]]:
    mx = steady_mean(data, kx, sx)
    my = steady_mean(data, ky, sy)
    if mx is None or my is None:
        return None
    return mx, my


def plot_liver_scatter(results: Dict[str, dict]) -> None:
    pairs = [
        ("SaO2", "C_lactate_blood", "SaO₂ (%)", "Лактат крови",
         100.0, 1.0, "Лактат ↔ SaO₂"),
        ("P_sa", "Q_liver_out", "P_sa (мм рт. ст.)", "Q_liver (мл/с)",
         1.0, 1.0, "Q_liver ↔ P_sa"),
        ("Q_aortic", "Q_liver_out", "Q_aortic (мл/с)", "Q_liver (мл/с)",
         1.0, 1.0, "Q_liver ↔ CO"),
        ("SaO2", "C_albumin_blood", "SaO₂ (%)", "Альбумин",
         100.0, 1.0, "Альбумин ↔ SaO₂"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Корреляции: печень ↔ гемодинамика / оксигенация",
                 fontsize=14, fontweight="bold")

    for ax, (kx, ky, xl, yl, sx, sy, title) in zip(axes.flat, pairs):
        xs, ys = [], []
        for sc, data in results.items():
            pair = _steady_pair(data, kx, ky, sx, sy)
            if pair is None:
                continue
            xs.append(pair[0])
            ys.append(pair[1])
            ax.scatter(pair[0], pair[1], s=140, c=_color(sc),
                       edgecolor="black", linewidth=1.3, zorder=5, label=sc)
            ax.annotate(_short(sc, 14), (pair[0], pair[1]),
                        xytext=(6, 4), textcoords="offset points", fontsize=7)

        if len(xs) >= 2:
            slope, intercept, r_val, _, _ = linregress(xs, ys)
            x_line = np.linspace(min(xs), max(xs), 50)
            ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.55,
                    label=f"R² = {r_val ** 2:.3f}")

        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="best")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig("fig_liver_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_liver_scatter.png")


# ---------------------------------------------------------------------------
# Рис. 4 — дашборд
# ---------------------------------------------------------------------------

def plot_liver_dashboard(results: Dict[str, dict]) -> None:
    scenarios = list(results.keys())
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.32)

    # 1. Q_liver timeseries
    ax = fig.add_subplot(gs[0, 0])
    for sc, data in results.items():
        if not has_field(data, "t", "Q_liver_out"):
            continue
        t, y = subsample(data, "Q_liver_out")
        m = np.isfinite(y)
        if np.any(m):
            ax.plot(t[m], y[m], color=_color(sc), lw=1.4, label=sc)
    ax.set_ylabel("Q_liver (мл/с)")
    ax.set_xlabel("Время (с)")
    ax.set_title("Печёночный кровоток", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 2. Lactate timeseries
    ax = fig.add_subplot(gs[0, 1])
    for sc, data in results.items():
        if not has_field(data, "t", "C_lactate_blood"):
            continue
        t, y = subsample(data, "C_lactate_blood")
        m = np.isfinite(y)
        if np.any(m):
            ax.plot(t[m], y[m], color=_color(sc), lw=1.4, label=sc)
    ax.set_ylabel("Лактат (мл/мл)")
    ax.set_xlabel("Время (с)")
    ax.set_title("Лактат крови", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 3. Albumin timeseries
    ax = fig.add_subplot(gs[0, 2])
    for sc, data in results.items():
        if not has_field(data, "t", "C_albumin_blood"):
            continue
        t, y = subsample(data, "C_albumin_blood")
        m = np.isfinite(y)
        if np.any(m):
            ax.plot(t[m], y[m], color=_color(sc), lw=1.4, label=sc)
    ax.set_ylabel("Альбумин")
    ax.set_xlabel("Время (с)")
    ax.set_title("Альбумин крови", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 4. Bar Q_liver
    ax = fig.add_subplot(gs[1, 0])
    means = [steady_mean(results[sc], "Q_liver_out") or 0.0 for sc in scenarios]
    ax.bar(scenarios, means, color=[_color(s) for s in scenarios],
           alpha=0.75, edgecolor="black")
    ax.set_ylabel("Q_liver (мл/с)")
    ax.set_title("Q_liver (steady)", fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.grid(True, alpha=0.3, axis="y")

    # 5. Scatter lactate vs SaO2
    ax = fig.add_subplot(gs[1, 1])
    xs, ys = [], []
    for sc, data in results.items():
        pair = _steady_pair(data, "SaO2", "C_lactate_blood", 100.0, 1.0)
        if pair is None:
            continue
        xs.append(pair[0]); ys.append(pair[1])
        ax.scatter(pair[0], pair[1], s=130, c=_color(sc),
                   edgecolor="black", linewidth=1.2, label=sc, zorder=5)
        ax.annotate(_short(sc, 12), (pair[0], pair[1]),
                    xytext=(5, 3), textcoords="offset points", fontsize=7)
    if len(xs) >= 2:
        slope, intercept, r_val, _, _ = linregress(xs, ys)
        x_line = np.linspace(min(xs), max(xs), 40)
        ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.5,
                label=f"R²={r_val**2:.2f}")
    ax.axvline(90, color="orange", ls=":", alpha=0.5)
    ax.set_xlabel("SaO₂ (%)")
    ax.set_ylabel("Лактат крови")
    ax.set_title("Лактат ↔ SaO₂", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)

    # 6. Scatter Q_liver vs P_sa
    ax = fig.add_subplot(gs[1, 2])
    for sc, data in results.items():
        pair = _steady_pair(data, "P_sa", "Q_liver_out")
        if pair is None:
            continue
        ax.scatter(pair[0], pair[1], s=130, c=_color(sc),
                   edgecolor="black", linewidth=1.2, label=sc, zorder=5)
        ax.annotate(_short(sc, 12), (pair[0], pair[1]),
                    xytext=(5, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("P_sa (мм рт. ст.)")
    ax.set_ylabel("Q_liver (мл/с)")
    ax.set_title("Q_liver ↔ P_sa", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)

    # 7. Таблица
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")

    header = ["Показатель"] + [_short(s, 18) for s in scenarios]
    rows = [header]

    table_keys = [
        ("Q_liver_out",     "Q_liver, мл/с",     "{:.1f}", 1.0),
        ("C_lactate_blood", "Лактат",            "{:.3f}", 1.0),
        ("C_albumin_blood", "Альбумин",          "{:.2f}", 1.0),
        ("dC_lactate_liver","dC_lact (liver)",   "{:.2e}", 1.0),
        ("P_sa",            "P_sa, мм рт.ст.",   "{:.1f}", 1.0),
        ("SaO2",            "SaO₂, %",           "{:.1f}", 100.0),
        ("Q_aortic",        "CO, мл/с",          "{:.1f}", 1.0),
        ("GFR",             "GFR, мл/с",         "{:.2f}", 1.0),
    ]
    for key, label, fmt, scale in table_keys:
        row = [label]
        for sc in scenarios:
            v = steady_mean(results[sc], key, scale)
            row.append(fmt.format(v) if v is not None else "N/A")
        rows.append(row)

    row = ["Qp/Qs"]
    for sc in scenarios:
        v = qp_qs_steady(results[sc])
        row.append(f"{v:.2f}" if v is not None else "N/A")
    rows.append(row)

    healthy_key = next((s for s in scenarios if "Здоров" in s), None)
    if healthy_key is not None:
        q_h = steady_mean(results[healthy_key], "Q_liver_out")
        l_h = steady_mean(results[healthy_key], "C_lactate_blood")
        row_dq = ["ΔQ_liver vs healthy, %"]
        row_dl = ["Δлактат vs healthy, %"]
        for sc in scenarios:
            q = steady_mean(results[sc], "Q_liver_out")
            l = steady_mean(results[sc], "C_lactate_blood")
            if q is not None and q_h and q_h != 0:
                row_dq.append(f"{(q / q_h - 1) * 100:+.1f}")
            else:
                row_dq.append("N/A")
            if l is not None and l_h and l_h != 0:
                row_dl.append(f"{(l / l_h - 1) * 100:+.1f}")
            else:
                row_dl.append("N/A")
        rows.append(row_dq)
        rows.append(row_dl)

    table = ax.table(
        cellText=rows, cellLoc="center", loc="center",
        colWidths=[0.22] + [0.78 / len(scenarios)] * len(scenarios),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.55)
    for i in range(len(rows)):
        try:
            table[(i, 0)].set_facecolor("#f0f0f0")
        except KeyError:
            pass

    ax.set_title(
        f"Сводка: печень при ДМЖП (t > {STEADY_FRAC_DEFAULT:.2f}·t_end)",
        fontsize=11, fontweight="bold", pad=12,
    )

    fig.suptitle(
        "ДАШБОРД: влияние ДМЖП / Эйзенменгера на печень\n"
        "(вторичные эффекты: перфузия, лактат, альбумин)",
        fontsize=14, fontweight="bold",
    )
    plt.savefig("fig_liver_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_liver_dashboard.png")


# ---------------------------------------------------------------------------
# Текстовая сводка
# ---------------------------------------------------------------------------

def print_liver_summary(results: Dict[str, dict]) -> None:
    scenarios = list(results.keys())
    print("\n" + "=" * 100)
    print("ПЕЧЕНЬ: СТАТИСТИЧЕСКАЯ СВОДКА (установившийся режим)")
    print("=" * 100)

    metrics = [
        ("Q_liver_out",      "Q_liver",        "мл/с",  "{:.2f} ± {:.2f}", 1.0),
        ("C_lactate_blood",  "Лактат крови",    "",     "{:.4f} ± {:.4f}", 1.0),
        ("C_albumin_blood",  "Альбумин",        "",     "{:.3f} ± {:.3f}", 1.0),
        ("C_bilirubin_blood","Билирубин",       "",     "{:.2e} ± {:.2e}", 1.0),
        ("C_ammonia_blood",  "Аммиак",          "",     "{:.2e} ± {:.2e}", 1.0),
        ("dC_lactate_liver", "dC_lactate_liver","",     "{:.2e} ± {:.2e}", 1.0),
        ("liver_functional", "liver_functional","",     "{:.2f} ± {:.2f}", 1.0),
        ("P_sa",             "P_sa",            "мм Hg","{:.1f} ± {:.1f}", 1.0),
        ("SaO2",             "SaO₂",            "%",    "{:.1f} ± {:.2f}", 100.0),
    ]

    header = f"{'Показатель':<28}"
    for s in scenarios:
        header += f"{_short(s, 18):>20}"
    print(header)
    print("-" * len(header))

    for key, label, unit, fmt, scale in metrics:
        suffix = f" [{unit}]" if unit else ""
        line = f"{(label + suffix):<28}"
        for sc in scenarios:
            r = steady_mean_std(results[sc], key, scale)
            if r is None:
                line += f"{'N/A':>20}"
            else:
                line += f"{fmt.format(*r):>20}"
        print(line)

    print("=" * 100)
    print("\n🔍 ИНТЕРПРЕТАЦИЯ:")
    healthy = next((s for s in scenarios if "Здоров" in s), None)
    decomp = next((s for s in scenarios if "декомп" in s.lower()), None)
    if healthy and decomp:
        q_h = steady_mean(results[healthy], "Q_liver_out")
        q_d = steady_mean(results[decomp], "Q_liver_out")
        l_h = steady_mean(results[healthy], "C_lactate_blood")
        l_d = steady_mean(results[decomp], "C_lactate_blood")
        a_h = steady_mean(results[healthy], "C_albumin_blood")
        a_d = steady_mean(results[decomp], "C_albumin_blood")
        if q_h and q_d:
            print(f"  • Q_liver: {q_h:.1f} → {q_d:.1f} мл/с  "
                  f"({(q_d/q_h - 1)*100:+.0f} %)  — гипоперфузия печени")
        if l_h and l_d:
            print(f"  • Лактат:  {l_h:.3f} → {l_d:.3f}  "
                  f"({(l_d/l_h - 1)*100:+.0f} %)  — накопление при гипоксемии")
        if a_h and a_d:
            print(f"  • Альбумин:{a_h:.2f} → {a_d:.2f}  "
                  f"({(a_d/a_h - 1)*100:+.1f} %)  — слабое снижение")
        print("  • Билирубин / аммиак ≈ 0 — модель не накапливает их при гипоксии")
        print("  • liver_functional = 1 всегда — нет прогрессирующей недостаточности")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(pattern: str = "vsd_results_*.npz") -> None:
    print("=" * 70)
    print("ВИЗУАЛИЗАЦИЯ: ВЛИЯНИЕ ДМЖП / ЭЙЗЕНМЕНГЕРА НА ПЕЧЕНЬ")
    print("=" * 70)

    results = load_all_results(pattern)
    if not results:
        return

    sample = next(iter(results.values()))
    liver_keys = [k for k in ("Q_liver_out", "C_lactate_blood", "C_albumin_blood")
                  if k in sample]
    if not liver_keys:
        print("❌ В npz нет печёночных ключей (Q_liver_out / C_lactate_blood / …).")
        return
    print(f"\nДоступные печёночные ключи: {liver_keys}")

    print("\n📊 Генерация графиков...")
    plot_liver_timeseries(results)
    plot_liver_bars(results)
    plot_liver_scatter(results)
    plot_liver_dashboard(results)
    print_liver_summary(results)

    print("\n✅ Готово. Файлы:")
    for f in ("fig_liver_timeseries.png",
              "fig_liver_bars.png",
              "fig_liver_scatter.png",
              "fig_liver_dashboard.png"):
        print(f"   - {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Влияние ДМЖП / Эйзенменгера на печень (из vsd_results_*.npz)."
    )
    parser.add_argument(
        "--pattern", type=str, default="vsd_results_*.npz",
        help="Маска npz-файлов (по умолчанию vsd_results_*.npz)",
    )
    args = parser.parse_args()
    main(pattern=args.pattern)