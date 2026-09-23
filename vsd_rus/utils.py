# utils.py
"""
Общие утилиты для run_simulation.py и visualize_vsd_comparison.py.

Содержит функции обработки временных рядов, которые раньше дублировались
(с расхождениями) в обоих скриптах:

    safe_savgol_filter — сглаживание Савицкого-Голея с защитой от NaN
    subsample          — прореживание ряда до ~n_target точек
    steady_mask        — маска последних (1 - frac) симуляции
    clean_nans         — замена NaN/Inf линейной интерполяцией

Каноничные дефолты:
    SAFE_SAVGOL_WINDOW  = 101
    SAFE_SAVGOL_POLY    = 3
    SUBSAMPLE_N_TARGET  = 5000
    STEADY_FRAC_DEFAULT = 0.75
"""

from __future__ import annotations

import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Дефолты (единый источник истины)
# ---------------------------------------------------------------------------

SAFE_SAVGOL_WINDOW   = 101
SAFE_SAVGOL_POLY     = 3
SUBSAMPLE_N_TARGET   = 5000
STEADY_FRAC_DEFAULT  = 0.75

# ---------------------------------------------------------------------------
# Сглаживание
# ---------------------------------------------------------------------------

def safe_savgol_filter(data, window_length=SAFE_SAVGOL_WINDOW,
                       polyorder=SAFE_SAVGOL_POLY):
    """
    Безопасный Savitzky-Golay: чистит NaN, подрезает окно под длину,
    молча возвращает исходный массив при любой ошибке.

    Единая версия для обоих скриптов. Раньше в run_simulation.py
    дефолтов не было, а в visualize_vsd_comparison.py функция
    называлась safe_savgol — теперь это одно и то же.
    """
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return np.asarray(data, dtype=float)

    arr = np.asarray(data, dtype=float).copy()
    if arr.ndim != 1 or arr.size == 0:
        return arr

    # Линейная интерполяция пропусков
    bad = ~np.isfinite(arr)
    if np.any(bad):
        good = ~bad
        if good.sum() < 2:
            return np.zeros_like(arr)
        idx = np.arange(arr.size)
        arr[bad] = np.interp(idx[bad], idx[good], arr[good])

    # Подгонка окна и порядка
    if window_length > arr.size:
        window_length = arr.size if arr.size % 2 == 1 else arr.size - 1
    if window_length < 3:
        return arr
    if polyorder >= window_length:
        polyorder = window_length - 1
    if polyorder < 1:
        return arr

    try:
        return savgol_filter(arr, window_length, polyorder)
    except Exception:
        return arr


# ---------------------------------------------------------------------------
# Прореживание
# ---------------------------------------------------------------------------

def subsample(data, key, n_target=SUBSAMPLE_N_TARGET):
    """
    Прореживает временной ряд data[key] до ~n_target точек.

    Если исходный ряд короче n_target, возвращает его как есть,
    но пишет предупреждение: разрешение симуляции слишком низкое,
    прореживание выродилось в no-op (симптом — увеличить N_EVAL).
    """
    t = np.asarray(data.get('t', []))
    if key not in data or t.size == 0:
        return np.array([]), np.array([])

    arr = np.asarray(data[key])
    n = min(len(t), len(arr))
    if n == 0:
        return np.array([]), np.array([])

    if n <= n_target:
        warnings.warn(
            f"subsample: len(t)={n} ≤ n_target={n_target} — "
            f"прореживание вырождено, увеличьте N_EVAL или уменьшите n_target"
        )
        return t[:n], arr[:n]

    step = max(1, n // n_target)
    return t[:n:step], arr[:n:step]


# ---------------------------------------------------------------------------
# Маска установившегося режима
# ---------------------------------------------------------------------------

def steady_mask(data, frac=STEADY_FRAC_DEFAULT):
    """Маска последних (1 - frac) симуляции."""
    t = np.asarray(data.get('t', []))
    if t.size == 0:
        return np.zeros(0, dtype=bool)
    return t >= frac * t[-1]


# ---------------------------------------------------------------------------
# Очистка NaN/Inf
# ---------------------------------------------------------------------------

def clean_nans(data):
    """Заменяет NaN/Inf в числовых полях линейной интерполяцией."""
    for key in list(data.keys()):
        if not isinstance(data[key], np.ndarray):
            continue
        arr = data[key]
        if arr.ndim != 1 or arr.dtype.kind not in 'fc':
            continue
        bad = ~np.isfinite(arr)
        if not np.any(bad):
            continue
        idx = np.arange(len(arr))
        if np.any(~bad):
            arr[bad] = np.interp(idx[bad], idx[~bad], arr[~bad])
        else:
            data[key] = np.zeros_like(arr)
    return data


# ---------------------------------------------------------------------------
# Авто-масштабирование оси Y
# ---------------------------------------------------------------------------

def auto_ylim(ax, *arrays, pad_frac: float = 0.10, pct=(1, 99)):
    """
    Устойчивое к выбросам авто-масштабирование оси Y.

    Диапазон берётся по перцентилям pct (по умолчанию 1–99), а не по
    min/max: это защищает панели от диастолических пиков Qp_Qs ~1e6,
    которые иначе растягивают ось и схлопывают полезный диапазон.

    Параметры
    ---------
    ax       : matplotlib.axes.Axes — ось, у которой меняется ylim
    *arrays  : один или несколько массивов значений (NaN/Inf игнорируются)
    pad_frac : доля диапазона, добавляемая сверху и снизу
    pct      : кортеж (lo, hi) перцентилей
    """
    try:
        all_y = np.concatenate(
            [np.asarray(a)[np.isfinite(a)] for a in arrays if np.size(a) > 0]
        )
        if all_y.size == 0:
            return
        lo, hi = np.percentile(all_y, pct)
        pad = pad_frac * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Установившиеся средние и их форматирование
# ---------------------------------------------------------------------------

def steady_mean(data, key, scale=1.0):
    """Среднее по установившемуся окну. Возвращает float или None."""
    if key not in data:
        return None
    m = steady_mask(data) & np.isfinite(np.asarray(data[key]))
    if not np.any(m):
        return None
    return float(np.mean(data[key][m]) * scale)


def steady_mean_std(data, key, scale=1.0):
    """Кортеж (mean, std) по установившемуся окну или None."""
    if key not in data:
        return None
    m = steady_mask(data) & np.isfinite(np.asarray(data[key]))
    vals = np.asarray(data[key])[m]
    if vals.size == 0:
        return None
    return float(np.mean(vals) * scale), float(np.std(vals) * scale)


def format_mean_std(data, key, scale=1.0, fmt="{:.1f}"):
    """Строка 'mean ± std' или 'N/A'. Для печати отчётов."""
    r = steady_mean_std(data, key, scale)
    if r is None:
        return "N/A"
    return fmt.format(r[0]) + f" ± {fmt.format(r[1])}"


def qp_qs_steady(data):
    """
    Qp/Qs = mean(Q_pulmonary) / mean(Q_aortic).

    Корректный способ усреднения для стационарного режима.
    Прямое усреднение мгновенного отношения Qp(t)/Qs(t) даёт
    артефакты порядка 1e6 из-за деления на почти нулевые
    диастолические потоки.
    """
    qp = steady_mean(data, "Q_pulmonary")
    qa = steady_mean(data, "Q_aortic")
    if qp is None or qa is None or qa <= 0:
        return None
    return qp / qa

def occlusion_profile(t, t_onset, severity=1.0, rise_time=5.0):
    """Плавный ramp окклюзии: 1.0 до t_onset, затем плавно к (1-severity)."""
    if t < t_onset:
        return 1.0
    frac = min((t - t_onset) / max(rise_time, 1e-6), 1.0)
    return 1.0 - severity * frac