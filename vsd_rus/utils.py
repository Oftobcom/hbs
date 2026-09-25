# utils.py
"""
Общие утилиты для run_simulation.py и visualize_vsd_comparison.py.

Содержит функции обработки временных рядов, которые раньше дублировались
(с расхождениями) в обоих скриптах:

    safe_savgol_filter — сглаживание Савицкого-Голея с защитой от NaN
    subsample          — прореживание ряда до ~n_target точек
    steady_mask        — маска последних (1 - frac) симуляции
    clean_nans         — замена NaN/Inf линейной интерполяцией
    auto_ylim          — устойчивое к выбросам авто-масштабирование оси Y
    steady_mean / steady_mean_std / format_mean_std
                       — установившиеся средние и их форматирование
    qp_qs_steady       — Qp/Qs из средних потоков
    occlusion_profile  — плавный ramp окклюзии

Каноничные дефолты:
    SAFE_SAVGOL_WINDOW  = 101
    SAFE_SAVGOL_POLY    = 3
    SUBSAMPLE_N_TARGET  = 5000
    STEADY_FRAC_DEFAULT = 0.75
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Дефолты (единый источник истины)
# ---------------------------------------------------------------------------

SAFE_SAVGOL_WINDOW   = 101
SAFE_SAVGOL_POLY     = 3
SUBSAMPLE_N_TARGET   = 5000
STEADY_FRAC_DEFAULT  = 0.75


# ===========================================================================
# Модуль-уровневая валидация констант — fail-fast при импорте.
# ===========================================================================
def _validate_module_constants() -> None:
    """
    Проверка констант модуля. RuntimeError → fail-fast при импорте,
    чтобы не тихо работать с неконсистентными дефолтами.
    """
    if SAFE_SAVGOL_WINDOW < 3 or SAFE_SAVGOL_WINDOW % 2 == 0:
        raise RuntimeError(
            f"utils: SAFE_SAVGOL_WINDOW={SAFE_SAVGOL_WINDOW} должно быть "
            f"нечётным ≥ 3 (savgol требует нечётное окно)."
        )
    if not (1 <= SAFE_SAVGOL_POLY < SAFE_SAVGOL_WINDOW):
        raise RuntimeError(
            f"utils: SAFE_SAVGOL_POLY={SAFE_SAVGOL_POLY} должно быть "
            f"в [1, SAFE_SAVGOL_WINDOW)."
        )
    if SUBSAMPLE_N_TARGET <= 0:
        raise RuntimeError(
            f"utils: SUBSAMPLE_N_TARGET={SUBSAMPLE_N_TARGET} должно быть > 0."
        )
    if not (0.0 < STEADY_FRAC_DEFAULT < 1.0):
        raise RuntimeError(
            f"utils: STEADY_FRAC_DEFAULT={STEADY_FRAC_DEFAULT} должно быть "
            f"в (0, 1)."
        )


_validate_module_constants()


# ===========================================================================
# Валидаторы аргументов — fail-fast для явно плохих входов.
# ===========================================================================
def _check_positive_int(name: str, v, min_val: int = 1) -> int:
    v_int = int(v)
    if v_int < min_val:
        raise ValueError(f"utils: {name}={v} должно быть ≥ {min_val}.")
    return v_int


def _check_fraction(name: str, v, allow_one: bool = False) -> float:
    """Проверка, что v ∈ (0, 1) или [0, 1) в зависимости от allow_one."""
    v = float(v)
    if not np.isfinite(v):
        raise ValueError(f"utils: {name}={v} не конечно.")
    if allow_one:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"utils: {name}={v} вне [0, 1].")
    else:
        if not (0.0 < v < 1.0):
            raise ValueError(f"utils: {name}={v} вне (0, 1).")
    return v


def _check_nonneg(name: str, v) -> float:
    v = float(v)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"utils: {name}={v} должно быть ≥ 0.")
    return v


# ===========================================================================
# Сглаживание
# ===========================================================================

def safe_savgol_filter(data,
                       window_length=SAFE_SAVGOL_WINDOW,
                       polyorder=SAFE_SAVGOL_POLY):
    """
    Безопасный Savitzky-Golay: чистит NaN, подрезает окно под длину,
    молча возвращает исходный массив при любой ошибке.

    Валидация аргументов — fail-fast: window_length и polyorder
    должны быть физически допустимыми; если передать 0 или −1,
    это ошибка вызывающего кода, а не «численный сбой».

    Поведение на коротких рядах, NaN или ошибках scipy — silent fallback
    (вернуть исходный или интерполированный массив). Это by design.
    """
    # --- Fail-fast на аргументах ---
    window_length = _check_positive_int("window_length", window_length, min_val=1)
    polyorder     = _check_positive_int("polyorder", polyorder, min_val=0)
    if polyorder >= window_length and window_length >= 2:
        # savgol требует polyorder < window_length; но окно может подрезаться
        # ниже по коду, поэтому здесь только предупреждение жёсткого случая.
        pass  # обработаем ниже после подрезки

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


# ===========================================================================
# Прореживание
# ===========================================================================

def subsample(data, key, n_target=SUBSAMPLE_N_TARGET):
    """
    Прореживает временной ряд data[key] до ~n_target точек.

    Если исходный ряд короче n_target, возвращает его как есть,
    но пишет предупреждение: разрешение симуляции слишком низкое,
    прореживание выродилось в no-op (симптом — увеличить N_EVAL).
    """
    # --- Fail-fast на аргументах ---
    if not isinstance(data, dict):
        raise ValueError(
            f"utils.subsample: data должен быть dict, "
            f"получено {type(data).__name__}."
        )
    if not isinstance(key, str) or not key:
        raise ValueError(
            f"utils.subsample: key должен быть непустой строкой, "
            f"получено {key!r}."
        )
    n_target = _check_positive_int("n_target", n_target, min_val=1)

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


# ===========================================================================
# Маска установившегося режима
# ===========================================================================

def steady_mask(data, frac=STEADY_FRAC_DEFAULT):
    """Маска последних (1 − frac) симуляции."""
    # --- Fail-fast на аргументах ---
    if not isinstance(data, dict):
        raise ValueError(
            f"utils.steady_mask: data должен быть dict, "
            f"получено {type(data).__name__}."
        )
    frac = _check_fraction("frac", frac, allow_one=True)

    t = np.asarray(data.get('t', []))
    if t.size == 0:
        return np.zeros(0, dtype=bool)
    return t >= frac * t[-1]


# ===========================================================================
# Очистка NaN/Inf
# ===========================================================================

def clean_nans(data):
    """Заменяет NaN/Inf в числовых полях линейной интерполяцией."""
    if not isinstance(data, dict):
        raise ValueError(
            f"utils.clean_nans: data должен быть dict, "
            f"получено {type(data).__name__}."
        )
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


# ===========================================================================
# Авто-масштабирование оси Y
# ===========================================================================

def auto_ylim(ax, *arrays, pad_frac: float = 0.10, pct: Tuple[float, float] = (1, 99)):
    """
    Устойчивое к выбросам авто-масштабирование оси Y.

    Диапазон берётся по перцентилям pct (по умолчанию 1–99), а не по
    min/max: это защищает панели от диастолических пиков Qp_Qs ~1e6,
    которые иначе растягивают ось и схлопывают полезный диапазон.

    Fail-fast: pad_frac ≥ 0, pct — пара чисел 0 ≤ lo < hi ≤ 100.
    Silent fallback: если ax не имеет set_ylim или все arrays пусты,
    функция молча ничего не делает.
    """
    # --- Fail-fast на аргументах ---
    pad_frac = _check_nonneg("pad_frac", pad_frac)
    if not (isinstance(pct, (tuple, list)) and len(pct) == 2):
        raise ValueError(
            f"utils.auto_ylim: pct должен быть tuple/list из 2 чисел, "
            f"получено {pct!r}."
        )
    lo_pct, hi_pct = float(pct[0]), float(pct[1])
    if not (0.0 <= lo_pct < hi_pct <= 100.0):
        raise ValueError(
            f"utils.auto_ylim: pct=({lo_pct}, {hi_pct}) — требуется "
            f"0 ≤ pct[0] < pct[1] ≤ 100."
        )

    try:
        if not hasattr(ax, "set_ylim"):
            return
        all_y = np.concatenate(
            [np.asarray(a)[np.isfinite(a)] for a in arrays if np.size(a) > 0]
        )
        if all_y.size == 0:
            return
        lo, hi = np.percentile(all_y, (lo_pct, hi_pct))
        pad = pad_frac * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
    except Exception:
        pass


# ===========================================================================
# Установившиеся средние и их форматирование
# ===========================================================================

def steady_mean(data, key, scale=1.0):
    """Среднее по установившемуся окну. Возвращает float или None."""
    if not isinstance(data, dict):
        raise ValueError(
            f"utils.steady_mean: data должен быть dict, "
            f"получено {type(data).__name__}."
        )
    if key not in data:
        return None
    m = steady_mask(data) & np.isfinite(np.asarray(data[key]))
    if not np.any(m):
        return None
    return float(np.mean(data[key][m]) * scale)


def steady_mean_std(data, key, scale=1.0):
    """Кортеж (mean, std) по установившемуся окну или None."""
    if not isinstance(data, dict):
        raise ValueError(
            f"utils.steady_mean_std: data должен быть dict, "
            f"получено {type(data).__name__}."
        )
    if key not in data:
        return None
    m = steady_mask(data) & np.isfinite(np.asarray(data[key]))
    vals = np.asarray(data[key])[m]
    if vals.size == 0:
        return None
    return float(np.mean(vals) * scale), float(np.std(vals) * scale)


def format_mean_std(data, key, scale=1.0, fmt="{:.1f}"):
    """Строка 'mean ± std' или 'N/A'. Для печати отчётов."""
    if not isinstance(fmt, str):
        raise ValueError(
            f"utils.format_mean_std: fmt должен быть str, "
            f"получено {type(fmt).__name__}."
        )
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
    if not isinstance(data, dict):
        raise ValueError(
            f"utils.qp_qs_steady: data должен быть dict, "
            f"получено {type(data).__name__}."
        )
    qp = steady_mean(data, "Q_pulmonary")
    qa = steady_mean(data, "Q_aortic")
    if qp is None or qa is None or qa <= 0:
        return None
    return qp / qa


# ===========================================================================
# Плавный профиль окклюзии
# ===========================================================================

def occlusion_profile(t, t_onset, severity=1.0, rise_time=5.0):
    """
    Плавный ramp окклюзии.

    Возвращает коэффициент перфузии:
        t <  t_onset                    →  1.0
        t ≥  t_onset, растёт до severity →  1.0 − severity·frac
        frac → 1                        →  1.0 − severity

    Параметры:
        t         — текущее время (с)
        t_onset   — момент начала окклюзии (с)
        severity  — глубина окклюзии ∈ [0, 1]
                    0 — нет окклюзии, 1 — полная остановка перфузии
        rise_time — время нарастания (с), > 0
    """
    # --- Fail-fast на аргументах ---
    severity = _check_fraction("severity", severity, allow_one=True)
    rise_time = float(rise_time)
    if not np.isfinite(rise_time) or rise_time <= 0.0:
        raise ValueError(
            f"utils.occlusion_profile: rise_time={rise_time} должно быть > 0."
        )
    t_onset = float(t_onset)
    if not np.isfinite(t_onset):
        raise ValueError(
            f"utils.occlusion_profile: t_onset={t_onset} не конечно."
        )

    t = float(t)
    if t < t_onset:
        return 1.0
    frac = min((t - t_onset) / rise_time, 1.0)
    return 1.0 - severity * frac