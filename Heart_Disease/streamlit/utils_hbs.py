# streamlit/utils_hbs.py
"""
Jism HBS — утилиты для страниц механистической модели (двуязычные).
Все пути вычисляются относительно этого файла.
"""
from pathlib import Path
from functools import lru_cache

import numpy as np
import streamlit as st

from i18n import t


# -------------------------------------------------------------------
# Пути
# -------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # .../Heart_Disease/streamlit
_PROJECT_ROOT = _HERE.parent                     # .../Heart_Disease

HBS_RESULTS_DIR = _PROJECT_ROOT / "hbs_results"
HBS_PLOTS_DIR   = HBS_RESULTS_DIR / "plots"


# -------------------------------------------------------------------
# Реестр дашбордов: (файл, title_key, descr_key, group_key)
# Языконезависимый — локализация в get_dashboards().
# -------------------------------------------------------------------
DASHBOARD_REGISTRY: list[tuple[str, str, str, str]] = [
    # --- Итоговые сводки / Summary ---
    ("comprehensive_dashboard.png",
     "dash.comprehensive.title",
     "dash.comprehensive.descr",
     "group.summary"),
    ("hemodynamics_comparison_enhanced.png",
     "dash.hemo_cmp.title",
     "dash.hemo_cmp.descr",
     "group.summary"),

    # --- Сердце и лёгкие / Heart & Lungs ---
    ("fig2_phase_portraits.png",
     "dash.pv_loops.title",
     "dash.pv_loops.descr",
     "group.heart_lungs"),
    ("fig4_detailed_cardiac.png",
     "dash.cardiac_detail.title",
     "dash.cardiac_detail.descr",
     "group.heart_lungs"),

    # --- Газообмен / Gas Exchange ---
    ("fig5_gas_exchange.png",
     "dash.gas.title",
     "dash.gas.descr",
     "group.gas"),

    # --- Гемодинамика / Hemodynamics ---
    ("fig1_hemodynamics_timeseries.png",
     "dash.timeseries.title",
     "dash.timeseries.descr",
     "group.hemo"),
    ("fig3_bar_comparison.png",
     "dash.steady.title",
     "dash.steady.descr",
     "group.hemo"),
    ("shunt_effect_analysis.png",
     "dash.shunt.title",
     "dash.shunt.descr",
     "group.hemo"),

    # --- Схемы / Schematics ---
    ("schematic_heart_comparison.png",
     "dash.schematic.title",
     "dash.schematic.descr",
     "group.schemas"),
]


def get_dashboards() -> list[tuple[str, str, str, str]]:
    """
    Локализованный список дашбордов: (файл, заголовок, описание, группа).
    Вызывать на каждой перерисовке страницы — при смене языка вернётся
    новый набор строк.
    """
    return [
        (fname, t(title_key), t(descr_key), t(group_key))
        for fname, title_key, descr_key, group_key in DASHBOARD_REGISTRY
    ]


# -------------------------------------------------------------------
# Локализация меток сценариев
# -------------------------------------------------------------------
# Метки приходят из .npz (записаны patient_*.yaml) и всегда на русском.
# Здесь — карта «оригинал → i18n-ключ». Dict-ключ load_all_scenarios()
# остаётся оригиналом, чтобы st.selectbox был стабилен при смене языка.
SCENARIO_LABEL_KEYS: dict[str, str] = {
    "Здоровый":                       "scenario.healthy",
    "Малый ДМЖП (R=5.0)":             "scenario.vsd_small",
    "Большой ДМЖП (R=1.0)":           "scenario.vsd_large",
    "Эйзенменгер, компенсированный":  "scenario.eisenmenger_comp",
    "Эйзенменгер, декомпенсированный":"scenario.eisenmenger_decomp",
}


def scenario_display_name(original_label: str) -> str:
    """
    Отображаемое имя сценария на текущем языке.
    Если метки нет в карте — вернётся оригинал (fail-safe).
    """
    key = SCENARIO_LABEL_KEYS.get(original_label)
    return t(key) if key else original_label


# -------------------------------------------------------------------
# Загрузка .npz
# -------------------------------------------------------------------
@lru_cache(maxsize=16)
def _load_npz_cached(path_str: str):
    with np.load(path_str, allow_pickle=True) as npz:
        label = str(npz["label"]) if "label" in npz.files else "unknown"
        payload = {
            k: np.array(npz[k]) for k in npz.files
            if k not in ("label", "id", "description")
        }
    return label, payload


def load_all_scenarios() -> dict[str, dict]:
    """
    Возвращает dict {original_label: payload} для всех .npz.
    Ключ — оригинальная русская метка; для отображения используйте
    scenario_display_name(label). Так st.selectbox не «прыгает» при
    смене языка.
    """
    files = sorted(HBS_RESULTS_DIR.glob("vsd_results_patient_*.npz"))
    if not files:
        return {}

    scenarios: dict[str, dict] = {}
    for path in files:
        try:
            label, payload = _load_npz_cached(str(path))
            scenarios[label] = payload
        except Exception as e:
            st.warning(t("hbs.load_error", name=path.name, err=str(e)))
    return scenarios


# -------------------------------------------------------------------
# Утилиты для временных рядов (языконезависимы)
# -------------------------------------------------------------------
def steady_mask(t_arr: np.ndarray, frac: float = 0.75) -> np.ndarray:
    """Маска последних (1 - frac) симуляции."""
    if t_arr.size == 0:
        return np.zeros(0, dtype=bool)
    return t_arr >= frac * t_arr[-1]


def steady_mean(data: dict, key: str, scale: float = 1.0):
    """Среднее по установившемуся окну. None, если ключа нет."""
    if key not in data or "t" not in data:
        return None
    t_arr = np.asarray(data["t"])
    m = steady_mask(t_arr) & np.isfinite(np.asarray(data[key]))
    if not np.any(m):
        return None
    return float(np.mean(data[key][m]) * scale)


def qp_qs_steady(data: dict):
    """Qp/Qs = mean(Q_pulmonary) / mean(Q_aortic)."""
    qp = steady_mean(data, "Q_pulmonary")
    qa = steady_mean(data, "Q_aortic")
    if qp is None or qa is None or qa <= 0:
        return None
    return qp / qa


# -------------------------------------------------------------------
# Проверка артефактов (использует DASHBOARD_REGISTRY — языконезависимо)
# -------------------------------------------------------------------
def artifacts_ready() -> tuple[bool, list[str]]:
    """
    Проверяет, что все артефакты на месте.
    Возвращает (ok, missing_list). Сообщения — языконезависимые
    (пути и имена файлов); локализуются на уровне страницы.
    """
    missing: list[str] = []
    if not HBS_RESULTS_DIR.exists():
        missing.append(str(HBS_RESULTS_DIR))
        return False, missing

    npz = list(HBS_RESULTS_DIR.glob("vsd_results_patient_*.npz"))
    if len(npz) < 5:
        missing.append(
            f"hbs_results/vsd_results_patient_*.npz ({len(npz)}/5)"
        )

    for fname, _, _, _ in DASHBOARD_REGISTRY:
        if not (HBS_PLOTS_DIR / fname).exists():
            missing.append(f"hbs_results/plots/{fname}")

    return len(missing) == 0, missing