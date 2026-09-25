# hbs
# HBS – Human Body Simulation is a modular Python framework
# for multi-organ physiological modeling.
# physio_config.py
"""
Загрузка конфигураций HBS.

Функции:
    load_physiology(path, overrides)  — physiology.yaml + рекурсивный merge
    load_patient(path)                — один patient_*.yaml
    load_all_patients(config_dir)     — все patient_*.yaml по порядку `order`

Особенности:
    • Кэш YAML по (path, mtime) — файл перечитывается только при изменении
    • Резолв YAML-строки 'inf' → np.inf (и 'inf'/'.inf' от PyYAML)
    • Валидация структуры и значений — fail-fast
"""

from functools import lru_cache
from pathlib import Path
import os
from typing import Optional, List

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Пути по умолчанию
# ---------------------------------------------------------------------------
DEFAULT_PATH = Path(__file__).parent / "config" / "physiology.yaml"
DEFAULT_PATIENT_DIR = Path(__file__).parent / "config"


# ---------------------------------------------------------------------------
# Обязательные ключи пациента
# ---------------------------------------------------------------------------
_REQUIRED_PATIENT_KEYS = (
    "id", "label", "order", "color",
    "vsd_resistance", "flow_dependent_lungs", "pressure_remodel",
)


# ===========================================================================
# Module-level валидация констант — fail-fast при импорте.
# ===========================================================================
def _validate_module_constants() -> None:
    """Проверка констант при импорте. RuntimeError → fail-fast."""
    if not isinstance(_REQUIRED_PATIENT_KEYS, tuple):
        raise RuntimeError(
            f"physio_config: _REQUIRED_PATIENT_KEYS должен быть tuple, "
            f"получено {type(_REQUIRED_PATIENT_KEYS).__name__}."
        )
    if len(_REQUIRED_PATIENT_KEYS) == 0:
        raise RuntimeError(
            "physio_config: _REQUIRED_PATIENT_KEYS пуст."
        )
    if not all(isinstance(k, str) and k for k in _REQUIRED_PATIENT_KEYS):
        raise RuntimeError(
            "physio_config: _REQUIRED_PATIENT_KEYS должен содержать "
            "непустые строки."
        )
    if len(set(_REQUIRED_PATIENT_KEYS)) != len(_REQUIRED_PATIENT_KEYS):
        raise RuntimeError(
            f"physio_config: _REQUIRED_PATIENT_KEYS содержит дубликаты: "
            f"{_REQUIRED_PATIENT_KEYS}."
        )


_validate_module_constants()


# ===========================================================================
# Вспомогательные проверки
# ===========================================================================
def _check_nonempty_str(name: str, v) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(
            f"physio_config: {name} должен быть непустой строкой, "
            f"получено {v!r}."
        )
    return v


def _check_int_positive(name: str, v) -> int:
    if isinstance(v, bool):
        raise ValueError(
            f"physio_config: {name}={v} — bool недопустим как целое."
        )
    try:
        iv = int(v)
    except (TypeError, ValueError):
        raise ValueError(
            f"physio_config: {name}={v!r} не конвертируется в int."
        )
    if iv <= 0:
        raise ValueError(
            f"physio_config: {name}={iv} должно быть > 0."
        )
    return iv


def _check_bool(name: str, v) -> bool:
    if not isinstance(v, bool):
        raise ValueError(
            f"physio_config: {name}={v!r} должен быть bool, "
            f"получено {type(v).__name__}."
        )
    return v


def _check_finite_range(name: str, v, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(
            f"physio_config: {name}={v!r} должен быть числом."
        )
    fv = float(v)
    if not np.isfinite(fv) or not (lo <= fv <= hi):
        raise ValueError(
            f"physio_config: {name}={fv} вне [{lo}, {hi}]."
        )
    return fv


def _check_hex_color(name: str, v) -> str:
    """Проверка цвета вида #RGB или #RRGGBB."""
    if not isinstance(v, str):
        raise ValueError(
            f"physio_config: {name}={v!r} должен быть строкой."
        )
    if not v.startswith("#") or len(v) not in (4, 7):
        raise ValueError(
            f"physio_config: {name}={v!r} — ожидается '#RGB' или '#RRGGBB'."
        )
    try:
        int(v[1:], 16)
    except ValueError:
        raise ValueError(
            f"physio_config: {name}={v!r} содержит не-hex символы."
        )
    return v


def _check_positive_or_inf(name: str, v) -> float:
    """Значение > 0 или np.inf. Используется для vsd_resistance."""
    if isinstance(v, bool):
        raise ValueError(
            f"physio_config: {name}={v!r} — bool недопустим."
        )
    if isinstance(v, (int, float)):
        fv = float(v)
        if np.isinf(fv) and fv > 0:
            return fv
        if np.isfinite(fv) and fv > 0:
            return fv
        raise ValueError(
            f"physio_config: {name}={fv} должно быть > 0 или +inf."
        )
    if isinstance(v, str) and v.lower() == "inf":
        return np.inf
    raise ValueError(
        f"physio_config: {name}={v!r} должно быть числом > 0 или 'inf'."
    )


# ===========================================================================
# Резолв 'inf' → np.inf
# ===========================================================================
_MAX_RECURSION_DEPTH = 50


def _resolve_inf(obj, _depth: int = 0):
    """
    Рекурсивно заменяет YAML-строку 'inf' → np.inf.

    Не трогает ключи: 'null' из YAML приходит сюда уже как None
    (yaml.safe_load), и его интерпретация — ответственность
    вызывающего кода (build_model / WholeBodyModel.__init__).

    Глубина рекурсии ограничена (fail-fast при патологически
    глубокой структуре).
    """
    if _depth > _MAX_RECURSION_DEPTH:
        raise RuntimeError(
            f"physio_config: _resolve_inf превысил глубину рекурсии "
            f"{_MAX_RECURSION_DEPTH} — подозрительная структура YAML."
        )
    if isinstance(obj, dict):
        return {k: _resolve_inf(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_inf(v, _depth + 1) for v in obj]
    if isinstance(obj, str) and obj.lower() == "inf":
        return np.inf
    return obj


# ===========================================================================
# Кэш сырого YAML по (path, mtime): файл перечитывается только при изменении
# ===========================================================================
@lru_cache(maxsize=8)
def _load_yaml_cached(path_str: str, mtime: float) -> dict:
    """
    Возвращает сырой dict из YAML (без резолва inf).

    Может вернуть None, если файл пуст или содержит только комментарии —
    вызывающий код обязан это проверить (fail-fast).
    """
    with open(path_str, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_yaml_checked(path: Path) -> dict:
    """
    Загружает YAML с проверками:
      • файл существует и является файлом
      • YAML парсится в непустой dict
    """
    if not path.exists():
        raise FileNotFoundError(
            f"physio_config: файл не найден: {path}."
        )
    if not path.is_file():
        raise ValueError(
            f"physio_config: путь {path} не является файлом."
        )
    try:
        mtime = os.path.getmtime(path)
    except OSError as e:
        raise RuntimeError(
            f"physio_config: не удалось получить mtime для {path}: {e}."
        )
    cfg = _load_yaml_cached(str(path), mtime)
    if cfg is None:
        raise ValueError(
            f"physio_config: {path.name} пуст или содержит только комментарии."
        )
    if not isinstance(cfg, dict):
        raise ValueError(
            f"physio_config: {path.name} — верхний уровень должен быть dict, "
            f"получено {type(cfg).__name__}."
        )
    if len(cfg) == 0:
        raise ValueError(
            f"physio_config: {path.name} — пустой dict."
        )
    return cfg


# ===========================================================================
# load_physiology
# ===========================================================================
def load_physiology(path: Optional[str] = None,
                    overrides: Optional[dict] = None) -> dict:
    """
    Загружает YAML с параметрами физиологии.

    Параметры
    ---------
    path      : путь к YAML (по умолчанию config/physiology.yaml)
    overrides : dict, который рекурсивно перекрывает загруженное

    Возвращает
    ----------
    dict с секциями heart, lungs, baroreflex, ...
    Все строки 'inf' заменены на np.inf.

    Ошибки
    ------
    FileNotFoundError  — файл не найден
    ValueError         — файл пуст / не dict / overrides не dict
    RuntimeError       — mtime недоступен, рекурсия слишком глубокая
    """
    p = Path(path) if path else DEFAULT_PATH
    cfg = _load_yaml_checked(p)
    cfg = _resolve_inf(cfg)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError(
                f"physio_config.load_physiology: overrides должен быть dict, "
                f"получено {type(overrides).__name__}."
            )
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Рекурсивное слияние: override перекрывает base."""
    if not isinstance(base, dict):
        raise ValueError(
            f"physio_config._deep_merge: base должен быть dict, "
            f"получено {type(base).__name__}."
        )
    if not isinstance(override, dict):
        raise ValueError(
            f"physio_config._deep_merge: override должен быть dict, "
            f"получено {type(override).__name__}."
        )
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ===========================================================================
# Загрузка описаний пациентов
# ===========================================================================
def _validate_patient(cfg: dict, path: Path) -> None:
    """
    Валидация полей одного пациента — fail-fast.

    Проверки:
      • Все _REQUIRED_PATIENT_KEYS присутствуют
      • id, label — непустые строки
      • order — целое > 0
      • color — hex-строка #RGB или #RRGGBB
      • vsd_resistance — число > 0 или 'inf'/np.inf
      • flow_dependent_lungs, pressure_remodel — bool
      • HR_base (если задан) — ∈ [20, 250]
      • E_max_rv (если задан) — ∈ [0.01, 20]
    """
    if not isinstance(cfg, dict):
        raise ValueError(
            f"physio_config: {path.name} — ожидался dict, "
            f"получено {type(cfg).__name__}."
        )

    # --- Обязательные ключи ---
    missing = [k for k in _REQUIRED_PATIENT_KEYS if k not in cfg]
    if missing:
        raise ValueError(
            f"physio_config: {path.name} — отсутствуют обязательные "
            f"ключи {missing}."
        )

    # --- Типы и значения ---
    _check_nonempty_str(f"{path.name}.id",    cfg["id"])
    _check_nonempty_str(f"{path.name}.label", cfg["label"])
    _check_int_positive(f"{path.name}.order", cfg["order"])
    _check_hex_color(f"{path.name}.color",    cfg["color"])
    _check_positive_or_inf(f"{path.name}.vsd_resistance", cfg["vsd_resistance"])
    _check_bool(f"{path.name}.flow_dependent_lungs", cfg["flow_dependent_lungs"])
    _check_bool(f"{path.name}.pressure_remodel",     cfg["pressure_remodel"])

    # --- Опциональные ключи ---
    if "HR_base" in cfg:
        _check_finite_range(f"{path.name}.HR_base", cfg["HR_base"], 20.0, 250.0)
    if "E_max_rv" in cfg:
        _check_finite_range(f"{path.name}.E_max_rv", cfg["E_max_rv"], 0.01, 20.0)


def load_patient(path) -> dict:
    """
    Загружает один patient_*.yaml.

    Резолвит 'inf' → np.inf и валидирует обязательные поля.
    """
    p = Path(path)
    cfg = _load_yaml_checked(p)
    cfg = _resolve_inf(cfg)
    _validate_patient(cfg, p)
    return cfg


def load_all_patients(config_dir: Optional[str] = None) -> dict:
    """
    Читает все patient_*.yaml из config_dir (по умолчанию ./config),
    сортирует по полю `order`, возвращает {label: cfg}.

    Ошибки
    ------
    FileNotFoundError  — директория или файлы не найдены
    ValueError         — дубликаты label или order
    """
    d = Path(config_dir) if config_dir else DEFAULT_PATIENT_DIR
    if not d.exists():
        raise FileNotFoundError(
            f"physio_config: директория не найдена: {d}."
        )
    if not d.is_dir():
        raise ValueError(
            f"physio_config: путь {d} не является директорией."
        )

    files = sorted(d.glob("patient_*.yaml"))
    if not files:
        raise FileNotFoundError(
            f"physio_config: не найдено patient_*.yaml в {d}."
        )

    patients = [load_patient(f) for f in files]
    patients.sort(key=lambda c: int(c["order"]))

    # --- Проверка уникальности label ---
    labels = [p["label"] for p in patients]
    if len(set(labels)) != len(labels):
        seen = set()
        dupes = []
        for lbl in labels:
            if lbl in seen:
                dupes.append(lbl)
            else:
                seen.add(lbl)
        raise ValueError(
            f"physio_config: дубликаты label среди patients: {sorted(set(dupes))}."
        )

    # --- Проверка уникальности order ---
    orders = [int(p["order"]) for p in patients]
    if len(set(orders)) != len(orders):
        seen = set()
        dupes = []
        for o in orders:
            if o in seen:
                dupes.append(o)
            else:
                seen.add(o)
        raise ValueError(
            f"physio_config: дубликаты order среди patients: {sorted(set(dupes))}."
        )

    return {p["label"]: p for p in patients}