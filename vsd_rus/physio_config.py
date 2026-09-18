# physio_config.py
from functools import lru_cache
from pathlib import Path
import os

import numpy as np
import yaml

DEFAULT_PATH = Path(__file__).parent / "config" / "physiology.yaml"


def _resolve_inf(obj):
    """Рекурсивно заменяет YAML-строку 'inf' → np.inf.

    Не трогает ключи: 'null' из YAML приходит сюда уже как None
    (yaml.safe_load), и его интерпретация — ответственность
    вызывающего кода (build_model / WholeBodyModel.__init__).
    """
    if isinstance(obj, dict):
        return {k: _resolve_inf(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_inf(v) for v in obj]
    if isinstance(obj, str) and obj.lower() == "inf":
        return np.inf
    return obj


# ---------------------------------------------------------------------------
# Кэш сырого YAML по (path, mtime): файл перечитывается только при изменении
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _load_yaml_cached(path_str: str, mtime: float) -> dict:
    with open(path_str, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_physiology(path=None, overrides=None) -> dict:
    """
    Загружает YAML с параметрами.

    Параметры
    ---------
    path      : путь к YAML (по умолчанию config/physiology.yaml)
    overrides : dict, который рекурсивно перекрывает загруженное

    Возвращает
    ----------
    dict с секциями heart, lungs, baroreflex, ...,
    где все 'inf' заменены на np.inf.
    """
    p = Path(path) if path else DEFAULT_PATH
    mtime = os.path.getmtime(p)
    cfg = _load_yaml_cached(str(p), mtime)   # кэш отдаёт СЫРОЙ dict
    cfg = _resolve_inf(cfg)                   # свежая копия — можно безопасно
    if overrides:                             # менять/мержить без порчи кэша
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Рекурсивное слияние: override перекрывает base."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
