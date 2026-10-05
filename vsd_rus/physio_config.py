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
from typing import Optional
import warnings
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

_REQUIRED_PHYSIOLOGY_SECTIONS = (
    'heart', 'lungs', 'baroreflex', 'blood', 'peripheral',
    'liver', 'kidney', 'brain', 'gitract', 'gas_exchange',
    'systemic', 'jugular_vein', 'simulation',
)

# ---------------------------------------------------------------------------
# Обязательные ключи meta-секций (проверяются после merge physiology+patient)
# ---------------------------------------------------------------------------
_REQUIRED_SYSTEMIC_KEYS = (
    'target_MAP', 'target_CO', 'C_sys_art', 'C_pul_ven',
    'P_sa0', 'P_sv0', 'P_pv0', 'SYS_VEN_FRACTION', 'C_sys_ven_eff',
    'R_sys_peripheral',                            # ← новый
    'VO2_rest', 'RQ', 'occlusion_factor',
    'fluid_intake_rate', 'insensible_loss_rate',
)

_REQUIRED_SIMULATION_KEYS = (
    'method', 'rtol', 'atol', 'max_step',
    't_calib', 't_span', 'n_samples_t', 'steady_frac',
)

_REQUIRED_BLOOD_KEYS = ('V0', 'initial_concentrations')

_VALID_SOLVER_METHODS = frozenset(
    {'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'}
)

def _validate_merged(cfg: dict, label: str) -> None:
    """Проверка итогового cfg после merge physiology + patient."""
    if not isinstance(cfg, dict):
        raise ValueError(
            f"physio_config: merged cfg для '{label}' должен быть dict."
        )
    missing = [s for s in _REQUIRED_PHYSIOLOGY_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(
            f"physio_config: [{label}] после merge отсутствуют секции {missing}."
        )
    # --- systemic ---
    sys = cfg['systemic']
    missing = [k for k in _REQUIRED_SYSTEMIC_KEYS if k not in sys]
    if missing:
        raise ValueError(
            f"physio_config: [{label}].systemic — отсутствуют {missing}."
        )

    R_sys = sys.get('R_sys_peripheral')
    if R_sys is not None:
        _check_finite_range(f"{label}.systemic.R_sys_peripheral",
                            R_sys, 0.1, 100.0)

    # --- simulation ---
    sim = cfg['simulation']
    missing = [k for k in _REQUIRED_SIMULATION_KEYS if k not in sim]
    if missing:
        raise ValueError(
            f"physio_config: [{label}].simulation — отсутствуют {missing}."
        )

    # --- blood ---
    blood = cfg['blood']
    missing = [k for k in _REQUIRED_BLOOD_KEYS if k not in blood]
    if missing:
        raise ValueError(
            f"physio_config: [{label}].blood — отсутствуют {missing}."
        )

    meth = cfg['simulation'].get('method')
    if not isinstance(meth, str) or meth not in _VALID_SOLVER_METHODS:
        raise ValueError(
            f"physio_config: [{label}].simulation.method={meth!r} "
            f"не входит в {sorted(_VALID_SOLVER_METHODS)}. "
            f"Задаётся в config/physiology.yaml."
        )

    if 'diagnostics' in cfg and not isinstance(cfg['diagnostics'], dict):
        raise ValueError(
            f"physio_config: [{label}].diagnostics должен быть dict."
        )
        
    for s in _REQUIRED_PHYSIOLOGY_SECTIONS:
        if not isinstance(cfg[s], dict):
            raise ValueError(
                f"physio_config: [{label}].{s} должен быть dict, "
                f"получено {type(cfg[s]).__name__}."
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

    # --- Поля лёгочного ремоделирования (Цель 1) ---
    if "R_remodel_max" in cfg:
        _check_finite_range(f"{path.name}.R_remodel_max",
                            cfg["R_remodel_max"], 1.0, 20.0)
    if "pressure_sensitivity" in cfg:
        _check_finite_range(f"{path.name}.pressure_sensitivity",
                            cfg["pressure_sensitivity"], 0.0, 2.0)
    if "tau_remodel" in cfg:
        _check_finite_range(f"{path.name}.tau_remodel",
                            cfg["tau_remodel"], 1.0, 1e5)
    if "flow_sensitivity" in cfg:
        _check_finite_range(f"{path.name}.flow_sensitivity",
                            cfg["flow_sensitivity"], 0.0, 5.0)
    # --- Поля гипертрофии ПЖ (heart.py) ---
    if "rv_hypertrophy_sensitivity" in cfg:
        _check_finite_range(f"{path.name}.rv_hypertrophy_sensitivity",
                            cfg["rv_hypertrophy_sensitivity"], 0.0, 5.0)
    # --- Поля пульмонального барорефлекса (baroreflex.py) ---
    if "k_inotropy_pulm" in cfg:
        _check_finite_range(f"{path.name}.k_inotropy_pulm",
                            cfg["k_inotropy_pulm"], 0.0, 5.0)
    if "k_rarefaction" in cfg:
        _check_finite_range(f"{path.name}.k_rarefaction",
                            cfg["k_rarefaction"], 0.0, 2.0)
    if "P_pa_set" in cfg:
        _check_finite_range(f"{path.name}.P_pa_set",
                            cfg["P_pa_set"], 1.0, 100.0)

    # --- Поля трёхветвевого барорефлекса (baroreflex.py) ---
    # Эти ключи пробрасываются per-scenario через
    # run_simulation_parallel.simulate_one_scenario → Baroreflex.
    # Диапазоны согласованы с _check_range в Baroreflex.__init__:
    # при выходе за них fail-fast сработает один раз здесь, а не в RHS.
    if "k_hr" in cfg:
        _check_finite_range(f"{path.name}.k_hr",
                            cfg["k_hr"], 0.0, 0.05)
    if "k_inotropy" in cfg:
        _check_finite_range(f"{path.name}.k_inotropy",
                            cfg["k_inotropy"], 0.0, 0.02)
    if "k_vasomotor" in cfg:
        _check_finite_range(f"{path.name}.k_vasomotor",
                            cfg["k_vasomotor"], 0.0, 0.05)
    if "tau_hr" in cfg:
        _check_finite_range(f"{path.name}.tau_hr",
                            cfg["tau_hr"], 0.1, 10.0)
    if "V_liver" in cfg:
        _check_finite_range(f"{path.name}.V_liver", 
                            cfg["V_liver"], 100.0, 3000.0)
    if "tau_inotropy" in cfg:
        _check_finite_range(f"{path.name}.tau_inotropy",
                            cfg["tau_inotropy"], 0.1, 15.0)
    if "tau_vaso" in cfg:
        _check_finite_range(f"{path.name}.tau_vaso",
                            cfg["tau_vaso"], 1.0, 60.0)        

    rv_sens = cfg.get("rv_hypertrophy_sensitivity", 0.0)
    if rv_sens > 0.0 and not cfg["pressure_remodel"]:
        raise ValueError(
            f"physio_config: {path.name} — "
            f"rv_hypertrophy_sensitivity={rv_sens} > 0, но "
            f"pressure_remodel=False. Гипертрофия ПЖ без лёгочного "
            f"ремоделирования физически невозможна и будет "
            f"молча проигнорирована (rv_afterload ≡ 0)."
        )

    k_pulm = cfg.get("k_inotropy_pulm", 0.0)
    if k_pulm > 0.0 and not cfg["pressure_remodel"]:
        warnings.warn(
            f"physio_config: {path.name} — "
            f"k_inotropy_pulm={k_pulm} > 0, но pressure_remodel=False. "
            f"Без лёгочной гипертензии P_pa ≈ 15 мм рт.ст. и "
            f"пульмональный барорефлекс не активируется."
        )

    thr = cfg.get("P_pa_threshold", 25.0)
    sens = cfg.get("pressure_sensitivity", 0.04)
    P_pa_ref = 60.0
    R_target_at_ref = 1.0 + sens * max(P_pa_ref - thr, 0.0)
    if cfg["pressure_remodel"] and R_target_at_ref < 5.0:
        warnings.warn(
            f"physio_config: {path.name} — при P_pa = {P_pa_ref:.0f} "
            f"(типичное систолическое при Эйзенменгере) "
            f"R_target = {R_target_at_ref:.2f} < 5.0. "
            f"Ремоделирование останется слабым; для Эйзенменгера "
            f"ожидается R_target ≥ 5. Проверьте pressure_sensitivity "
            f"(текущее {sens}) и P_pa_threshold (текущее {thr})."
        )

    # R_remodel_max = 1.0 при pressure_remodel=True означает, что
    # R_remodel никогда не вырастет выше 1.0 → remodeling отключён.
    rmax = cfg.get("R_remodel_max", 5.0)
    if cfg["pressure_remodel"] and rmax <= 1.0:
        warnings.warn(
            f"physio_config: {path.name} — pressure_remodel=True, "
            f"но R_remodel_max={rmax}. R_remodel останется 1.0, "
            f"лёгочное сопротивление не вырастет."
        )

    if "aliases" in cfg:
        aliases = cfg["aliases"]
        if not isinstance(aliases, list):
            raise ValueError(
                f"physio_config: {path.name}.aliases должен быть list, "
                f"получено {type(aliases).__name__}."
            )
        if not all(isinstance(a, str) and a for a in aliases):
            raise ValueError(
                f"physio_config: {path.name}.aliases должен содержать "
                f"непустые строки, получено {aliases!r}."
            )


def load_patient(path, base: Optional[dict] = None) -> dict:
    p = Path(path)
    cfg = _load_yaml_checked(p)
    cfg = _resolve_inf(cfg)
    _validate_patient(cfg, p)          # валидация сырого patient-файла
    if base is not None:
        if not isinstance(base, dict):
            raise ValueError(
                f"physio_config.load_patient: base должен быть dict, "
                f"получено {type(base).__name__}."
            )
        cfg = _deep_merge(base, cfg)
    _validate_merged(cfg, cfg.get("label", p.stem))   # ← после merge
    return cfg


def load_all_patients(config_dir=None,
                      base_physiology: Optional[dict] = None) -> dict:
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

    patients = [load_patient(f, base=base_physiology) for f in files]
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

    # --- Проверка уникальности aliases ---
    seen_aliases: dict[str, str] = {}
    for p in patients:
        for alias in p.get("aliases", []):
            if alias in seen_aliases:
                raise ValueError(
                    f"physio_config: alias '{alias}' встречается у нескольких "
                    f"пациентов: '{seen_aliases[alias]}' и '{p['label']}'."
                )
            seen_aliases[alias] = p["label"]

    return {p["label"]: p for p in patients}