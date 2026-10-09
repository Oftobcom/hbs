# ml/identifiability_config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

from physio_config import load_physiology

DEFAULT_IDENT_PATH = Path(__file__).parent.parent / "config" / "identifiability.yaml"


@dataclass(frozen=True)
class IdentifiabilityConfig:
    vary_params: tuple[str, ...]
    alt_params: tuple[str, ...]
    theta0: dict[str, float | None]
    param_scales: dict[str, float | None]
    param_bounds: dict[str, tuple[float, float]]
    x_names: tuple[str, ...]
    x_typical_scale: dict[str, float]
    d_vsd_ref: float
    r_vsd_ref: float
    k_vsd: float
    rel_step: float
    d_vsd_min_delta: float
    cond_threshold: float
    priority_to_fix: tuple[str, ...]
    n_fix_primary: int
    thr_priority: float
    thr_other: float
    extra_candidates: tuple[str, ...]
    alt_replaces: str
    sim_cfg: dict
    hard_limits: dict
    report_top_k: int
    physiology: dict = field(repr=False)
    ident: dict = field(repr=False)
    target_n_keep: int


def _validate_against_physiology(physio: dict, ident: dict) -> None:
    for p, src in ident["theta_sources"].items():
        if src.get("from") == "geometry":
            continue
        sec, key = src["section"], src["key"]
        if sec not in physio:
            raise ValueError(f"identifiability.yaml: theta_sources[{p!r}] → "
                             f"нет секции physiology.{sec!r}")
        if key not in physio[sec]:
            raise ValueError(f"identifiability.yaml: theta_sources[{p!r}] → "
                             f"нет ключа physiology.{sec}.{key!r}")


def _project_theta0(physio: dict, ident: dict) -> dict[str, float | None]:
    sources = ident["theta_sources"]
    geom = ident["vsd_geometry"]
    out: dict[str, float | None] = {}
    for p in list(ident["vary_params"]) + list(ident.get("alt_params", [])):
        src = sources.get(p)
        if src is None:
            raise ValueError(f"identifiability.yaml: нет theta_sources[{p!r}]")
        if src.get("from") == "geometry":
            out[p] = float(geom["d_vsd_ref_mm"])
            continue
        v = physio[src["section"]].get(src["key"])
        if v is None and p != "R_sys":
            raise ValueError(f"physiology.yaml: {src['section']}.{src['key']} = None "
                             f"для параметра {p!r}")
        out[p] = None if v is None else float(v)
    return out


def load_identifiability_config(
    ident_path: Optional[str | Path] = None,
    physio_path: Optional[str] = None,
    physio_overrides: Optional[dict] = None,
) -> IdentifiabilityConfig:
    ipath = Path(ident_path) if ident_path else DEFAULT_IDENT_PATH
    if not ipath.exists():
        raise FileNotFoundError(f"identifiability.yaml не найден: {ipath}")
    with open(ipath, "r", encoding="utf-8") as f:
        ident = yaml.safe_load(f)
    if not isinstance(ident, dict):
        raise ValueError(f"{ipath.name}: верхний уровень должен быть dict.")

    physio = load_physiology(physio_path, physio_overrides)
    _validate_against_physiology(physio, ident)

    geom = ident["vsd_geometry"]
    d_ref, r_ref = float(geom["d_vsd_ref_mm"]), float(geom["r_vsd_ref"])
    k_vsd = r_ref * (d_ref / 2.0) ** 4

    scales = {k: (None if v is None else float(v))
              for k, v in ident["param_scales"].items()}
    bounds = {k: (float(v[0]), float(v[1]))
              for k, v in ident["param_bounds"].items()}

    sim = dict(physio.get("simulation", {}))
    sim.update(ident.get("simulation", {}))          # override для Stage 1

    jac = ident.get("jacobian", {})
    sel = ident["selection"]
    target_n_keep = int(sel.get("target_n_keep", 6))

    return IdentifiabilityConfig(
        vary_params=tuple(ident["vary_params"]),
        alt_params=tuple(ident.get("alt_params", [])),
        theta0=_project_theta0(physio, ident),
        param_scales=scales,
        param_bounds=bounds,
        x_names=tuple(ident["x_names"]),
        x_typical_scale={k: float(v) for k, v in ident["x_typical_scale"].items()},
        d_vsd_ref=d_ref, r_vsd_ref=r_ref, k_vsd=k_vsd,
        rel_step=float(jac.get("rel_step", 0.01)),
        d_vsd_min_delta=float(jac.get("d_vsd_min_delta", 0.04)),
        cond_threshold=float(sel["cond_threshold"]),
        priority_to_fix=tuple(sel["priority_to_fix"]),
        n_fix_primary=int(sel["n_fix_primary"]),
        thr_priority=float(sel["thr_priority"]),
        thr_other=float(sel["thr_other"]),
        extra_candidates=tuple(sel["extra_candidates"]),
        alt_replaces=str(sel["alt_replaces"]),
        sim_cfg=sim,
        hard_limits=dict(ident.get("hard_limits", {})),
        report_top_k=int(ident.get("report", {}).get("top_param_to_log", 3)),
        physiology=physio,
        ident=ident,
        target_n_keep=target_n_keep
    )

def _validate_ident(ident: dict) -> None:
    vary = list(ident["vary_params"])
    alt = list(ident.get("alt_params", []))
    all_params = vary + alt
    for key in ("param_scales", "param_bounds"):
        missing = [p for p in all_params if p not in ident[key]]
        if missing:
            raise ValueError(
                f"identifiability.yaml: {key} не содержит {missing} "
                f"(нужны для vary_params ∪ alt_params)."
            )
    missing_src = [p for p in all_params
                   if p not in ident["theta_sources"]]
    if missing_src:
        raise ValueError(
            f"identifiability.yaml: theta_sources не содержит "
            f"{missing_src}."
        )
    sel = ident["selection"]
    for p in sel.get("priority_to_fix", []):
        if p not in vary:
            raise ValueError(
                f"identifiability.yaml: priority_to_fix содержит {p!r}, "
                f"но его нет в vary_params."
            )