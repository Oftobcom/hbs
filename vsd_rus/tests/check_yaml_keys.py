"""Проверяет, что все ключи в physiology.yaml известны конструкторам органов.
Защищает от опечаток и устаревших ключей."""
import sys
from pathlib import Path

# --- ROOT: родительская директория тестов, откуда импортируются модули ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import inspect
from heart import Heart4Chambers
from lungs import Lungs2Chamber
from liver import Liver
from kidney import KidneyHemodynamic
from brain import Brain
from gitract import GITract
from gas_exchange import GasExchange
from peripheral_tissues import PeripheralTissues
from jugular_vein import JugularVein
from baroreflex import Baroreflex
from physio_config import load_physiology
from blood import BloodPool

_CHECKS = [
    ('heart',         Heart4Chambers),
    ('lungs',         Lungs2Chamber),
    ('baroreflex',    Baroreflex),
    ('peripheral',    PeripheralTissues),
    ('liver',         Liver),
    ('kidney',        KidneyHemodynamic),
    ('brain',         Brain),
    ('gitract',       GITract),
    ('gas_exchange',  GasExchange),
    ('jugular_vein',  JugularVein),
]

# --- Meta-секции: проверяем обязательные ключи, а не сигнатуры ---
_REQUIRED_SYSTEMIC_KEYS = (
    'target_MAP', 'target_CO', 'C_sys_art', 'C_pul_ven',
    'P_sa0', 'P_sv0', 'P_pv0', 'SYS_VEN_FRACTION', 'C_sys_ven_eff',
    'R_sys_peripheral', 'VO2_rest', 'RQ', 'occlusion_factor',
    'fluid_intake_rate', 'insensible_loss_rate',
)
_REQUIRED_SIMULATION_KEYS = (
    'method', 'rtol', 'atol', 'max_step', 't_calib',
    't_span', 'n_samples_t', 'steady_frac',
)

_REQUIRED_BLOOD_KEYS = ('V0', 'initial_concentrations')

# В `check_yaml_keys.py`, перед основным циклом:

# Параметры сигнатуры, которые приходят НЕ из physiology.yaml,
# а из других источников (patient_*.yaml, sim_builder). Не считаем
# их отсутствие в physiology.yaml ошибкой.
_EXTERNAL_KEYS = {
    'heart': {'R_vsd'},   # приходит из patient.vsd_resistance
}

def _check_required_in_yaml(section: str, cls, phys: dict) -> bool:
    sig = inspect.signature(cls.__init__).parameters
    required = {
        name for name, p in sig.items()
        if name != 'self'
        and p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    required -= _EXTERNAL_KEYS.get(section, set())
    if not required:
        return True   # нет required-параметров — нечего проверять, не шумим

    missing = required - set(phys[section])
    if missing:
        print(f"  ✗ {section:15s}: в YAML нет required-ключей "
              f"{sorted(missing)}")
        return False
    print(f"  ✓ {section:15s}: все required-ключи в YAML ({len(required)})")
    return True

def _check_meta_sections(phys: dict) -> bool:
    ok = True
    for section, required in (
        ('systemic',   _REQUIRED_SYSTEMIC_KEYS),
        ('simulation', _REQUIRED_SIMULATION_KEYS),
        ('blood',      _REQUIRED_BLOOD_KEYS),
    ):
        missing = [k for k in required if k not in phys[section]]
        if missing:
            print(f"  ✗ {section:15s}: пропущены ключи {missing}")
            ok = False
        else:
            print(f"  ✓ {section:15s}: OK ({len(phys[section])} keys)")
    return ok

def main() -> None:
    phys = load_physiology()
    failed = False
    for section, cls in _CHECKS:
        # Forward: все ключи YAML известны конструктору
        allowed = set(inspect.signature(cls.__init__).parameters) - {'self'}
        unknown = set(phys[section]) - allowed
        if unknown:
            print(f"  ✗ {section:15s}: неизвестные ключи {sorted(unknown)}")
            failed = True
        else:
            print(f"  ✓ {section:15s}: OK ({len(phys[section])} keys)")
        # Reverse: все required-параметры конструктора есть в YAML
        if not _check_required_in_yaml(section, cls, phys):
            failed = True

    if not _check_meta_sections(phys):
        failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()