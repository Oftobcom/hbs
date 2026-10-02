# tests/check_volume.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# tests/check_volume.py
import numpy as np
from whole_body import WholeBodyModel


def check_volume_invariant(seed=0, n_samples=20, tol=1e-8):
    rng = np.random.default_rng(seed)
    model = WholeBodyModel()
    y0 = model.get_initial_state()
    s = model.idx

    for _ in range(n_samples):
        y = y0 + rng.normal(0, 0.05, size=y0.shape) * np.abs(y0)
        y = np.maximum(y, 1e-3)
        d = model.derivatives(0.0, y)
        dV = (
            d[s['heart']].sum()
            + model.lungs.C1 * d[s['lungs']][0]
            + model.lungs.C2 * d[s['lungs']][1]
            + d[s['sys_ven']][0] + d[s['jugular_vein']][0]
            + model.sys_art.C * d[s['sys_art']][0]
            + model.pul_ven.C * d[s['pul_ven']][0]
            + model.liver.C * d[s['liver']][0]
            + model.liver.C_portal * d[s['liver']][5]
            + model.gitract.C_art * d[s['gitract']][0]
            + model.gitract.C_cap * d[s['gitract']][1]
            + model.brain.C * d[s['brain']][0]
        )
        dV_total = model._compute_organ_flows(0.0, y)['dV_total']
        assert abs(dV - dV_total) < tol, (
            f"Mass balance broken: dV={dV:.6e}, dV_total={dV_total:.6e}, "
            f"diff={dV - dV_total:.3e}"
        )
    print(f"Volume invariant OK ({n_samples} samples, |diff| < {tol})")


if __name__ == "__main__":
    check_volume_invariant()