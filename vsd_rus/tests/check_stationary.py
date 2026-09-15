# tests/check_stationary.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel


def cycle_mean(data, model, t_center, n_cycles=1.0):
    """
    Возвращает средние за окно n_cycles кардиоциклов,
    заканчивающееся в t_center.

    Параметры
    ---------
    data : dict
        Результат model.simulate (или словарь с ключами 't', 'y').
    model : WholeBodyModel
    t_center : float
        Центр окна усреднения.
    n_cycles : float
        Ширина окна в кардиоциклах (1.0 = один цикл, 5.0 = пять циклов).

    Возвращает
    ----------
    dict усреднённых значений и саму маску.
    """
    t_arr = data['t']

    # Оцениваем HR из мгновенного состояния в t_center
    idx_center = int(np.argmin(np.abs(t_arr - t_center)))
    y_center = data['y'][:, idx_center]
    out_center = model.compute_outputs(t_arr[idx_center], y_center)
    HR = out_center['HR']
    T = 60.0 / max(HR, 1e-6)

    # Окно: [t_center - n_cycles*T, t_center]
    t_start = t_center - n_cycles * T
    mask = (t_arr > t_start) & (t_arr <= t_center)

    if mask.sum() < 3:
        return None, mask  # слишком узкое окно

    # Усредняем всё, что нужно
    keys = ['P_sa', 'P_sv', 'P_pv', 'P_pa',
            'Q_aortic', 'Q_pulmonary', 'HR']
    means = {k: [] for k in keys}
    for i in np.where(mask)[0]:
        out = model.compute_outputs(t_arr[i], data['y'][:, i])
        for k in keys:
            means[k].append(out[k])

    result = {k: float(np.mean(v)) for k, v in means.items()}
    result['t_center'] = t_center
    result['t_start'] = t_start
    result['n_points'] = int(mask.sum())
    return result, mask


def main():
    model = WholeBodyModel(vsd_resistance=5.0)
    y0 = model.calibrate_initial_state(t_calib=60, p_sa_lo=20.0)

    sol = model.simulate((0, 600), y0=y0, method='LSODA',
                         max_step=0.05,
                         t_eval=np.linspace(0, 600, 4000))

    # Упаковываем результат в dict для удобства
    data = {'t': sol.t, 'y': sol.y}

    # Ширина окна в циклах
    N_CYCLES = 5.0

    print(f"Средние за {N_CYCLES:.0f} циклов "
          f"(все величины усреднены, как в compare_configs.py):")
    print(f"{'t, с':>6}  {'P_sa':>7}  {'P_sv':>7}  {'P_pv':>7}  "
          f"{'P_pa':>7}  {'Q_aortic':>10}  {'HR':>7}")
    print("-" * 70)

    results = {}
    for t_check in [100, 200, 300, 400, 500, 600]:
        res, mask = cycle_mean(data, model, t_check, n_cycles=N_CYCLES)
        if res is None:
            print(f"{t_check:6d}  (недостаточно точек в окне)")
            continue
        results[t_check] = res
        print(f"{t_check:6d}  {res['P_sa']:7.2f}  {res['P_sv']:7.2f}  "
              f"{res['P_pv']:7.2f}  {res['P_pa']:7.2f}  "
              f"{res['Q_aortic']:10.2f}  {res['HR']:7.2f}")

    # Дрейф между t=400 и t=600 по усреднённым значениям
    if 400 in results and 600 in results:
        print(f"\nДрейф между t=400 и t=600 "
              f"(по средним за {N_CYCLES:.0f} циклов):")
        for key in ['P_sa', 'P_sv', 'P_pv', 'P_pa']:
            v400 = results[400][key]
            v600 = results[600][key]
            drift = abs(v600 - v400)
            flag = "OK" if drift < 1.0 else "DRIFT"
            print(f"  {key:6s}: {v400:6.2f} → {v600:6.2f}  "
                  f"(Δ={drift:5.2f})  [{flag}]")


if __name__ == "__main__":
    main()