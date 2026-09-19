# tests/check_stationary.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from whole_body import WholeBodyModel
from utils import HR_base

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
    HR = max(out_center['HR'], 20.0)
    T = 60.0 / HR
    mask = (t_arr > t_center - n_cycles * T) & (t_arr <= t_center)

    # уточнение: HR_mean по первичной маске, пересчёт T и маски
    if mask.sum() >= 3:
        hrs = np.array([model.compute_outputs(t_arr[i], data['y'][:, i])['HR']
                        for i in np.where(mask)[0]])
        T = 60.0 / max(hrs.mean(), 20.0)
        mask = (t_arr > t_center - n_cycles * T) & (t_arr <= t_center)

    # Окно: [t_center - n_cycles*T, t_center]
    t_start = t_center - n_cycles * T
    mask = (np.abs(t_arr - t_center) <= 0.5 * n_cycles * T)

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
    model = WholeBodyModel(
        heart_params={'hr': HR_base, 'R_vsd': 5.0},
        baroreflex_params={'P_set': 80.0, 'HR_base': HR_base},
    )
    y0 = model.calibrate_initial_state(t_calib=800)
    sol = model.simulate((0, 600), y0=y0, method='LSODA', max_step=0.07,
                         t_eval=np.arange(0.0, 600.005, 0.05))

    # Упаковываем результат в dict для удобства
    data = {'t': sol.t, 'y': sol.y}

    # Ширина окна в циклах
    N_CYCLES = 5.0

    print(f"Средние за {N_CYCLES:.0f} циклов (установившийся режим):")
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
        for key in ['P_sa', 'P_sv', 'P_pv', 'P_pa', 'Q_aortic', 'HR']:
            v400 = results[400][key]
            v600 = results[600][key]
            drift = abs(v600 - v400)
            flag = "OK" if drift < 1.0 else "DRIFT"
            print(f"  {key:6s}: {v400:6.2f} → {v600:6.2f}  "
                  f"(Δ={drift:5.2f})  [{flag}]")


if __name__ == "__main__":
    main()