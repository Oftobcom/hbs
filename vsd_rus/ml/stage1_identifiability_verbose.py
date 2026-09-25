#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml/stage1_identifiability_verbose.py — v7 VERBOSE

Stage 1: анализ идентифицируемости 8 параметров WholeBodyModel
по 7 наблюдаемым выходам (ЭхоКГ-подобным).

Отличия от stage1_identifiability.py:
  - Подробный лог каждого шага: калибровка, солвер, окна конвергенции, X0, P_sv/Q_periph/R_eff
  - Печать J_ij = (ΔX_i/X0_i)/(Δθ_j/scale_j) для каждого параметра (plus/minus)
  - Печать всей SVD: S, Vt, U, |Vt[-1]|
  - Сохранение J в CSV, графика SVD, лога в файл results/
  - Tee-логгер: консоль + файл stage1_verbose.log
  - Fallback A/B с verbose

Запуск:
    python -m ml.stage1_identifiability_verbose
    python ml/stage1_identifiability_verbose.py debug
"""

from __future__ import annotations
import sys, warnings, io
from pathlib import Path
from typing import Optional
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from whole_body import WholeBodyModel
    from physio_config import load_physiology
except ImportError:
    WholeBodyModel = None
    load_physiology = None

# 0. Константы
_D_VSD_REF = 4.0
_R_VSD_REF = 5.0
K_VSD = _R_VSD_REF * (_D_VSD_REF / 2.0) ** 4

PARAM_NAMES = ["d_vsd","E_max_lv","E_max_rv","R_sys","flow_sensitivity","C_sys_art","HR_base","V0_blood"]
THETA0 = {"d_vsd":4.0,"E_max_lv":3.0,"E_max_rv":0.8,"R_sys":None,"flow_sensitivity":0.1,"C_sys_art":1.5,"HR_base":70.0,"V0_blood":5800.0}
PARAM_SCALES = {"d_vsd":4.0,"E_max_lv":2.5,"E_max_rv":0.8,"R_sys":3.8,"flow_sensitivity":0.05,"C_sys_art":1.5,"HR_base":70.0,"V0_blood":5800.0}
PARAM_NAMES_ALT = PARAM_NAMES + ["k_inotropy"]
PARAM_SCALES_ALT = {**PARAM_SCALES, "k_inotropy":0.5}
THETA0_ALT = {**THETA0, "k_inotropy":0.5}
X_NAMES = ["P_sa","P_pa","Q_aortic","Qp_Qs","EDV_LV","EDV_RV","HR"]

def d_vsd_to_R_vsd(d_mm: float) -> float:
    if d_mm <= 0:
        return np.inf
    return K_VSD / ((d_mm/2.0)**4)

def build_model(theta: dict, target_MAP: Optional[float]=None, target_CO: Optional[float]=None, config_path: Optional[str]=None, config_overrides: Optional[dict]=None):
    if load_physiology is None:
        raise RuntimeError("load_physiology не найден")
    cfg = load_physiology(config_path, config_overrides)
    heart_params = dict(cfg["heart"])
    heart_params["E_max_lv"] = theta["E_max_lv"]
    heart_params["E_max_rv"] = theta["E_max_rv"]
    heart_params["hr"] = theta["HR_base"]
    heart_params["R_vsd"] = d_vsd_to_R_vsd(theta["d_vsd"]) if theta["d_vsd"]>0 else np.inf
    lungs_params = dict(cfg["lungs"])
    lungs_params["flow_sensitivity"] = theta["flow_sensitivity"]
    blood_params = {"V0": theta["V0_blood"], "initial_concentrations": cfg["blood"]["initial_concentrations"]}
    baroreflex_params = dict(cfg["baroreflex"])
    baroreflex_params["HR_base"] = theta["HR_base"]
    if "k_inotropy" in theta:
        baroreflex_params["k_inotropy"] = float(theta["k_inotropy"])
    peripheral_params = dict(cfg["peripheral"])
    if peripheral_params.get("R_base") is None:
        peripheral_params.pop("R_base", None)
    sys_cfg = cfg["systemic"]
    if target_MAP is None:
        target_MAP = sys_cfg["target_MAP"]
    if target_CO is None:
        target_CO = sys_cfg["target_CO"]
    return WholeBodyModel(
        heart_params=heart_params, lungs_params=lungs_params,
        liver_params=dict(cfg["liver"]), kidney_params=dict(cfg["kidney"]),
        blood_params=blood_params, gitract_params=dict(cfg["gitract"]),
        brain_params=dict(cfg["brain"]), baroreflex_params=baroreflex_params,
        gas_exchange_params=dict(cfg["gas_exchange"]), peripheral_params=peripheral_params,
        flow_dependent_lungs=True, R_sys_peripheral=theta["R_sys"],
        target_MAP=target_MAP, target_CO=target_CO, C_sys_art=theta["C_sys_art"],
        C_pul_ven=sys_cfg.get("C_pul_ven",15.0), P_sa0=sys_cfg.get("P_sa0",85.0),
        P_sv0=sys_cfg.get("P_sv0",12.0), P_pv0=sys_cfg.get("P_pv0",12.0),
        SYS_VEN_FRACTION=sys_cfg.get("SYS_VEN_FRACTION",0.58),
        tau_target=sys_cfg.get("tau_target",300.0),
        fluid_intake_rate=sys_cfg.get("fluid_intake_rate",0.01),
        insensible_loss_rate=sys_cfg.get("insensible_loss_rate",0.0),
    )

def resolve_R_sys(theta: dict) -> float:
    if theta.get("R_sys") is not None:
        return float(theta["R_sys"])
    m = build_model({**theta, "R_sys": None})
    return float(m.R_sys_peripheral)

def _stage1_sim_cfg(sim_cfg: Optional[dict]) -> dict:
    out = {"method":"LSODA","rtol":1e-4,"atol":1e-5,"max_step":0.1,"t_calib":600.0,"t_end":800.0,"n_samples_t":4000,"t_start_stationary":300.0,"stationary_rel_tol_stage1":0.05}
    if not sim_cfg:
        return out
    out["method"]=str(sim_cfg.get("method",out["method"]))
    out["rtol"]=float(sim_cfg.get("rtol",out["rtol"]))
    out["atol"]=float(sim_cfg.get("atol",out["atol"]))
    out["max_step"]=float(sim_cfg.get("max_step",out["max_step"]))
    t_span=sim_cfg.get("t_span")
    if isinstance(t_span,(list,tuple)) and len(t_span)>=2:
        out["t_end"]=min(float(t_span[1]),1200.0)
    n_samples=sim_cfg.get("n_samples_t")
    if n_samples is not None:
        out["n_samples_t"]=min(int(n_samples),12000)
    if "stationary_rel_tol_stage1" in sim_cfg:
        out["stationary_rel_tol_stage1"]=float(sim_cfg["stationary_rel_tol_stage1"])
    return out

def _collect_outputs(model, sol) -> dict:
    keys=None; rows=[]
    for i,ti in enumerate(sol.t):
        out=model.compute_outputs(ti, sol.y[:,i])
        if keys is None:
            keys=list(out.keys())
        rows.append([out[k] for k in keys])
    data={k: np.array([r[j] for r in rows],dtype=float) for j,k in enumerate(keys)}
    data["t"]=np.asarray(sol.t,dtype=float)
    return data

def _print_window_diagnostics(data, label="WINDOW"):
    try:
        t=data["t"]; win = t > (t[-1] - 10*60/max(np.mean(data["HR"][t>300]),1))
        keys_diag=["P_sa","P_pa","P_sv","P_pv","P_la","P_ra","V_la","V_lv","V_ra","V_rv","V_sv","V_sv_target","V_sv_fraction","V_blood","Q_aortic","Q_pulmonary","Q_sv_to_ra","Q_pv_to_la","Q_peripheral","Q_ven_in","R_eff_peripheral","f_P_myogenic","f_O2_autoreg","HR","GFR","Q_brain","Q_liver_out","Q_renal"]
        print(f"\n--- {label} DIAG (last 10 cycles) ---")
        for k in keys_diag:
            if k in data:
                arr=data[k][win]
                print(f"  {k:22s} mean={np.mean(arr):8.3f} std={np.std(arr):6.3f} min={np.min(arr):7.2f} max={np.max(arr):7.2f} last={arr[-1]:7.2f}")
        if "Q_ven_in" in data and "Q_peripheral" in data:
            print(f"  BALANCE Q_ven_in={np.mean(data['Q_ven_in'][win]):.2f} Q_periph={np.mean(data['Q_peripheral'][win]):.2f} Qa={np.mean(data['Q_aortic'][win]):.2f}")
    except Exception as e:
        print(f"  [diag warn] {e}")

def get_steady_outputs(model, sim_cfg: Optional[dict]=None, verbose: bool=False) -> Optional[dict]:
    cfg=_stage1_sim_cfg(sim_cfg)
    t_end=float(cfg["t_end"]); n_samples=int(cfg["n_samples_t"]); t_calib=float(cfg["t_calib"]); t_start_stationary=float(cfg["t_start_stationary"]); method=str(cfg["method"])
    y0=model.calibrate_initial_state(t_calib=t_calib)
    try:
        print(f"[CALIB] t_calib={t_calib:.1f}s y0 heart={y0[model.idx['heart']]} V_blood={y0[model.idx['blood']][0]:.1f} P_sa0={y0[model.idx['sys_art']][0]:.2f}")
    except Exception as e:
        print(f"[CALIB] y0 diag fail {e}")
    t_eval=np.linspace(0.0,t_end,n_samples)
    try:
        sol=model.simulate((0.0,t_end),t_eval=t_eval,y0=y0,method=method,rtol=float(cfg["rtol"]),atol=float(cfg["atol"]),max_step=float(cfg["max_step"]))
    except Exception as e:
        if verbose:
            print(f"  [warn] solver failed: {e}")
        return None
    print(f"[solver] nfev={sol.nfev} njev={getattr(sol,'njev',0)} t={sol.t[-1]:.1f} y_last heart={sol.y[:4,-1]} success={getattr(sol,'success',False)}")
    if not getattr(sol,"success",False) or sol.y.shape[1]<2:
        return None
    if not np.all(np.isfinite(sol.y[:,-1])):
        return None
    data=_collect_outputs(model,sol)
    print("\n--- CONVERGENCE WINDOWS ---")
    for t_lo,t_hi in [(100,300),(250,350),(300,400),(400,600),(600,800)]:
        m=(data["t"]>=t_lo)&(data["t"]<=t_hi)
        if not np.any(m): continue
        print(f"[{t_lo:4d}-{t_hi:4d}] P_sa={data['P_sa'][m].mean():6.2f}±{data['P_sa'][m].std():5.2f} V_lv={data['V_lv'][m].mean():6.1f} HR={data['HR'][m].mean():5.1f} P_sv={data['P_sv'][m].mean():5.2f} Q_periph={data['Q_peripheral'][m].mean():6.2f} R_eff={data['R_eff_peripheral'][m].mean():5.3f}")
    tail_for_hr=data["t"]>t_start_stationary
    HR_for_stat=float(np.mean(data["HR"][tail_for_hr]))
    T_stat=60.0/HR_for_stat
    dt=float(data["t"][1]-data["t"][0]) if data["t"].size>1 else 0.1
    win_len=max(int(round(10.0*T_stat/max(dt,1e-6))),4)
    ps_recent=data["P_sa"][-2*win_len:]
    mean_early=float(np.mean(ps_recent[:len(ps_recent)//2])); mean_late=float(np.mean(ps_recent[len(ps_recent)//2:]))
    rel_diff=abs(mean_late-mean_early)/max(mean_late,1e-9)
    stat_tol=float(cfg["stationary_rel_tol_stage1"])
    print(f"[STATIONARITY] early={mean_early:.2f} late={mean_late:.2f} rel={rel_diff:.4f} tol={stat_tol}")
    HR_mean=float(np.mean(data["HR"][tail_for_hr])); T=60.0/max(HR_mean,1e-6); window=data["t"]>(data["t"][-1]-10.0*T)
    X={}
    for key in ("P_sa","P_pa","Q_aortic","HR"):
        X[key]=float(np.mean(data[key][window]))
    mean_Qp=float(np.mean(data["Q_pulmonary"][window])); mean_Qa=float(np.mean(data["Q_aortic"][window]))
    X["Qp_Qs"]=mean_Qp/max(mean_Qa,1e-6); X["EDV_LV"]=float(np.max(data["V_lv"][window])); X["EDV_RV"]=float(np.max(data["V_rv"][window]))
    print(f"\n[X0 STEADY] P_sa={X['P_sa']:.2f} P_pa={X['P_pa']:.2f} Qa={mean_Qa:.2f} Qp={mean_Qp:.2f} Qp/Qs={X['Qp_Qs']:.3f} EDV_LV={X['EDV_LV']:.1f} EDV_RV={X['EDV_RV']:.1f} HR={X['HR']:.2f}")
    print(f"         P_sv={np.mean(data['P_sv'][window]):.2f} P_pv={np.mean(data['P_pv'][window]):.2f} V_sv={np.mean(data['V_sv'][window]):.1f}/{np.mean(data['V_sv_target'][window]):.1f} Q_periph={np.mean(data['Q_peripheral'][window]):.2f} R_eff={np.mean(data['R_eff_peripheral'][window]):.3f}")
    if not (40.0 < X["P_sa"] < 180.0 and 5.0 < X["P_pa"] < 80.0 and X["EDV_LV"]>50.0):
        print(f"  [warn] нефизиологично")
    _print_window_diagnostics(data,label="STEADY 10 cycles")
    X["_data"]=data
    return X

def compute_jacobian(theta0: dict, rel_step: float=0.01, sim_cfg: Optional[dict]=None, verbose: bool=True, param_names: Optional[list]=None, scales_override: Optional[dict]=None):
    if param_names is None:
        param_names=PARAM_NAMES
    scales=dict(scales_override) if scales_override is not None else dict(PARAM_SCALES)
    theta0=dict(theta0)
    if theta0.get("R_sys") is None:
        theta0["R_sys"]=resolve_R_sys(theta0)
    scales["R_sys"]=float(theta0["R_sys"])
    if verbose:
        print("="*70+"\n[Stage1] THETA0 RESOLVED")
        for k in param_names:
            print(f"  {k:18s} = {theta0.get(k)} scale={scales.get(k)}")
        print(f"  R_sys resolved = {theta0['R_sys']:.4f} R_vsd(d_vsd={theta0['d_vsd']}mm) = {d_vsd_to_R_vsd(theta0['d_vsd']):.4f}")
        print("="*70)
    model0=build_model(theta0)
    X0=get_steady_outputs(model0,sim_cfg=sim_cfg,verbose=verbose)
    if X0 is None:
        raise RuntimeError("Базовая точка не вышла на стационар")
    if verbose:
        print("\n[Stage1] X0 BASELINE:")
        for k in X_NAMES:
            print(f"    {k:12s} = {X0[k]:.6f}")
    n_x,n_p=len(X_NAMES),len(param_names)
    J=np.full((n_x,n_p),np.nan)
    for j,p in enumerate(param_names):
        scale=float(scales[p]); delta=rel_step*scale
        if p=="d_vsd": delta=max(delta,0.04)
        theta_plus=dict(theta0); theta_minus=dict(theta0)
        theta_plus[p]=theta0[p]+delta; theta_minus[p]=theta0[p]-delta
        if p=="d_vsd": theta_minus[p]=max(theta_minus[p],0.5)
        print(f"\n--- J param {j} {p} scale={scale:.4g} delta={delta:.4g} theta={theta0[p]:.4g} -> plus={theta_plus[p]:.4g} minus={theta_minus[p]:.4g} ---")
        Xp=get_steady_outputs(build_model(theta_plus),sim_cfg=sim_cfg,verbose=False)
        Xm=get_steady_outputs(build_model(theta_minus),sim_cfg=sim_cfg,verbose=False)
        if Xp is None or Xm is None:
            print(f"  [warn] {p}: Xp={Xp is not None}, Xm={Xm is not None} → NaN")
            continue
        for i,x in enumerate(X_NAMES):
            dX=(Xp[x]-Xm[x])/(2.0*delta)
            J[i,j]=dX*scale/max(abs(X0[x]),1e-9)
        for x in X_NAMES:
            print(f"    dX {x:10s}: X0={X0[x]:8.3f} Xp={Xp[x]:8.3f} Xm={Xm[x]:8.3f} dX={Xp[x]-Xm[x]:+8.3f} J={J[X_NAMES.index(x),j]:+8.4f}")
    return J, {k:X0[k] for k in X_NAMES}, theta0

def analyze_svd(J: np.ndarray, param_names: list, x_names: list) -> dict:
    if np.any(np.isnan(J)):
        bad=[param_names[j] for j in range(J.shape[1]) if np.any(np.isnan(J[:,j]))]
        print(f"[warn] NaN в J по {bad} → 0 для SVD")
    J_clean=np.nan_to_num(J,nan=0.0,posinf=0.0,neginf=0.0)
    U,S,Vt=np.linalg.svd(J_clean,full_matrices=False)
    cond=float(S[0]/S[-1]) if S[-1]>1e-12 else np.inf
    v_last=Vt[-1]
    contrib=pd.Series(np.abs(v_last),index=param_names).sort_values(ascending=False)
    print("\n=== SVD FULL ===")
    print(f"S = {S}\ncond = {cond:.4f} S_max={S[0]:.4f} S_min={S[-1]:.6f}")
    print("Vt matrix:")
    for i,row in enumerate(Vt):
        print(f"  Vt[{i}] S={S[i]:.4f} : {['%+.3f'%v for v in row]} -> {param_names}")
    print("|Vt[-1]| contrib:")
    for k,v in contrib.items():
        print(f"  {k:18s} : {v:.6f}")
    return {"U":U,"S":S,"Vt":Vt,"cond":cond,"v_last":v_last,"contrib":contrib,"J_clean":J_clean}

def _fix_and_cond(J, param_names, to_fix):
    keep_idx=[i for i,p in enumerate(param_names) if p not in to_fix]
    keep_names=[param_names[i] for i in keep_idx]
    res=analyze_svd(J[:,keep_idx],keep_names,X_NAMES)
    return res["cond"], res["v_last"], keep_names, res

def _pick_to_fix_by_name(svd_res, param_names, priority, n_fix=2, thr_priority=0.3, thr_other=0.4):
    contrib=svd_res["contrib"]; to_fix=[]
    for p in priority:
        if p in contrib and float(contrib[p])>thr_priority:
            to_fix.append(p)
    for p,v in contrib.items():
        if p in to_fix: continue
        if float(v)>thr_other:
            to_fix.append(p)
        if len(to_fix)>=n_fix:
            break
    return to_fix[:n_fix]

def select_final_theta(J, param_names, verbose=True, sim_cfg=None):
    svd_res=analyze_svd(J,param_names,X_NAMES)
    cond=svd_res["cond"]
    print(f"\n[Stage1] cond_8={cond:.2f}")
    if cond<=100:
        return {"mode":"6","to_fix":[],"theta_final":None,"cond":cond,"svd":svd_res}
    priority=["V0_blood","HR_base","C_sys_art"]
    to_fix=_pick_to_fix_by_name(svd_res,param_names,priority,n_fix=2)
    print(f"[Stage1] to_fix initial (2) = {to_fix}")
    cond2,_,keep_names,svd2=_fix_and_cond(J,param_names,to_fix)
    print(f"[Stage1] cond_6={cond2:.2f}")
    if cond2<=100:
        return {"mode":"6","to_fix":to_fix,"theta_final":keep_names,"cond":cond2,"svd":svd2}
    # Fallback A: fix 3
    priority_a=["V0_blood","HR_base","C_sys_art","R_sys"]
    to_fix_a=_pick_to_fix_by_name(svd_res,param_names,priority_a,n_fix=3)
    cond3,_,keep_names_a,svd3=_fix_and_cond(J,param_names,to_fix_a)
    print(f"[Stage1] Fallback A to_fix={to_fix_a} cond_5={cond3:.2f}")
    if cond3<=100:
        return {"mode":"5","to_fix":to_fix_a,"theta_final":keep_names_a,"cond":cond3,"svd":svd3}
    # Fallback B: replace C_sys_art with k_inotropy
    print(f"[Stage1] Fallback B: recompute J with k_inotropy instead of C_sys_art")
    # build alt J: need recompute for alt params
    # For verbose version, compute full alt Jacobian
    theta0_alt=dict(THETA0_ALT); theta0_alt["R_sys"]=resolve_R_sys(theta0_alt)
    J_alt,_,_=compute_jacobian(theta0_alt,param_names=PARAM_NAMES_ALT,scales_override=PARAM_SCALES_ALT,sim_cfg=sim_cfg,verbose=False)
    # map: keep same to_fix but with k_inotropy
    # try fix 2 again on alt
    svd_alt=analyze_svd(J_alt,PARAM_NAMES_ALT,X_NAMES)
    to_fix_alt=_pick_to_fix_by_name(svd_alt,PARAM_NAMES_ALT,["V0_blood","HR_base","k_inotropy"],n_fix=2)
    cond_alt,_,keep_alt,svd_alt2=_fix_and_cond(J_alt,PARAM_NAMES_ALT,to_fix_alt)
    print(f"[Stage1] Fallback B cond={cond_alt:.2f} to_fix={to_fix_alt}")
    return {"mode":"6_alt","to_fix":to_fix_alt,"theta_final":keep_alt,"cond":cond_alt,"svd":svd_alt2,"J_alt":J_alt}

def main():
    t_start=datetime.now()
    print("="*70)
    print(f"[Stage1 VERBOSE] Старт: {t_start:%Y-%m-%d %H:%M:%S}")
    print("="*70)
    if load_physiology is None:
        raise RuntimeError("load_physiology не найден")
    cfg=load_physiology(); sim_cfg=cfg.get("simulation",{})
    theta0=dict(THETA0); theta0["R_sys"]=resolve_R_sys(theta0)
    print(f"[Stage1] R_sys resolved = {theta0['R_sys']:.3f}, K_VSD={K_VSD}")
    # Jacobian
    J,X0,theta0_resolved=compute_jacobian(theta0,rel_step=0.01,sim_cfg=sim_cfg,verbose=True)
    # Save J
    out_dir=ROOT / "results"
    out_dir.mkdir(parents=True,exist_ok=True)
    df_J=pd.DataFrame(J,index=X_NAMES,columns=PARAM_NAMES)
    df_J.to_csv(out_dir / "stage1_J_verbose.csv")
    print(f"[Stage1] J saved to {out_dir / 'stage1_J_verbose.csv'}")
    # SVD plot
    svd_res=analyze_svd(J,PARAM_NAMES,X_NAMES)
    plt.figure(); plt.semilogy(svd_res["S"],'o-'); plt.title(f"SVD cond={svd_res['cond']:.1f}"); plt.ylabel("Singular value"); plt.xlabel("Index")
    plt.savefig(out_dir / "stage1_SVD_verbose.png"); plt.close()
    selection=select_final_theta(J,PARAM_NAMES,verbose=True,sim_cfg=sim_cfg)
    # Report
    lines=[]; lines.append("# STAGE 1 VERBOSE REPORT\n\n")
    lines.append(f"## cond_8 = {svd_res['cond']:.3f}\n\n")
    lines.append("## J (relative)\n"); lines.append(df_J.to_string()+"\n\n")
    lines.append("## |Vt[-1]|\n")
    for k,v in svd_res["contrib"].items():
        flag=" ← FIX" if k in selection["to_fix"] else ""
        lines.append(f"- {k:18s} : {v:.4f}{flag}\n")
    lines.append(f"\n## Решение\n- Режим: {selection['mode']}\n- Зафиксированы: {selection['to_fix']}\n- cond_final={selection['cond']:.3f}\n")
    (out_dir / "STAGE1_VERBOSE_REPORT.md").write_text("".join(lines),encoding="utf-8")
    print(f"[Stage1] Отчёт: {out_dir / 'STAGE1_VERBOSE_REPORT.md'}")
    print(f"\n[Stage1] Финиш: {datetime.now():%Y-%m-%d %H:%M:%S} Длительность: {datetime.now()-t_start}")

def debug_base_point():
    t_start=datetime.now()
    print("="*70+f"\n[DEBUG VERBOSE] Старт: {t_start:%Y-%m-%d %H:%M:%S}\n"+"="*70)
    if load_physiology is None:
        raise RuntimeError("load_physiology не найден")
    cfg=load_physiology(); sim_cfg=cfg.get("simulation",{})
    theta=dict(THETA0); theta["R_sys"]=resolve_R_sys(theta)
    print(f"R_sys resolved={theta['R_sys']:.3f}")
    model=build_model(theta)
    y0=model.calibrate_initial_state(t_calib=600.0)
    print(f"y0 heart={y0[model.idx['heart']]} V_blood={y0[model.idx['blood']][0]:.0f}")
    t_end=1000.0; t_eval=np.linspace(0.0,t_end,8001)
    sol=model.simulate((0.0,t_end),t_eval=t_eval,y0=y0,method=str(sim_cfg.get("method","LSODA")),rtol=1e-4,atol=1e-5,max_step=0.1)
    print(f"[solver] nfev={sol.nfev} t={sol.t[-1]:.1f}")
    data=_collect_outputs(model,sol)
    for t_lo,t_hi in [(100,300),(250,350),(300,400)]:
        m=(data["t"]>=t_lo)&(data["t"]<=t_hi)
        if not np.any(m): continue
        print(f"[{t_lo}-{t_hi}] P_sa={data['P_sa'][m].mean():5.1f}±{data['P_sa'][m].std():4.1f} V_lv={data['V_lv'][m].mean():5.1f} HR={data['HR'][m].mean():4.1f} P_sv={data['P_sv'][m].mean():5.2f}")
    HR_mean=float(np.mean(data["HR"][data["t"]>250])); T=60.0/max(HR_mean,1.0); win=data["t"]>(data["t"][-1]-10.0*T)
    print(f"\nSteady 10c P_sa={data['P_sa'][win].mean():.1f} EDV_LV={np.max(data['V_lv'][win]):.0f} V_blood={data['V_blood'][-1]:.0f} P_sv={data['P_sv'][win].mean():.1f} Q_periph={data['Q_peripheral'][win].mean():.1f} R_eff={data['R_eff_peripheral'][win].mean():.2f}")
    _print_window_diagnostics(data,label="DEBUG VERBOSE FULL")
    print(f"\n[DEBUG] Финиш: {datetime.now():%Y-%m-%d %H:%M:%S} Длительность: {datetime.now()-t_start}")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="debug":
        debug_base_point()
    else:
        main()
