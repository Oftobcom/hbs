# streamlit/pages/4_hbs_scenarios.py
"""
Страница «Сценарии ДМЖП» / “VSD Scenarios”: живые графики из .npz.
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from i18n import t
from utils_hbs import (
    load_all_scenarios, qp_qs_steady, steady_mean, scenario_display_name,
)

st.title(t("scen.title"))
st.caption(t("scen.caption"))
st.caption(t("scen.caption_npz"))

scenarios = load_all_scenarios()
if not scenarios:
    st.error(t("scen.not_found"))
    st.stop()

label = st.selectbox(
    t("scen.selector"),
    list(scenarios.keys()),      # стабильные оригинальные ключи
    format_func=scenario_display_name,
    key="scen_selector",
)
data = scenarios[label]

# --- Метрики сверху ---
col1, col2, col3, col4 = st.columns(4)
qp_qs = qp_qs_steady(data)
sao2  = steady_mean(data, "SaO2", scale=100.0)
p_pa  = steady_mean(data, "P_pa")
p_sa  = steady_mean(data, "P_sa")

col1.metric("Qp/Qs", f"{qp_qs:.2f}" if qp_qs is not None else "N/A")
col2.metric("SaO₂", f"{sao2:.1f}{t('scen.metric_sao2')}" if sao2 is not None else "N/A")
col3.metric("P_pa",  f"{p_pa:.1f} {t('scen.metric_ppa')}" if p_pa is not None else "N/A")
col4.metric("P_sa",  f"{p_sa:.1f} {t('scen.metric_psa')}" if p_sa is not None else "N/A")

st.divider()

# --- Табы с группами графиков ---
tab1, tab2, tab3 = st.tabs([
    t("scen.tab_press"), t("scen.tab_flow"), t("scen.tab_vol"),
])

with tab1:
    st.subheader(t("scen.subheader_press"))
    st.line_chart(
        {"P_sa": data["P_sa"], "P_pa": data["P_pa"]},
        x_label=t("scen.time_axis"),
        y_label=t("scen.metric_ppa"),
    )
    st.caption(t("scen.caption_press"))

with tab2:
    st.subheader(t("scen.subheader_flow"))
    st.line_chart(
        {"Q_aortic": data["Q_aortic"], "Q_pulmonary": data["Q_pulmonary"]},
        x_label=t("scen.time_axis"),
        y_label=t("scen.unit_ml_per_s"),
    )
    st.subheader(t("scen.subheader_qpqs"))
    qp = np.asarray(data["Q_pulmonary"], dtype=float)
    qa = np.asarray(data["Q_aortic"], dtype=float)
    win = max(20, len(qp) // 15)
    kernel = np.ones(win) / win
    qp_s = np.convolve(qp, kernel, mode="same")
    qa_s = np.convolve(qa, kernel, mode="same")
    ratio = qp_s / np.maximum(qa_s, 1e-6)
    st.line_chart({"Qp/Qs": ratio}, x_label=t("scen.time_axis"))

with tab3:
    st.subheader(t("scen.subheader_vol"))
    st.line_chart(
        {"V_lv": data["V_lv"], "V_rv": data["V_rv"]},
        x_label=t("scen.time_axis"),
        y_label=t("scen.unit_ml"),
    )
    st.subheader(t("scen.subheader_sao2"))
    st.line_chart(
        {"SaO2": np.asarray(data["SaO2"]) * 100.0},
        x_label=t("scen.time_axis"),
        y_label=t("scen.metric_sao2"),
    )
    st.caption(t("scen.caption_sao2"))

st.divider()
st.page_link("pages/5_hbs_dashboards.py",
             label=t("scen.link_dash"), icon="🖼️")