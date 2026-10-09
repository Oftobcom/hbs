# -------------------------------------------------------------------
# Точка входа приложения.
# -------------------------------------------------------------------
import streamlit as st
from i18n import t, language_selector


def to_string(X):
    """Приводит категориальные признаки к строке."""
    return X.astype(str)


st.set_page_config(
    page_title="Jism HBS — Heart & Body Simulation",
    page_icon="🫀",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
        height: 0rem;
        background: transparent;
    }
    .stAppDeployButton { display: none; }
    [data-testid="stStatusWidget"] { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 8px 0 16px 0;">
        <div style="font-size: 22px; font-weight: 700;">🧬 Jism HBS</div>
        <div style="font-size: 11px; color: #888;">Human Body Simulation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Переключатель языка (ОБЯЗАТЕЛЬНО до st.navigation) ---
language_selector()

# --- Навигация (локализованные заголовки) ---
pg = st.navigation({
    t("nav.hbs_group"): [
        st.Page("pages/4_hbs_scenarios.py",  title=t("nav.hbs_scenarios"),  icon="📊"),
        st.Page("pages/5_hbs_dashboards.py", title=t("nav.hbs_dashboards"), icon="🖼️"),
        st.Page("pages/3_hbs_overview.py",   title=t("nav.hbs_about"),      icon="🫀"),
    ],
    t("nav.ml_group"): [
        st.Page("pages/2_predict.py",     title=t("nav.predict"),  icon="🩺"),
        st.Page("pages/1_description.py", title=t("nav.about_ml"), icon="📄", default=True),
    ],
})

pg.run()