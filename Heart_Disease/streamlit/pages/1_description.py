# -------------------------------------------------------------------
# Страница «О модели» / “About”.
# Никаких set_page_config и page_link — навигация общая, из app.py.
# -------------------------------------------------------------------
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from i18n import t


st.title(t("about.title"))

st.markdown(t("about.tagline1"))
st.markdown(t("about.tagline2"))
st.markdown(t("about.overview_md"))

st.warning(t("about.disclaimer"))

st.divider()

st.markdown(t("about.ml_details_md"))

st.divider()

st.markdown(t("about.comparison_md"))

st.page_link("pages/3_hbs_overview.py",
             label=t("about.link_hbs"), icon="🫀")