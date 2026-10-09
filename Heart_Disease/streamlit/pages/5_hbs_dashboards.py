# streamlit/pages/5_hbs_dashboards.py
"""
Страница «Дашборды HBS» / “HBS Dashboards”: просмотр готовых PNG.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from i18n import t
from utils_hbs import get_dashboards, HBS_PLOTS_DIR

st.title(t("dash.title"))
st.caption(t("dash.caption"))

if not HBS_PLOTS_DIR.exists():
    st.error(t("dash.dir_missing", path=HBS_PLOTS_DIR))
    st.stop()


@st.cache_data(show_spinner=False)
def load_png(path_str: str) -> bytes:
    return Path(path_str).read_bytes()


# --- Сгруппировать по группам ---
groups: dict[str, list] = {}
for fname, title, descr, group in get_dashboards():
    groups.setdefault(group, []).append((fname, title, descr))

# --- Селекторы ---
col_g, col_d = st.columns([1, 2])
with col_g:
    group = st.radio(t("dash.section"), list(groups.keys()))
with col_d:
    items = groups[group]
    titles = [it[1] for it in items]
    choice_title = st.radio(
        t("dash.dashboard"), titles,
        label_visibility="collapsed",
    )

fname, title, descr = next(it for it in items if it[1] == choice_title)
path = HBS_PLOTS_DIR / fname

if not path.exists():
    st.error(t("dash.file_missing", path=path))
    st.stop()

st.subheader(title)
if descr:
    st.caption(descr)

img_bytes = load_png(str(path))
st.image(img_bytes, use_container_width=True)

st.download_button(
    t("dash.download"),
    data=img_bytes,
    file_name=fname,
    mime="image/png",
)

st.divider()
st.page_link("pages/4_hbs_scenarios.py",
             label=t("dash.link_scen"), icon="📊")
st.page_link("pages/2_predict.py",
             label=t("dash.link_predict"), icon="🩺")