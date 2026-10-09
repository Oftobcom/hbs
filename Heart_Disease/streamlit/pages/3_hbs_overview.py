# streamlit/pages/3_hbs_overview.py
"""
Страница «О HBS»: описание механистической модели.
"""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from i18n import t
from utils_hbs import artifacts_ready

st.title(t("hbs.title"))
st.markdown(t("hbs.intro_md"))
st.markdown(t("hbs.modules_md"))
st.markdown(t("hbs.scenarios_md"))
st.markdown(t("hbs.howto_md"))
st.markdown(t("hbs.tech_md"))

# --- Проверка артефактов ---
ok, missing = artifacts_ready()
if ok:
    st.success(t("hbs.artifacts_ok"))
else:
    st.error(t("hbs.artifacts_miss"))
    for m in missing:
        st.code(m)          # пути и имена файлов — языконезависимы
    st.info(t("hbs.artifacts_hint"))