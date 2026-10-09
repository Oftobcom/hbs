# -------------------------------------------------------------------
# Страница «Прогноз»: форма + вызов модели.
# set_page_config здесь не вызывается — он только в app.py.
# -------------------------------------------------------------------
import __main__
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# i18n лежит в streamlit/, страница — в streamlit/pages/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from i18n import t

# -------------------------------------------------------------------
# Страховка: если app.py по какой-то причине не определил __main__.to_string
# (например, страница запускается в изоляции), добавим функцию в __main__,
# чтобы joblib.load корректно распаковал FunctionTransformer(to_string).
# -------------------------------------------------------------------
def _to_string(X):
    """Приводит категориальные признаки к строке."""
    return X.astype(str)


if not hasattr(__main__, "to_string"):
    __main__.to_string = _to_string


# -------------------------------------------------------------------
# Путь к модели: ../models/xgb_heart_accuracy.pkl относительно app.py
# (файл pages/2_predict.py → parents[2] == streamlit/ → parent → Heart_Disease/)
# -------------------------------------------------------------------
PAGES_DIR  = Path(__file__).resolve().parent          # ...\streamlit\pages
BASE_DIR   = PAGES_DIR.parent                          # ...\streamlit
MODEL_PATH = BASE_DIR.parent / "models" / "xgb_heart_accuracy.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(t("predict.model_missing", path=MODEL_PATH))
        st.stop()
    return joblib.load(MODEL_PATH)


model = load_model()

st.title(t("predict.title"))
st.caption(t("predict.caption"))
st.write(t("predict.subtitle"))

# -------------------------------------------------------------------
# Дисклеймер
# -------------------------------------------------------------------
st.warning(t("predict.disclaimer_md"))

# -------------------------------------------------------------------
# Форма ввода
# -------------------------------------------------------------------
with st.form("heart_form"):
    st.subheader(t("predict.form_header"))

    age    = st.number_input(f"age — {t('field.age')}", min_value=0, max_value=120, value=50)
    gender = st.selectbox(f"gender — {t('field.gender')}", ["Male", "Female"])

    cp = st.selectbox(
        f"cp — {t('field.cp')}",
        ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    )

    trestbps = st.number_input(f"trestbps — {t('field.trestbps')}", min_value=0, max_value=250, value=120)
    chol     = st.number_input(f"chol — {t('field.chol')}", min_value=0, max_value=700, value=200)

    fbs     = st.selectbox(f"fbs — {t('field.fbs')}", ["TRUE", "FALSE"])
    restecg = st.selectbox(
        f"restecg — {t('field.restecg')}",
        ["normal", "st-t abnormality", "lv hypertrophy"],
    )

    thalch = st.number_input(f"thalch — {t('field.thalch')}", min_value=0, max_value=250, value=150)

    exang   = st.selectbox(f"exang — {t('field.exang')}", ["TRUE", "FALSE"])
    oldpeak = st.number_input(f"oldpeak — {t('field.oldpeak')}", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

    slope = st.selectbox(f"slope — {t('field.slope')}", ["upsloping", "flat", "downsloping"])
    ca    = st.number_input(f"ca — {t('field.ca')}", min_value=0, max_value=3, value=0)

    thal = st.selectbox(
        f"thal — {t('field.thal')}",
        ["normal", "fixed defect", "reversable defect", "missing"],
    )

    submitted = st.form_submit_button(t("predict.submit"))

# -------------------------------------------------------------------
# Предсказание и вывод
# -------------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    pred    = model.predict(input_df)[0]
    proba   = model.predict_proba(input_df)[0]
    classes = model.classes_

    st.success(t("predict.success", cls=pred))

    result_df = (
        pd.DataFrame({"class": classes, "probability": proba})
        .set_index("class")
    )

    st.write(t("predict.probas"))
    st.bar_chart(result_df)
    st.dataframe(result_df.style.format("{:.4f}"))
    
    # --- Перекрёстная навигация в HBS ---
    st.divider()
    st.subheader(t("predict.cross_header"))
    st.markdown(t("predict.cross_body"))
    col_a, col_b = st.columns(2)
    with col_a:
        st.page_link("pages/4_hbs_scenarios.py",
                     label=t("predict.link_scen"), icon="📊")
    with col_b:
        st.page_link("pages/5_hbs_dashboards.py",
                     label=t("predict.link_dash"), icon="🖼️")