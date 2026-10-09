# streamlit/i18n.py
"""
Двуязычность Jism HBS. Ключи — плоские строки вида "predict.title".
"""
import streamlit as st

DEFAULT_LANG = "ru"
SUPPORTED_LANGS = ("ru", "en", "tj")

TRANSLATIONS: dict[str, dict[str, str]] = {
    "ru": {
        # --- Навигация (app.py) ---
        "nav.ml_group":       "Jism HBS · ML-скрининг",
        "nav.hbs_group":      "Jism HBS · Механистическая модель",
        "nav.predict":        "Прогноз",
        "nav.about_ml":       "О ML-модели",
        "nav.hbs_scenarios":  "Сценарии ДМЖП",
        "nav.hbs_dashboards": "Дашборды",
        "nav.hbs_about":      "О HBS",
        "lang.label":         "Язык / Language",

        # --- 1_description.py ---
        "about.title":        "🫀 О проекте Jism HBS",
        "about.disclaimer":   "⚠️ **Важно:** это **не медицинский диагноз**. Проект является исследовательским и не заменяет консультацию врача.",
        "about.link_hbs":     "→ О HBS",
        "about.tagline1":     "### **Jism** от тадж. *jism* — «тело»",
        "about.tagline2":     "### **HBS** — Human Body Simulation — механистическая модель организма",
        "about.overview_md": (
            "**Jism HBS** — исследовательский проект, объединяющий два подхода "
            "к сердечно-сосудистой патологии:\n\n"
            "- **HBS** — механистическая ODE-модель организма с динамикой во времени.\n"
            "- **ML-модель** — data-driven скрининг на основе 13 клинических признаков."
        ),
        "about.ml_details_md": (
            "### О модели\n"
            "- **Алгоритм**: XGBoost (многоклассовая классификация, 5 классов: 0–4)\n"
            "- **Данные**: Heart Disease UCI (920 записей после очистки)\n"
            "- **Признаки**: age, gender, cp, trestbps, chol, fbs, restecg, thalch,\n"
            "  exang, oldpeak, slope, ca, thal\n"
            "- **Предобработка**:\n"
            "  - числовые признаки — без изменений;\n"
            "  - категориальные — заполнение пропусков значением `'missing'`,\n"
            "    приведение к строке, OneHotEncoder.\n"
            "- **Гиперпараметры лучшей модели** (подобраны RandomizedSearchCV по accuracy):\n"
            "  - `n_estimators=500`, `max_depth=4`, `learning_rate=0.03`,\n"
            "    `subsample=0.6`, `colsample_bytree=0.7`, `gamma=0`,\n"
            "    `reg_alpha=0.1`, `reg_lambda=1.5`, `min_child_weight=3`.\n"
            "- **Метрики на тестовой выборке**:\n"
            "  - Accuracy ≈ **0.625**\n"
            "  - Macro F1 ≈ **0.418**\n"
            "- **Особенности**: класс `4` почти не предсказывается (F1 = 0).\n"
            "- **Файл модели**: `models/xgb_heart_accuracy.pkl`"
        ),
        "about.comparison_md": (
            "| Подход | Инструмент | Отвечает на вопрос |\n"
            "|---|---|---|\n"
            "| Mechanistic | Human Body Simulation | *Почему именно так и что будет дальше?* |\n"
            "| Data-driven | XGBoost (этот раздел) | *Есть ли риск и какой?* |"
        ),        

        "predict.title":      "Прогноз сердечных заболеваний",
        "predict.subtitle":   "Многоклассовая модель XGBoost: классы 0–4",
        "predict.form_header":"Введите признаки пациента",
        "predict.submit":     "Предсказать",
        "predict.success":    "Предсказанный класс: {cls}",
        "predict.probas":     "Вероятности по классам:",
        "predict.link_scen":  "📊 Сценарии ДМЖП",
        "predict.link_dash":  "🖼️ Дашборды HBS",
        # поля формы
        "field.age":          "возраст",
        "field.gender":       "пол",
        "field.cp":           "тип боли в груди",
        # --- 2_predict.py (продолжение) ---
        "predict.caption":      "Jism HBS · ML-скрининг",
        "predict.model_missing":"Файл модели не найден: {path}",
        "predict.disclaimer_md": (
            "**Дисклеймер**\n"
            "- Модель обучена на **920 записях**.\n"
            "- Класс `4` почти не предсказывается.\n"
            "- Accuracy ≈ **0.625**, Macro F1 ≈ **0.418**.\n"
            "- Это **не медицинский диагноз**."
        ),
        "predict.cross_header": "🔬 Механистическая интерпретация",
        "predict.cross_body": (
            "ML-модель даёт **статичную оценку риска**. Чтобы увидеть, как этот риск "
            "разворачивается в динамике (давления, Qp/Qs, SaO₂, ремоделирование), "
            "перейдите в раздел HBS."
        ),
        "field.trestbps":       "давление в покое",
        "field.chol":           "холестерин",
        "field.fbs":            "сахар натощак > 120",
        "field.restecg":        "ЭКГ в покое",
        "field.thalch":         "максимальный пульс",
        "field.exang":          "стенокардия при нагрузке",
        "field.oldpeak":        "депрессия ST",
        "field.slope":          "наклон ST",
        "field.ca":             "число крупных сосудов",
        "field.thal":           "талассемия",

        # --- 3_hbs_overview.py ---
        "hbs.title":          "🫀 Jism HBS · механистическая модель организма",
        "hbs.intro_md": (
            "### **HBS** — Human Body Simulation\n\n"
            "HBS — механистическая модель организма: 12 органов, связанных "
            "дифференциальными уравнениями. В отличие от ML-модели, которая даёт "
            "статичную оценку риска, HBS показывает **динамику** во времени."
        ),
        "hbs.modules_md": (
            "### Органы и подсистемы\n\n"
            "| Модуль | Что моделирует |\n"
            "|---|---|\n"
            "| `heart.py` | 4-камерное сердце с клапанами и ДМЖП |\n"
            "| `lungs.py` | 2-компартментные лёгкие + рекруитмент + ремоделирование |\n"
            "| `baroreflex.py` | Барорефлекс: ЧСС, инотропия, вазомоторика |\n"
            "| `brain.py` | Мозг: ауторегуляция, O₂/CO₂, лактат, аммиак |\n"
            "| `liver.py` | Печень: портальная гемодинамика, метаболизм |\n"
            "| `kidney.py` | Почки: СКФ, диурез, клиренс токсина |\n"
            "| `gitract.py` | ЖКТ: всасывание, портальный кровоток |\n"
            "| `peripheral_tissues.py` | Периферия: ауторегуляция, VO₂, лактат |\n"
            "| `jugular_vein.py` | Яремная вена: буфер мозгового оттока |\n"
            "| `blood.py` | Кровь: единый резервуар концентраций |\n"
            "| `gas_exchange.py` | Альвеолярно-капиллярный обмен O₂/CO₂ |\n"
            "| `whole_body.py` | Сборка всех органов + Windkessel-сосуды |"
        ),
        "hbs.scenarios_md": (
            "### Сценарии ДМЖП\n\n"
            "Модель воспроизводит 5 клинических фенотипов:\n\n"
            "1. **Здоровый** — нет шунта, Qp/Qs ≈ 1.0\n"
            "2. **Малый ДМЖП (R=5.0)** — незначительный L→R шунт\n"
            "3. **Большой ДМЖП (R=1.0)** — выраженный L→R шунт, Qp/Qs > 1.4\n"
            "4. **Эйзенменгер, компенсированный** — PVR ≈ SVR, ПЖ гипертрофирован\n"
            "5. **Эйзенменгер, декомпенсированный** — PVR > SVR, R→L шунт, цианоз"
        ),
        "hbs.howto_md": (
            "### Как читать дашборды\n\n"
            "- **PV-портреты** — форма петли «объём–давление» показывает работу желудочка\n"
            "- **Qp/Qs** — отношение лёгочного и системного кровотока (норма ≈ 1.0)\n"
            "- **SaO₂** — сатурация артериальной крови; < 90% = гипоксемия\n"
            "- **R→L шунт** — фракция венозной крови, идущей напрямую в аорту"
        ),
        "hbs.tech_md": (
            "### Технические детали\n\n"
            "- Интегратор: **LSODA** (scipy), rtol=1e-4, atol=1e-5\n"
            "- Время симуляции: **800 с** (калибровка 400–600 с + рабочий прогон)\n"
            "- 5 сценариев считаются параллельно через **joblib/loky**\n"
            "- Результаты сохранены в `.npz`, дашборды — в `.png`"
        ),
        "hbs.artifacts_ok":   "✅ Все артефакты HBS на месте (5 сценариев + 9 дашбордов).",
        "hbs.artifacts_miss": "❌ Не найдены артефакты HBS:",
        "hbs.artifacts_hint": (
            "Запустите `run_simulation_parallel.py` и `visualize_vsd_comparison.py` "
            "в проекте HBS, затем скопируйте `.npz` в `hbs_results/`, "
            "а PNG — в `hbs_results/plots/`."
        ),

        # --- 4_hbs_scenarios.py ---
        "scen.title":         "📊 Сценарии ДМЖП — механистическая динамика",
        "scen.selector":      "Сценарий",
        "scen.tab_press":     "Давления",
        "scen.tab_flow":      "Потоки",
        "scen.tab_vol":       "Объёмы и газы",
        "scen.caption":           "Jism HBS · живые данные из .npz",
        "scen.caption_npz":       "Данные из `.npz`, сгенерированных HBS (`run_simulation_parallel.py`).",
        "scen.not_found":         "Не найдено `.npz` в `hbs_results/`. Сначала запустите HBS-симуляцию (см. страницу «О HBS»).",
        "scen.subheader_press":   "Системное и лёгочное давление",
        "scen.subheader_flow":    "Системный и лёгочный кровоток",
        "scen.subheader_qpqs":    "Qp/Qs (скользящее среднее)",
        "scen.subheader_vol":     "Объёмы желудочков",
        "scen.subheader_sao2":    "Сатурация артериальной крови",
        "scen.caption_press":     "Норма: P_sa ≈ 90–110, P_pa ≈ 15–25 мм рт. ст.",
        "scen.caption_sao2":      "Норма: SaO₂ ≥ 95%. При R→L шунте падает ниже 90%.",
        "scen.link_dash":         "→ Готовые дашборды HBS",
        "scen.metric_ppa":        "мм рт. ст.",
        "scen.metric_psa":        "мм рт. ст.",
        "scen.metric_sao2":       "%",
        "scen.time_axis":         "Время (с)",
        "scen.unit_ml_per_s":     "мл/с",
        "scen.unit_ml":           "мл",

        # --- Имена сценариев ---
        "scenario.healthy":                "Здоровый",
        "scenario.vsd_small":              "Малый ДМЖП (R=5.0)",
        "scenario.vsd_large":              "Большой ДМЖП (R=1.0)",
        "scenario.eisenmenger_comp":       "Эйзенменгер, компенсированный",
        "scenario.eisenmenger_decomp":     "Эйзенменгер, декомпенсированный",

        # --- 5_hbs_dashboards.py ---
        "dash.title":         "🖼️ Дашборды Jism HBS",
        "dash.caption":       "Готовые визуализации механистической модели (5 сценариев ДМЖП).",
        "dash.section":       "Раздел",
        "dash.dashboard":     "Дашборд",
        "dash.download":      "⬇️ Скачать PNG",
        "dash.dir_missing":       "Папка не найдена: `{path}`. Скопируйте PNG из HBS-проекта.",
        "dash.file_missing":      "Файл не найден: `{path}`",
        "dash.link_scen":         "← Живые графики из .npz",
        "dash.link_predict":      "← ML-прогноз",

        # --- Заголовки и описания дашбордов ---
        "dash.comprehensive.title": "Комплексный дашборд",
        "dash.comprehensive.descr": "Сводная панель 4×4: Qp/Qs, давления, объёмы, региональные кровотоки, функция почек, газы крови и итоговая таблица.",
        "dash.hemo_cmp.title":      "Сравнение гемодинамики 3×3",
        "dash.hemo_cmp.descr":      "P_sa/P_pa, Q_aortic/Q_pulmonary, Qp/Qs, шунт через ДМЖП, объёмы желудочков, волемия, Q_brain, SaO₂ и сводная таблица.",
        "dash.pv_loops.title":      "Фазовые PV-портреты",
        "dash.pv_loops.descr":      "Петли «объём–давление» для ЛЖ и ПЖ за последние ~2 кардиоцикла. Показывают, как ремоделирование меняет работу желудочков.",
        "dash.cardiac_detail.title":"Детальный анализ сердца",
        "dash.cardiac_detail.descr":"Qp/Qs во времени, объёмы ЛЖ/ПЖ, корреляция шунта и Qp/Qs, региональное распределение кровотока.",
        "dash.gas.title":           "Газообмен O₂ / CO₂",
        "dash.gas.descr":           "Концентрации и парциальные давления O₂/CO₂, компоненты VO₂, баланс CO₂ по сценариям.",
        "dash.timeseries.title":    "Временные ряды",
        "dash.timeseries.descr":    "9 панелей: давления, потоки, Qp/Qs, шунт, объёмы ЛЖ/ПЖ, SaO₂.",
        "dash.steady.title":        "Установившиеся показатели",
        "dash.steady.descr":        "P_sa, P_pa, Q_aortic, V_rv, V_blood, GFR, SaO₂, Qp/Qs.",
        "dash.shunt.title":         "Анализ влияния размера ДМЖП",
        "dash.shunt.descr":         "Динамика Qp/Qs, давления, корреляция шунт↔Qp/Qs, R→L фракция, радар нормированных показателей.",
        "dash.schematic.title":     "Схема сердца: норма vs Эйзенменгер",
        "dash.schematic.descr":     "Направление шунта и геометрия камер: здоровый vs декомпенсированный.",

        # --- Служебные ---
        "hbs.load_error":         "Не удалось загрузить {name}: {err}",


        # --- Группы дашбордов (utils_hbs) ---
        "group.summary":      "📋 Итоговые сводки",
        "group.heart_lungs":  "🫀 Сердце и лёгкие",
        "group.gas":          "💨 Газообмен",
        "group.hemo":         "🩸 Гемодинамика",
        "group.schemas":      "🎨 Схемы",
    },

    "en": {
        # --- Navigation ---
        "nav.ml_group":       "Jism HBS · ML Screening",
        "nav.hbs_group":      "Jism HBS · Mechanistic Model",
        "nav.predict":        "Prediction",
        "nav.about_ml":       "About ML Model",
        "nav.hbs_scenarios":  "VSD Scenarios",
        "nav.hbs_dashboards": "Dashboards",
        "nav.hbs_about":      "About HBS",
        "lang.label":         "Language / Язык",

        # --- About ---
        "about.title":        "🫀 About Jism HBS",
        "about.disclaimer":   "⚠️ **Important:** this is **not a medical diagnosis**. The project is for research only and does not replace a doctor's consultation.",
        "about.link_hbs":     "→ About HBS",
        "about.tagline1":     "### **Jism** from Tajiki *jism* — “body”",
        "about.tagline2":     "### **HBS** — Human Body Simulation — a mechanistic whole-body model",
        "about.overview_md": (
            "**Jism HBS** is a research project combining two approaches "
            "to cardiovascular pathology:\n\n"
            "- **HBS** — a mechanistic ODE whole-body model with time dynamics.\n"
            "- **ML model** — data-driven screening based on 13 clinical features."
        ),
        "about.ml_details_md": (
            "### About the model\n"
            "- **Algorithm**: XGBoost (multi-class classification, 5 classes: 0–4)\n"
            "- **Data**: Heart Disease UCI (920 records after cleaning)\n"
            "- **Features**: age, gender, cp, trestbps, chol, fbs, restecg, thalch,\n"
            "  exang, oldpeak, slope, ca, thal\n"
            "- **Preprocessing**:\n"
            "  - numeric features — unchanged;\n"
            "  - categorical — impute missing as `'missing'`,\n"
            "    cast to string, OneHotEncoder.\n"
            "- **Hyperparameters of the best model** (tuned via RandomizedSearchCV by accuracy):\n"
            "  - `n_estimators=500`, `max_depth=4`, `learning_rate=0.03`,\n"
            "    `subsample=0.6`, `colsample_bytree=0.7`, `gamma=0`,\n"
            "    `reg_alpha=0.1`, `reg_lambda=1.5`, `min_child_weight=3`.\n"
            "- **Test-set metrics**:\n"
            "  - Accuracy ≈ **0.625**\n"
            "  - Macro F1 ≈ **0.418**\n"
            "- **Notes**: class `4` is almost never predicted (F1 = 0).\n"
            "- **Model file**: `models/xgb_heart_accuracy.pkl`"
        ),
        "about.comparison_md": (
            "| Approach | Tool | Question it answers |\n"
            "|---|---|---|\n"
            "| Mechanistic | Human Body Simulation | *Why is it like this, and what happens next?* |\n"
            "| Data-driven | XGBoost (this section) | *Is there a risk, and what kind?* |"
        ),
        
        # --- Predict ---
        "predict.title":      "Heart Disease Prediction",
        "predict.subtitle":   "Multi-class XGBoost model: classes 0–4",
        "predict.form_header":"Enter patient features",
        "predict.submit":     "Predict",
        "predict.success":    "Predicted class: {cls}",
        "predict.probas":     "Class probabilities:",
        "predict.link_scen":  "📊 VSD Scenarios",
        "predict.link_dash":  "🖼️ HBS Dashboards",
        "field.age":          "age",
        "field.gender":       "gender",
        "field.cp":           "chest pain type",
        # --- 2_predict.py (продолжение) ---
        "predict.caption":      "Jism HBS · ML Screening",
        "predict.model_missing":"Model file not found: {path}",
        "predict.disclaimer_md": (
            "**Disclaimer**\n"
            "- The model was trained on **920 records**.\n"
            "- Class `4` is almost never predicted.\n"
            "- Accuracy ≈ **0.625**, Macro F1 ≈ **0.418**.\n"
            "- This is **not a medical diagnosis**."
        ),
        "predict.cross_header": "🔬 Mechanistic interpretation",
        "predict.cross_body": (
            "The ML model gives a **static risk estimate**. To see how this risk "
            "unfolds over time (pressures, Qp/Qs, SaO₂, remodeling), "
            "switch to the HBS section."
        ),
        "field.trestbps":       "resting blood pressure",
        "field.chol":           "cholesterol",
        "field.fbs":            "fasting blood sugar > 120",
        "field.restecg":        "resting ECG",
        "field.thalch":         "max heart rate",
        "field.exang":          "exercise-induced angina",
        "field.oldpeak":        "ST depression",
        "field.slope":          "ST slope",
        "field.ca":             "number of major vessels",
        "field.thal":           "thalassemia",

        # --- HBS overview ---
        "hbs.title":          "🫀 Jism HBS · mechanistic whole-body model",
        "hbs.intro_md": (
            "### **HBS** — Human Body Simulation\n\n"
            "HBS is a mechanistic whole-body model: 12 organs linked by "
            "differential equations. Unlike the ML model, which gives a "
            "static risk estimate, HBS shows **dynamics** over time."
        ),
        "hbs.modules_md": (
            "### Organs and subsystems\n\n"
            "| Module | What it models |\n"
            "|---|---|\n"
            "| `heart.py` | 4-chamber heart with valves and VSD |\n"
            "| `lungs.py` | 2-compartment lungs + recruitment + remodeling |\n"
            "| `baroreflex.py` | Baroreflex: HR, inotropy, vasomotion |\n"
            "| `brain.py` | Brain: autoregulation, O₂/CO₂, lactate, ammonia |\n"
            "| `liver.py` | Liver: portal hemodynamics, metabolism |\n"
            "| `kidney.py` | Kidneys: GFR, diuresis, toxin clearance |\n"
            "| `gitract.py` | GI tract: absorption, portal blood flow |\n"
            "| `peripheral_tissues.py` | Periphery: autoregulation, VO₂, lactate |\n"
            "| `jugular_vein.py` | Jugular vein: cerebral outflow buffer |\n"
            "| `blood.py` | Blood: single concentration reservoir |\n"
            "| `gas_exchange.py` | Alveolar-capillary O₂/CO₂ exchange |\n"
            "| `whole_body.py` | Assembly of all organs + Windkessel vessels |"
        ),
        "hbs.scenarios_md": (
            "### VSD scenarios\n\n"
            "The model reproduces 5 clinical phenotypes:\n\n"
            "1. **Healthy** — no shunt, Qp/Qs ≈ 1.0\n"
            "2. **Small VSD (R=5.0)** — minor L→R shunt\n"
            "3. **Large VSD (R=1.0)** — pronounced L→R shunt, Qp/Qs > 1.4\n"
            "4. **Eisenmenger, compensated** — PVR ≈ SVR, RV hypertrophied\n"
            "5. **Eisenmenger, decompensated** — PVR > SVR, R→L shunt, cyanosis"
        ),
        "hbs.howto_md": (
            "### How to read the dashboards\n\n"
            "- **PV loops** — the shape of the pressure–volume loop shows ventricular work\n"
            "- **Qp/Qs** — pulmonary-to-systemic flow ratio (normal ≈ 1.0)\n"
            "- **SaO₂** — arterial oxygen saturation; < 90% = hypoxemia\n"
            "- **R→L shunt** — fraction of venous blood going directly to the aorta"
        ),
        "hbs.tech_md": (
            "### Technical details\n\n"
            "- Integrator: **LSODA** (scipy), rtol=1e-4, atol=1e-5\n"
            "- Simulation time: **800 s** (calibration 400–600 s + working run)\n"
            "- 5 scenarios computed in parallel via **joblib/loky**\n"
            "- Results saved to `.npz`, dashboards to `.png`"
        ),
        "hbs.artifacts_hint": (
            "Run `run_simulation_parallel.py` and `visualize_vsd_comparison.py` "
            "in the HBS project, then copy `.npz` to `hbs_results/`, "
            "and PNGs to `hbs_results/plots/`."
        ),        
        "hbs.artifacts_ok":   "✅ All HBS artifacts present (5 scenarios + 9 dashboards).",
        "hbs.artifacts_miss": "❌ Missing HBS artifacts:",

        # --- Scenarios ---
        "scen.title":         "📊 VSD Scenarios — mechanistic dynamics",
        "scen.selector":      "Scenario",
        "scen.tab_press":     "Pressures",
        "scen.tab_flow":      "Flows",
        "scen.tab_vol":       "Volumes & gases",
        "scen.caption":           "Jism HBS · live data from .npz",
        "scen.caption_npz":       "Data from `.npz` generated by HBS (`run_simulation_parallel.py`).",
        "scen.not_found":         "No `.npz` found in `hbs_results/`. Run the HBS simulation first (see the “About HBS” page).",
        "scen.subheader_press":   "Systemic and pulmonary pressure",
        "scen.subheader_flow":    "Systemic and pulmonary flow",
        "scen.subheader_qpqs":    "Qp/Qs (rolling mean)",
        "scen.subheader_vol":     "Ventricular volumes",
        "scen.subheader_sao2":    "Arterial oxygen saturation",
        "scen.caption_press":     "Normal: P_sa ≈ 90–110, P_pa ≈ 15–25 mmHg.",
        "scen.caption_sao2":      "Normal: SaO₂ ≥ 95%. With R→L shunt it drops below 90%.",
        "scen.link_dash":         "→ Pre-rendered HBS dashboards",
        "scen.metric_ppa":        "mmHg",
        "scen.metric_psa":        "mmHg",
        "scen.metric_sao2":       "%",
        "scen.time_axis":         "Time (s)",
        "scen.unit_ml_per_s":     "mL/s",
        "scen.unit_ml":           "mL",

        # --- Scenario names ---
        "scenario.healthy":                "Healthy",
        "scenario.vsd_small":              "Small VSD (R=5.0)",
        "scenario.vsd_large":              "Large VSD (R=1.0)",
        "scenario.eisenmenger_comp":       "Eisenmenger, compensated",
        "scenario.eisenmenger_decomp":     "Eisenmenger, decompensated",

        # --- Dashboards ---
        "dash.title":         "🖼️ Jism HBS Dashboards",
        "dash.caption":       "Pre-rendered visualizations of the mechanistic model (5 VSD scenarios).",
        "dash.section":       "Section",
        "dash.dashboard":     "Dashboard",
        "dash.download":      "⬇️ Download PNG",
        "dash.dir_missing":       "Directory not found: `{path}`. Copy PNGs from the HBS project.",
        "dash.file_missing":      "File not found: `{path}`",
        "dash.link_scen":         "← Live plots from .npz",
        "dash.link_predict":      "← ML prediction",

        # --- Dashboard titles & descriptions ---
        "dash.comprehensive.title": "Comprehensive dashboard",
        "dash.comprehensive.descr": "4×4 summary panel: Qp/Qs, pressures, volumes, regional flows, renal function, blood gases and a summary table.",
        "dash.hemo_cmp.title":      "Hemodynamics comparison 3×3",
        "dash.hemo_cmp.descr":      "P_sa/P_pa, Q_aortic/Q_pulmonary, Qp/Qs, VSD shunt, ventricular volumes, volemia, Q_brain, SaO₂ and a summary table.",
        "dash.pv_loops.title":      "Phase PV loops",
        "dash.pv_loops.descr":      "Pressure–volume loops for LV and RV over the last ~2 cardiac cycles. Show how remodeling changes ventricular work.",
        "dash.cardiac_detail.title":"Detailed cardiac analysis",
        "dash.cardiac_detail.descr":"Qp/Qs over time, LV/RV volumes, shunt↔Qp/Qs correlation, regional flow distribution.",
        "dash.gas.title":           "Gas exchange O₂ / CO₂",
        "dash.gas.descr":           "O₂/CO₂ concentrations and partial pressures, VO₂ components, CO₂ balance across scenarios.",
        "dash.timeseries.title":    "Time series",
        "dash.timeseries.descr":    "9 panels: pressures, flows, Qp/Qs, shunt, LV/RV volumes, SaO₂.",
        "dash.steady.title":        "Steady-state metrics",
        "dash.steady.descr":        "P_sa, P_pa, Q_aortic, V_rv, V_blood, GFR, SaO₂, Qp/Qs.",
        "dash.shunt.title":         "VSD size effect analysis",
        "dash.shunt.descr":         "Qp/Qs dynamics, pressures, shunt↔Qp/Qs correlation, R→L fraction, radar of normalized metrics.",
        "dash.schematic.title":     "Heart schematic: normal vs Eisenmenger",
        "dash.schematic.descr":     "Shunt direction and chamber geometry: healthy vs decompensated.",

        # --- Service ---
        "hbs.load_error":         "Failed to load {name}: {err}",

        # --- Dashboard groups ---
        "group.summary":      "📋 Summary",
        "group.heart_lungs":  "🫀 Heart & Lungs",
        "group.gas":          "💨 Gas Exchange",
        "group.hemo":         "🩸 Hemodynamics",
        "group.schemas":      "🎨 Schematics",
    },

    "tj": {
        # --- Навигация (app.py) ---
        "nav.ml_group":       "Jism HBS · Скрининги ML",
        "nav.hbs_group":      "Jism HBS · Модели механики",
        "nav.predict":        "Пешгӯӣ",
        "nav.about_ml":       "Дар бораи модели ML",
        "nav.hbs_scenarios":  "Сценарияҳои ДМЖП",
        "nav.hbs_dashboards": "Дашбордҳо",
        "nav.hbs_about":      "Дар бораи HBS",
        "lang.label":         "Забон / Language / Язык",

        # --- 1_description.py ---
        "about.title":        "🫀 Дар бораи лоиҳаи Jism HBS",
        "about.disclaimer":   "⚠️ **Муҳим:** ин **ташхиси тиббӣ нест**. Лоиҳа хусусияти таҳқиқотӣ дорад ва машварати духтурро иваз намекунад.",
        "about.link_hbs":     "→ Дар бораи HBS",
        "about.tagline1":     "### **Jism** аз тоҷ. *jism* — «бадан, тани инсон»",
        "about.tagline2":     "### **HBS** — Human Body Simulation — модели механикии организми инсон",
        "about.overview_md": (
            "**Jism HBS** — лоиҳаи таҳқиқотист, ки ду равишро ба патологияи дилу рагҳо муттаҳид мекунад:\n\n"
            "- **HBS** — модели механикии ODE-и организм бо динамика дар вақт.\n"
            "- **Модели ML** — скрининги data-driven дар асоси 13 нишонаи клиникӣ."
        ),
        "about.ml_details_md": (
            "### Дар бораи модел\n"
            "- **Алгоритм**: XGBoost (таснифоти бисёрсинфӣ, 5 синф: 0–4)\n"
            "- **Маълумот**: Heart Disease UCI (920 сабт пас аз поксозӣ)\n"
            "- **Нишонаҳо**: age, gender, cp, trestbps, chol, fbs, restecg, thalch,\n"
            "  exang, oldpeak, slope, ca, thal\n"
            "- **Пешпардоз (Preprocessing)**:\n"
            "  - нишонаҳои рақамӣ — бе тағйир;\n"
            "  - категориалӣ — пур кардани ҷойҳои холӣ бо қимати `'missing'`,\n"
            "    овардан ба сатр, OneHotEncoder.\n"
            "- **Гиперпараметрҳои модели беҳтарин** (тавассути RandomizedSearchCV аз рӯи accuracy интихоб шудаанд):\n"
            "  - `n_estimators=500`, `max_depth=4`, `learning_rate=0.03`,\n"
            "    `subsample=0.6`, `colsample_bytree=0.7`, `gamma=0`,\n"
            "    `reg_alpha=0.1`, `reg_lambda=1.5`, `min_child_weight=3`.\n"
            "- **Метрикаҳо дар интихоби тестӣ**:\n"
            "  - Accuracy ≈ **0.625**\n"
            "  - Macro F1 ≈ **0.418**\n"
            "- **Хусусиятҳо**: синфи `4` тақрибан пешгӯӣ карда намешавад (F1 = 0).\n"
            "- **Файли модел**: `models/xgb_heart_accuracy.pkl`"
        ),
        "about.comparison_md": (
            "| Равиш | Инструмент | Ба кадом савол ҷавоб медиҳад |\n"
            "|---|---|---|\n"
            "| Mechanistic | Human Body Simulation | *Чаро маҳз ҳамин тавр ва баъд чӣ мешавад?* |\n"
            "| Data-driven | XGBoost (ин бахш) | *Оё хавф ҳаст ва кадом хавф?* |"
        ),        

        "predict.title":      "Пешгӯии бемориҳои дил",
        "predict.subtitle":   "Модели бисёрсинфии XGBoost: синфҳои 0–4",
        "predict.form_header":"Нишондиҳандаҳои беморро ворид кунед",
        "predict.submit":     "Пешгӯӣ кардан",
        "predict.success":    "Синфи пешгӯишуда: {cls}",
        "predict.probas":     "Эҳтимолиятҳо аз рӯи синфҳо:",
        "predict.link_scen":  "📊 Сценарияҳои ДМЖП",
        "predict.link_dash":  "🖼️ Дашбордҳои HBS",
        "field.age":          "синну сол",
        "field.gender":       "ҷинс",
        "field.cp":           "намуди дард дар сина",
        "predict.caption":      "Jism HBS · Скрининги ML",
        "predict.model_missing":"Файли модел ёфт нашуд: {path}",
        "predict.disclaimer_md": (
            "**Огоҳӣ**\n"
            "- Модел дар **920 сабт** омӯзонида шудааст.\n"
            "- Синфи `4` тақрибан пешгӯӣ карда намешавад.\n"
            "- Accuracy ≈ **0.625**, Macro F1 ≈ **0.418**.\n"
            "- Ин **ташхиси тиббӣ нест**."
        ),
        "predict.cross_header": "🔬 Тафсири механики",
        "predict.cross_body": (
            "Модели ML **баҳои статикии хавф** медиҳад. Барои дидани он, ки ин хавф "
            "дар динамика чӣ гуна зоҳир мешавад (фишорҳо, Qp/Qs, SaO₂, ремоделятсия), "
            "ба бахши HBS гузаред."
        ),
        "field.trestbps":       "фишор дар ҳолати оромӣ",
        "field.chol":           "холестерин",
        "field.fbs":            "қанди хун дар наҳор > 120",
        "field.restecg":        "ЭКГ дар ҳолати оромӣ",
        "field.thalch":         "набзи максималӣ",
        "field.exang":          "стенокардия ҳангоми борбардорӣ",
        "field.oldpeak":        "депрессияи ST",
        "field.slope":          "майли ST",
        "field.ca":             "шумораи рагҳои калон",
        "field.thal":           "талассемия",

        # --- 3_hbs_overview.py ---
        "hbs.title":          "🫀 Jism HBS · модели механикии организм",
        "hbs.intro_md": (
            "### **HBS** — Human Body Simulation\n\n"
            "HBS — модели механикии организм: 12 аъзо, ки бо "
            "муодилаҳои дифференсиалӣ алоқаманданд. Бархилофи модели ML, ки "
            "баҳои статикии хавфро медиҳад, HBS **динамикаро** дар вақт нишон медиҳад."
        ),
        "hbs.modules_md": (
            "### Аъзоҳо ва зерсистемаҳо\n\n"
            "| Модул | Чиро моделсозӣ мекунад |\n"
            "|---|---|\n"
            "| `heart.py` | Дили 4-камерагӣ бо клапанҳо ва ДМЖП |\n"
            "| `lungs.py` | Шушҳои 2-компартментӣ + рекруитмент + ремоделятсия |\n"
            "| `baroreflex.py` | Барорефлекс: ЧСС, инотропия, вазомоторика |\n"
            "| `brain.py` | Мағзи сар: ауторегулятсия, O₂/CO₂, лактат, аммиак |\n"
            "| `liver.py` | Ҷигар: гемодинамикаи порталӣ, метаболизм |\n"
            "| `kidney.py` | Буйракҳо: СКФ, диурез, клиренси токсин |\n"
            "| `gitract.py` | ЖКТ: ҷаббиш, гардиши хуни порталӣ |\n"
            "| `peripheral_tissues.py` | Периферия: ауторегулятсия, VO₂, лактат |\n"
            "| `jugular_vein.py` | Венаи яремӣ: буфери ҷараёни мағзӣ |\n"
            "| `blood.py` | Хун: обанбори ягонаи консентратсияҳо |\n"
            "| `gas_exchange.py` | Мубодилаи гази алвеолавӣ-капиллярӣ O₂/CO₂ |\n"
            "| `whole_body.py` | Ҷамъоварии ҳамаи аъзоҳо + рагҳои Windkessel |"
        ),
        "hbs.scenarios_md": (
            "### Сценарияҳои ДМЖП\n\n"
            "Модел 5 фенотипи клиникиеро нишон медиҳад:\n\n"
            "1. **Солим** — шунт нест, Qp/Qs ≈ 1.0\n"
            "2. **ДМЖП-и хурд (R=5.0)** — шунти ночизи L→R\n"
            "3. **ДМЖП-и калон (R=1.0)** — шунти назарраси L→R, Qp/Qs > 1.4\n"
            "4. **Эйзенменгер, компенсацияшуда** — PVR ≈ SVR, ПЖ гипертрофияшуда\n"
            "5. **Эйзенменгер, декомпенсацияшуда** — PVR > SVR, шунти R→L, цианоз"
        ),
        "hbs.howto_md": (
            "### Чӣ тавр дашбордҳоро хонем\n\n"
            "- **Портретҳои PV** — шакли ҳалқаи «ҳаҷм–фишор» кори меъдачаро нишон медиҳад\n"
            "- **Qp/Qs** — нисбати гардиши хуни шушҳо ва системӣ (меъёр ≈ 1.0)\n"
            "- **SaO₂** — сатуратсияи хуни артериалӣ; < 90% = гипоксемия\n"
            "- **Шунти R→L** — ҳиссаи хуни венавӣ, ки мустақиман ба аорта меравад"
        ),
        "hbs.tech_md": (
            "### Тафсилоти техникӣ\n\n"
            "- Интегратор: **LSODA** (scipy), rtol=1e-4, atol=1e-5\n"
            "- Вақти симулятсия: **800 с** (калибризатсия 400–600 с + иҷрои корӣ)\n"
            "- 5 сценария ба таври параллелӣ тавассути **joblib/loky** ҳисоб карда мешаванд\n"
            "- Натиҷаҳо дар `.npz`, дашбордҳо — дар `.png` нигоҳ дошта мешаванд"
        ),
        "hbs.artifacts_ok":   "✅ Ҳама артефактҳои HBS мавҷуданд (5 сценария + 9 дашборд).",
        "hbs.artifacts_miss": "❌ Артефактҳои HBS ёфт нашуданд:",
        "hbs.artifacts_hint": (
            "Скриптҳои `run_simulation_parallel.py` ва `visualize_vsd_comparison.py`-ро "
            "дар лоиҳаи HBS иҷро кунед, сипас `.npz`-ро ба `hbs_results/` ва "
            "PNG-ро ба `hbs_results/plots/` нусхабардорӣ кунед."
        ),

        # --- 4_hbs_scenarios.py ---
        "scen.title":         "📊 Сценарияҳои ДМЖП — динамикаи механики",
        "scen.selector":      "Сценария",
        "scen.tab_press":     "Фишорҳо",
        "scen.tab_flow":      "Ҷараёнҳо",
        "scen.tab_vol":       "Ҳаҷмҳо ва газҳо",
        "scen.caption":           "Jism HBS · маълумоти зинда аз .npz",
        "scen.caption_npz":       "Маълумот аз `.npz`, ки тавассути HBS (`run_simulation_parallel.py`) сохта шудааст.",
        "scen.not_found":         "Файли `.npz` дар `hbs_results/` ёфт нашуд. Аввал симулятсияи HBS-ро иҷро кунед (ба саҳифаи «Дар бораи HBS» нигаред).",
        "scen.subheader_press":   "Фишори системӣ ва шушҳо",
        "scen.subheader_flow":    "Ҷараёни хуни системӣ ва шушҳо",
        "scen.subheader_qpqs":    "Qp/Qs (миёнаи ҳаракаткунанда)",
        "scen.subheader_vol":     "Ҳаҷми меъдачаҳо",
        "scen.subheader_sao2":    "Сатуратсияи хуни артериалӣ",
        "scen.caption_press":     "Меъёр: P_sa ≈ 90–110, P_pa ≈ 15–25 мм рт. ст.",
        "scen.caption_sao2":      "Меъёр: SaO₂ ≥ 95%. Ҳангоми шунти R→L аз 90% поён меояд.",
        "scen.link_dash":         "→ Дашбордҳои тайёри HBS",
        "scen.metric_ppa":        "мм рт. ст.",
        "scen.metric_psa":        "мм рт. ст.",
        "scen.metric_sao2":       "%",
        "scen.time_axis":         "Вақт (с)",
        "scen.unit_ml_per_s":     "мл/с",
        "scen.unit_ml":           "мл",

        # --- Имена сценариев ---
        "scenario.healthy":                "Солим",
        "scenario.vsd_small":              "ДМЖП-и хурд (R=5.0)",
        "scenario.vsd_large":              "ДМЖП-и калон (R=1.0)",
        "scenario.eisenmenger_comp":       "Эйзенменгер, компенсацияшуда",
        "scenario.eisenmenger_decomp":     "Эйзенменгер, декомпенсацияшуда",

        # --- 5_hbs_dashboards.py ---
        "dash.title":         "🖼️ Дашбордҳои Jism HBS",
        "dash.caption":       "Визуализатсияҳои тайёри модели механикӣ (5 сценарияи ДМЖП).",
        "dash.section":       "Бахш",
        "dash.dashboard":     "Дашборд",
        "dash.download":      "⬇️ Боргирии PNG",
        "dash.dir_missing":       "Папка ёфт нашуд: `{path}`. Файлҳои PNG-ро аз лоиҳаи HBS нусхабардорӣ кунед.",
        "dash.file_missing":      "Файл ёфт нашуд: `{path}`",
        "dash.link_scen":         "← Графикҳои зинда аз .npz",
        "dash.link_predict":      "← Пешгӯии ML",

        # --- Заголовки и описания дашбордов ---
        "dash.comprehensive.title": "Дашборди мукаммал",
        "dash.comprehensive.descr": "Панели ҷамъбастии 4×4: Qp/Qs, фишорҳо, ҳаҷмҳо, ҷараёни хуни минтақавӣ, функсияи буйракҳо, газҳои хун ва ҷадвали ҷамъбастӣ.",
        "dash.hemo_cmp.title":      "Муқоисаи гемодинамика 3×3",
        "dash.hemo_cmp.descr":      "P_sa/P_pa, Q_aortic/Q_pulmonary, Qp/Qs, шунт тавассути ДМЖП, ҳаҷми меъдачаҳо, волемия, Q_brain, SaO₂ ва ҷадвали ҷамъбастӣ.",
        "dash.pv_loops.title":      "Портретҳои фазавии PV",
        "dash.pv_loops.descr":      "Ҳалқаҳои «ҳаҷм–фишор» барои ЛЖ ва ПЖ дар ~2 кардиоцикли охирин. Нишон медиҳанд, ки чӣ тавр ремоделятсия кори меъдачаҳоро тағйир медиҳад.",
        "dash.cardiac_detail.title":"Таҳлили муфассали дил",
        "dash.cardiac_detail.descr":"Qp/Qs дар вақт, ҳаҷми ЛЖ/ПЖ, коррелятсияи шунт ва Qp/Qs, тақсимоти минтақавии ҷараёни хун.",
        "dash.gas.title":           "Мубодилаи гази O₂ / CO₂",
        "dash.gas.descr":           "Консентратсия ва фишори парсиалии O₂/CO₂, компонентҳои VO₂, тавозуни CO₂ аз рӯи сценарияҳо.",
        "dash.timeseries.title":    "Қаторҳои вақт",
        "dash.timeseries.descr":    "9 панел: фишорҳо, ҷараёнҳо, Qp/Qs, шунт, ҳаҷми ЛЖ/ПЖ, SaO₂.",
        "dash.steady.title":        "Нишондиҳандаҳои устуворшуда",
        "dash.steady.descr":        "P_sa, P_pa, Q_aortic, V_rv, V_blood, GFR, SaO₂, Qp/Qs.",
        "dash.shunt.title":         "Таҳлили таъсири андозаи ДМЖП",
        "dash.shunt.descr":         "Динамикаи Qp/Qs, фишорҳо, коррелятсияи шунт↔Qp/Qs, фраксияи R→L, радари нишондиҳандаҳои меъёршуда.",
        "dash.schematic.title":     "Схемаи дил: меъёр vs Эйзенменгер",
        "dash.schematic.descr":     "Самти шунт ва геометрияи камераҳо: солим vs декомпенсацияшуда.",

        # --- Служебные ---
        "hbs.load_error":         "Боргирии {name} муяссар нашуд: {err}",

        # --- Dashboard groups ---
        "group.summary":      "📋 Ҷамъбастҳо",
        "group.heart_lungs":  "🫀 Дил ва шушҳо",
        "group.gas":          "💨 Мубодилаи газҳо",
        "group.hemo":         "🩸 Гемодинамика",
        "group.schemas":      "🎨 Схемаҳо",
    },

}


def get_lang() -> str:
    """
    Текущий язык. Источники приоритета:
      1. query_params["lang"] — персистентность через URL;
      2. session_state["lang"] — на время сессии;
      3. DEFAULT_LANG.
    """
    qp_lang = st.query_params.get("lang")
    if qp_lang in SUPPORTED_LANGS:
        st.session_state.lang = qp_lang
        return qp_lang
    if "lang" in st.session_state:
        return st.session_state.lang
    st.session_state.lang = DEFAULT_LANG
    return DEFAULT_LANG


def set_lang(lang: str) -> None:
    """Смена языка + синхронизация URL."""
    if lang not in SUPPORTED_LANGS:
        return
    st.session_state.lang = lang
    st.query_params["lang"] = lang
    st.rerun()


def t(key: str, **fmt) -> str:
    """
    Перевод по ключу. Если ключа нет — возвращает сам ключ (fail-visible,
    удобно при разработке). Поддерживает форматирование через .format().
    """
    lang = get_lang()
    val = TRANSLATIONS.get(lang, {}).get(key)
    if val is None:
        # попробуем DEFAULT_LANG, чтобы UI не ломался
        val = TRANSLATIONS.get(DEFAULT_LANG, {}).get(key)
    if val is None:
        return key
    return val.format(**fmt) if fmt else val


def language_selector() -> None:
    """Переключатель языка для sidebar."""
    lang = get_lang()
    options = {"ru": "🇷🇺 Русский", "en": "🇬🇧 English", "tj": "🇹🇯 Тоҷикӣ"}
    keys = list(options.keys())
    choice = st.sidebar.radio(
        t("lang.label"),
        keys,
        format_func=lambda k: options[k],
        index=keys.index(lang),
        key="_lang_selector",
    )
    if choice != lang:
        set_lang(choice)