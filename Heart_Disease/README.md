# 🧬 Jism HBS

Учебно-исследовательский проект гибридного моделирования
сердечно-сосудистой патологии.

## Что внутри

**Jism HBS** объединяет два подхода:

| Раздел | Инструмент | Что делает |
|---|---|---|
| ML-скрининг | XGBoost (UCI Heart Disease) | Классификация риска по 13 признакам |
| HBS | Механистическая ODE-модель (12 органов) | Динамика гемодинамики во времени |

## Запуск

```bash
cd streamlit
streamlit run app.py
```

## Обновление артефактов HBS

1. В HBS-проекте:
   ```bash
   python run_simulation_parallel.py --log-flat
   python visualize_vsd_comparison.py
   ```
2. Скопировать:
   ```powershell
   copy <HBS>\vsd_results_patient_*.npz hbs_results\
   copy <HBS>\*.png hbs_results\plots\
   ```

## Структура

```
Heart_Disease\
├── streamlit\        # Streamlit-приложение Jism HBS
├── models\           # XGBoost-модель
└── hbs_results\      # артефакты HBS (.npz + .png)
```
