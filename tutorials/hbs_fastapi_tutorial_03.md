# HBS FastAPI Tutorial 03 — Интеграция реальной модели WholeBodyModel

Теперь мы делаем **ключевой шаг** в проекте **Human Body Simulation (HBS)**.  
В этом уроке мы заменим mock-данные в эндпоинте `/simulate` на **реальную физиологическую модель** `WholeBodyModel`, которая уже реализована в модулях `whole_body.py`, `heart.py`, `lungs.py` и других. Вы научитесь подключать сложные расчёты на Python к веб-серверу FastAPI, обрабатывать параметры запроса и возвращать временные ряды для визуализации.

---

## 1. Цель урока

Научиться:
- Организовывать код симуляции в отдельный сервисный слой
- Интегрировать `WholeBodyModel` в эндпоинт FastAPI
- Передавать параметры (сопротивление ДМЖП, адаптация лёгких) в модель
- Обрабатывать ошибки и возвращать структурированный ответ
- Сохранять результаты симуляции в in-memory хранилище для последующего доступа

---

## 2. Установка зависимостей

Убедитесь, что установлены необходимые пакеты:

```bash
pip install numpy scipy
```

Если вы ещё не установили `fastapi`, `uvicorn`, `pydantic` и `python-dotenv` — сделайте это:

```bash
pip install fastapi uvicorn pydantic python-dotenv
```

---

## 3. Организация кода модели

Скопируйте все файлы модели из вашей рабочей директории (где находятся `whole_body.py`, `organ_base.py`, `heart.py` и т.д.) в папку `app/models/`. Это позволит импортировать их из сервисного слоя.

Структура после копирования:

```
app/
├── models/
│   ├── __init__.py
│   ├── whole_body.py
│   ├── organ_base.py
│   ├── heart.py
│   ├── lungs.py
│   ├── liver.py
│   ├── kidney.py
│   ├── blood.py
│   ├── gitract.py
│   └── brain.py
├── services/
│   └── simulation_service.py   (создадим)
├── api/v1/
│   └── simulations.py          (обновим)
└── ...
```

---

## 4. Создание сервиса симуляции (`app/services/simulation_service.py`)

Этот слой будет отвечать за вызов `WholeBodyModel` и преобразование результата в формат, ожидаемый API.

```python
# app/services/simulation_service.py
import numpy as np
from typing import Dict, Any
from app.models.whole_body import WholeBodyModel

def run_simulation(
    vsd_resistance: float,
    flow_dependent_lungs: bool,
    t_span: list,
    t_eval_points: int,
    initial_concentrations: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Запускает симуляцию WholeBodyModel с заданными параметрами.

    Возвращает словарь с ключами:
        - time: list[float]
        - outputs: dict[str, list[float]]
    """
    # Подготовка начальных концентраций (если не переданы – используем здоровые значения)
    if initial_concentrations is None:
        initial_concentrations = {
            'tox': 0.0,
            'bilirubin': 0.2,
            'ammonia': 0.5,
            'albumin': 4.0
        }

    # Параметры крови
    blood_params = {
        'initial_concentrations': initial_concentrations,
        'V0': 5000.0
    }

    # Параметры лёгких (адаптация сопротивления)
    lungs_params = {
        'flow_dependent_resistance': flow_dependent_lungs
    }

    # Создаём модель
    model = WholeBodyModel(
        blood_params=blood_params,
        vsd_resistance=vsd_resistance,
        flow_dependent_lungs=flow_dependent_lungs,
        lungs_params=lungs_params
    )

    # Временная сетка
    t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)

    # Запуск симуляции (синхронный вызов)
    sol = model.simulate(t_span, t_eval)

    # Сбор выходных переменных для каждого момента времени
    outputs_list = []
    for i, ti in enumerate(sol.t):
        out = model.compute_outputs(ti, sol.y[:, i])
        outputs_list.append(out)

    # Формируем словарь с временными рядами
    outputs = {}
    if outputs_list:
        # Берём ключи из первого элемента
        for key in outputs_list[0].keys():
            outputs[key] = [out[key] for out in outputs_list]

    # Возвращаем результат
    return {
        'time': sol.t.tolist(),
        'outputs': outputs
    }
```

---

## 5. Обновление эндпоинта `/simulate` (`app/api/v1/simulations.py`)

Теперь заменим mock-генерацию на вызов нашего сервиса.

```python
# app/api/v1/simulations.py
from fastapi import APIRouter, HTTPException
from app.schemas.simulation import SimulationRequest, SimulationResponse, SimulationSummary
from app.core.storage import simulations_db, simulation_summaries
from app.services.simulation_service import run_simulation
import uuid
import numpy as np

router = APIRouter()

@router.post("/simulate/", response_model=SimulationResponse)
async def run_simulation_endpoint(request: SimulationRequest):
    """
    Запуск симуляции гемодинамики с заданными параметрами.
    Использует реальную модель WholeBodyModel.
    """
    sim_id = str(uuid.uuid4())

    try:
        # Конвертируем vsd_resistance: если бесконечность, передаём np.inf
        vsd = float('inf') if request.vsd_resistance == float('inf') else request.vsd_resistance

        # Запускаем симуляцию
        result = run_simulation(
            vsd_resistance=vsd,
            flow_dependent_lungs=request.flow_dependent_lungs,
            t_span=request.t_span,
            t_eval_points=request.t_eval_points,
            initial_concentrations=None  # можно расширить для получения из запроса
        )

        # Формируем ответ
        response = SimulationResponse(
            time=result['time'],
            outputs=result['outputs'],
            message="Симуляция завершена успешно"
        )

        # Сохраняем в хранилище
        simulations_db[sim_id] = response

        # Вычисляем summary (средние за последние 50 секунд)
        t_arr = np.array(result['time'])
        mask = t_arr >= (request.t_span[1] - 50)
        if np.any(mask):
            qp_qs_mean = np.mean(np.array(result['outputs']['Qp_Qs'])[mask])
            ps_mean = np.mean(np.array(result['outputs']['P_sa'])[mask])
            pp_mean = np.mean(np.array(result['outputs']['P_pa'])[mask])
            vb_mean = np.mean(np.array(result['outputs']['V_blood'])[mask])
            gfr_mean = np.mean(np.array(result['outputs']['GFR'])[mask])
            summary = SimulationSummary(
                qp_qs=qp_qs_mean,
                mean_sa_pressure=ps_mean,
                mean_pa_pressure=pp_mean,
                blood_volume=vb_mean,
                gfr=gfr_mean
            )
        else:
            summary = SimulationSummary(
                qp_qs=1.0,
                mean_sa_pressure=80,
                mean_pa_pressure=15,
                blood_volume=5000,
                gfr=1.2
            )
        simulation_summaries[sim_id] = summary

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка симуляции: {str(e)}")
```

---

## 6. Проверка доступности модели

Убедитесь, что все импорты корректны. В `app/models/__init__.py` можно добавить экспорт `WholeBodyModel` для удобства.

Также проверьте, что в вашей модели `whole_body.py` нет ошибок импорта внутри (например, `from heart import Heart4Chambers`). Если модули лежат в одной папке, это должно работать.

---

## 7. Запуск и тестирование

1. Запустите бэкенд:

```bash
cd hbs/backend
uvicorn app.main:app --reload --port 8000
```

2. Отправьте POST-запрос на `/api/v1/simulate/` через Postman или curl:

```json
{
  "vsd_resistance": 5.0,
  "flow_dependent_lungs": false,
  "t_span": [0, 200],
  "t_eval_points": 2000
}
```

3. В ответе вы должны получить реальные временные ряды, сгенерированные `WholeBodyModel`. Обратите внимание, что расчёт может занять несколько секунд — это нормально.

4. Проверьте, что эндпоинт `/api/v1/simulations/{sim_id}/summary` возвращает корректные средние значения.

---

## 8. Обработка ошибок и оптимизация

- Если симуляция занимает слишком много времени, можно вынести её в отдельный поток (используя `asyncio.to_thread` или `run_in_executor`) и добавить WebSocket для прогресса — это будет рассмотрено в следующих уроках.
- Для больших `t_eval_points` (например, > 5000) может потребоваться увеличить таймаут на стороне клиента или использовать потоковую передачу.
- Добавьте валидацию параметров: например, `t_span[0] < t_span[1]`, `t_eval_points > 0`.

---

## 9. Что мы приблизили к реальному проекту HBS

- ✅ Интегрировали реальную физиологическую модель вместо mock-данных
- ✅ Организовали чёткое разделение на сервисный слой и API
- ✅ Сохраняем результаты для последующего доступа по ID
- ✅ Готовим основу для дальнейшего расширения (калибровка, сравнение сценариев, WebSocket)

---

## 10. Дальнейшие шаги

В следующих уроках (`hbs_fastapi_tutorial_04.md` и далее) мы:

- Добавим **WebSocket** для потоковой передачи промежуточных результатов, чтобы фронтенд мог отображать графики в реальном времени
- Реализуем **сравнение сценариев** (запуск нескольких симуляций с разными параметрами)
- Подключим **калибровку модели** по клиническим данным пациента
- Настроим **базу данных** (PostgreSQL + TimescaleDB) для хранения результатов и истории
- Добавим **аутентификацию** (JWT) и ролевую модель

---

## Готово!

Теперь ваш бэкенд способен выполнять реальные расчёты гемодинамики и возвращать их в веб-интерфейс. Это важнейший шаг на пути к созданию клинического инструмента для моделирования физиологии человека.

**Удачи в разработке HBS!** 🧬🚀