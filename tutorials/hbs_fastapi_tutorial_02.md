# HBS FastAPI Tutorial 02 — Структурированный бэкенд для Human Body Simulation

Теперь мы сделаем шаг ближе к реальному проекту **HBS (Human Body Simulation)**.  
Мы создадим **структурированный FastAPI-сервис**, который станет основой для всей backend-логики: управления пациентами, запуска симуляций, хранения результатов и взаимодействия с React.js фронтендом.

Цель — организовать код по архитектурным принципам, подготовить Pydantic-схемы для обмена данными и реализовать первые эндпоинты (заглушки) для будущей интеграции с моделью `WholeBodyModel`.

---

## 1. Цель урока

Научиться:
- Использовать Pydantic модели для описания запросов и ответов
- Создавать организованную структуру папок (как будет в production-проекте)
- Реализовать роутеры для основных сущностей: пациенты, симуляции, сценарии
- Подготовить сервис к подключению реальной физиологической модели (WholeBodyModel)
- Использовать in-memory хранилище для быстрого прототипирования

---

## 2. Создание структуры проекта

```bash
cd hbs/backend

mkdir -p app/api/v1
mkdir -p app/core
mkdir -p app/schemas
mkdir -p app/services
mkdir -p app/models
mkdir -p app/utils

cd app
touch __init__.py
touch main.py
touch core/__init__.py
touch schemas/__init__.py
touch services/__init__.py
touch api/__init__.py
touch api/v1/__init__.py
```

**Финальная структура (после урока):**
```
hbs/backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   └── __init__.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── patient.py
│   │   ├── simulation.py
│   │   └── scenario.py
│   ├── services/
│   │   └── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── patients.py
│   │       ├── simulations.py
│   │       └── scenarios.py
│   └── utils/
│       └── __init__.py
├── requirements.txt
└── .env
```

---

## 3. Установка зависимостей

```bash
pip install fastapi uvicorn pydantic python-dotenv
# Для будущей симуляции:
pip install numpy scipy
```

---

## 4. Pydantic-схемы

### 4.1. Схема пациента (`app/schemas/patient.py`)

```python
from pydantic import BaseModel
from typing import Optional
from datetime import date

class PatientBase(BaseModel):
    patient_id: str
    name: str
    age: int
    diagnosis: str
    admission_date: date

class PatientLabs(BaseModel):
    bilirubin: float = 0.0
    ammonia: float = 0.0
    albumin: float = 0.0
    toxins: float = 0.0

class Patient(PatientBase):
    labs: PatientLabs = PatientLabs()

class PatientCreate(PatientBase):
    labs: Optional[PatientLabs] = None
```

### 4.2. Схемы симуляции (`app/schemas/simulation.py`)

```python
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import numpy as np

class SimulationRequest(BaseModel):
    vsd_resistance: float = float('inf')   # сопротивление ДМЖП
    flow_dependent_lungs: bool = False
    t_span: List[float] = [0, 200]         # [start, end]
    t_eval_points: int = 2000              # количество точек для вывода
    patient_id: Optional[str] = None       # если привязано к пациенту

class SimulationResponse(BaseModel):
    time: List[float]
    outputs: Dict[str, List[float]]        # ключ – имя показателя, значение – временной ряд
    message: Optional[str] = None

class SimulationSummary(BaseModel):
    qp_qs: float
    mean_sa_pressure: float
    mean_pa_pressure: float
    blood_volume: float
    gfr: float
```

### 4.3. Схемы сценариев (`app/schemas/scenario.py`)

```python
from pydantic import BaseModel
from typing import List, Dict

class ScenarioPreset(BaseModel):
    name: str
    description: str
    vsd_resistance: float
    flow_dependent_lungs: bool
    color: str = '#cccccc'

class ScenarioComparisonRequest(BaseModel):
    scenario_ids: List[str]   # список имён сценариев
    metric: str = 'P_sa'      # какой показатель сравнивать
```

---

## 5. In-memory хранилище (заглушка)

Создадим временное хранилище в `app/core/storage.py` (позже заменим на БД).

```python
# app/core/storage.py
from typing import Dict, List
from app.schemas.patient import Patient
from app.schemas.simulation import SimulationResponse, SimulationSummary

# Хранилище пациентов
patients_db: Dict[str, Patient] = {}

# Хранилище результатов симуляций (ключ – id симуляции, значение – объект SimulationResponse)
simulations_db: Dict[str, SimulationResponse] = {}

# Хранилище summaries для быстрого доступа
simulation_summaries: Dict[str, SimulationSummary] = {}

# Предопределённые сценарии (можно хранить здесь или в конфиге)
SCENARIO_PRESETS = {
    'healthy': {
        'name': 'Здоровый',
        'description': 'Нормальная гемодинамика без дефектов',
        'vsd_resistance': float('inf'),
        'flow_dependent_lungs': False,
        'color': '#2ecc71'
    },
    'small_vsd': {
        'name': 'Малый ДМЖП (R=5.0)',
        'description': 'Небольшой лево-правый шунт',
        'vsd_resistance': 5.0,
        'flow_dependent_lungs': False,
        'color': '#f39c12'
    },
    'large_vsd': {
        'name': 'Большой ДМЖП (R=1.0)',
        'description': 'Значительный шунт с перегрузкой лёгких',
        'vsd_resistance': 1.0,
        'flow_dependent_lungs': True,
        'color': '#e74c3c'
    },
    'post_op': {
        'name': 'После операции',
        'description': 'Коррекция ДМЖП, восстановленная гемодинамика',
        'vsd_resistance': float('inf'),
        'flow_dependent_lungs': False,
        'color': '#3498db'
    }
}
```

---

## 6. Роутеры (эндпоинты)

### 6.1. Роутер для пациентов (`app/api/v1/patients.py`)

```python
from fastapi import APIRouter, HTTPException
from app.schemas.patient import Patient, PatientCreate
from app.core.storage import patients_db
from uuid import uuid4

router = APIRouter()

@router.post("/patients/", response_model=Patient)
async def create_patient(patient_data: PatientCreate):
    """Регистрация нового пациента в системе."""
    if patient_data.patient_id in patients_db:
        raise HTTPException(status_code=400, detail="Patient already exists")
    patient = Patient(
        patient_id=patient_data.patient_id,
        name=patient_data.name,
        age=patient_data.age,
        diagnosis=patient_data.diagnosis,
        admission_date=patient_data.admission_date,
        labs=patient_data.labs or {}
    )
    patients_db[patient.patient_id] = patient
    return patient

@router.get("/patients/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str):
    """Получить данные пациента по ID."""
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patients_db[patient_id]

@router.get("/patients/", response_model=list[Patient])
async def list_patients():
    """Список всех пациентов."""
    return list(patients_db.values())
```

### 6.2. Роутер для симуляций (`app/api/v1/simulations.py`)

```python
from fastapi import APIRouter, HTTPException
from app.schemas.simulation import SimulationRequest, SimulationResponse, SimulationSummary
from app.core.storage import simulations_db, simulation_summaries
import numpy as np
import uuid

router = APIRouter()

@router.post("/simulate/", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """
    Запуск симуляции гемодинамики с заданными параметрами.
    (Пока возвращает mock-данные, позже будет вызывать WholeBodyModel)
    """
    # Генерация уникального ID для симуляции
    sim_id = str(uuid.uuid4())

    # Временная сетка
    t = np.linspace(request.t_span[0], request.t_span[1], request.t_eval_points)

    # Mock-результаты (заглушка)
    # В реальности здесь будет вызов WholeBodyModel.simulate()
    freq = 0.5
    P_sa = 80 + 20 * np.sin(freq * t) + 10 * np.sin(0.3 * t)
    P_pa = 15 + 5 * np.sin(0.7 * t) + 3 * np.sin(0.2 * t)
    Q_aortic = 80 + 10 * np.sin(freq * t) + 5 * np.cos(0.1 * t)
    Q_pulmonary = Q_aortic * (1.0 if request.vsd_resistance == float('inf') else 2.5)
    Q_vsd = np.zeros_like(t) if request.vsd_resistance == float('inf') else 30 * np.ones_like(t)
    Qp_Qs = Q_pulmonary / (Q_aortic + 1e-6)
    V_lv = 120 + 20 * np.sin(freq * t * 1.2)
    V_rv = 60 + 15 * np.sin(freq * t * 1.3 + 0.5)
    V_blood = 5000 + 100 * np.sin(0.05 * t)

    outputs = {
        'P_sa': P_sa.tolist(),
        'P_pa': P_pa.tolist(),
        'Q_aortic': Q_aortic.tolist(),
        'Q_pulmonary': Q_pulmonary.tolist(),
        'Qp_Qs': Qp_Qs.tolist(),
        'Q_vsd': Q_vsd.tolist(),
        'V_lv': V_lv.tolist(),
        'V_rv': V_rv.tolist(),
        'V_blood': V_blood.tolist()
    }

    # Дополнительные метаболические показатели (заглушка)
    outputs['C_bilirubin_blood'] = (0.2 + 0.05 * np.sin(0.02 * t)).tolist()
    outputs['C_ammonia_blood'] = (0.5 + 0.1 * np.sin(0.03 * t)).tolist()
    outputs['C_albumin_blood'] = (4.0 + 0.2 * np.sin(0.01 * t)).tolist()
    outputs['GFR'] = (1.2 + 0.1 * np.sin(0.02 * t)).tolist()
    outputs['Q_brain'] = (50 + 5 * np.sin(0.4 * t)).tolist()
    outputs['O2_consumption'] = (10 + 2 * np.sin(0.4 * t)).tolist()

    response = SimulationResponse(
        time=t.tolist(),
        outputs=outputs,
        message="Симуляция завершена (mock-данные)"
    )

    # Сохраняем результат в хранилище
    simulations_db[sim_id] = response

    # Вычисляем summary (средние за последние 50 секунд)
    t_arr = np.array(t)
    mask = t_arr >= (request.t_span[1] - 50)
    if np.any(mask):
        qp_qs_mean = np.mean(np.array(outputs['Qp_Qs'])[mask])
        ps_mean = np.mean(np.array(outputs['P_sa'])[mask])
        pp_mean = np.mean(np.array(outputs['P_pa'])[mask])
        vb_mean = np.mean(np.array(outputs['V_blood'])[mask])
        gfr_mean = np.mean(np.array(outputs['GFR'])[mask])
        summary = SimulationSummary(
            qp_qs=qp_qs_mean,
            mean_sa_pressure=ps_mean,
            mean_pa_pressure=pp_mean,
            blood_volume=vb_mean,
            gfr=gfr_mean
        )
        simulation_summaries[sim_id] = summary
    else:
        simulation_summaries[sim_id] = SimulationSummary(
            qp_qs=1.0,
            mean_sa_pressure=80,
            mean_pa_pressure=15,
            blood_volume=5000,
            gfr=1.2
        )

    return response

@router.get("/simulations/{sim_id}/summary", response_model=SimulationSummary)
async def get_simulation_summary(sim_id: str):
    """Получить сводку (установившиеся значения) по ID симуляции."""
    if sim_id not in simulation_summaries:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return simulation_summaries[sim_id]

@router.get("/simulations/{sim_id}", response_model=SimulationResponse)
async def get_simulation_result(sim_id: str):
    """Получить полные результаты симуляции по ID."""
    if sim_id not in simulations_db:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return simulations_db[sim_id]
```

### 6.3. Роутер для сценариев (`app/api/v1/scenarios.py`)

```python
from fastapi import APIRouter
from app.core.storage import SCENARIO_PRESETS
from app.schemas.scenario import ScenarioPreset, ScenarioComparisonRequest

router = APIRouter()

@router.get("/scenarios/", response_model=list[ScenarioPreset])
async def list_scenarios():
    """Получить список предопределённых сценариев."""
    return list(SCENARIO_PRESETS.values())

@router.post("/scenarios/compare")
async def compare_scenarios(request: ScenarioComparisonRequest):
    """
    Сравнение нескольких сценариев (запускает симуляции для каждого).
    Пока возвращает заглушку.
    """
    # В будущем здесь будет запуск нескольких симуляций и возврат объединённых данных
    return {
        "message": "Сравнение сценариев будет реализовано в следующих уроках",
        "scenario_ids": request.scenario_ids,
        "metric": request.metric
    }
```

---

## 7. Обновлённый `main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from app.api.v1 import patients, simulations, scenarios

load_dotenv()

app = FastAPI(
    title="HBS Backend API",
    description="Сервер для Human Body Simulation — интеграция с ЭМК и симуляцией физиологии",
    version="0.2.0"
)

# Настройка CORS (для взаимодействия с React.js фронтендом)
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры
app.include_router(patients.router, prefix="/api/v1", tags=["Patients"])
app.include_router(simulations.router, prefix="/api/v1", tags=["Simulations"])
app.include_router(scenarios.router, prefix="/api/v1", tags=["Scenarios"])

@app.get("/")
def root():
    return {
        "service": "HBS Backend",
        "status": "running",
        "version": "0.2.0",
        "components": ["patients", "simulations", "scenarios"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 8. Запуск сервиса

```bash
cd hbs/backend
uvicorn app.main:app --reload --port 8000
```

Сервер будет доступен по адресу: **http://localhost:8000**

---

## 9. Проверка в Postman / curl

### 9.1. Создать пациента

**POST** `http://localhost:8000/api/v1/patients/`

Body (raw → JSON):
```json
{
  "patient_id": "p_001",
  "name": "Иванов Иван Иванович",
  "age": 54,
  "diagnosis": "Дефект межжелудочковой перегородки",
  "admission_date": "2025-03-10",
  "labs": {
    "bilirubin": 18.2,
    "ammonia": 45.0,
    "albumin": 3.2,
    "toxins": 2.1
  }
}
```

**Ответ** — созданный объект пациента.

### 9.2. Получить пациента

**GET** `http://localhost:8000/api/v1/patients/p_001`

### 9.3. Запустить симуляцию

**POST** `http://localhost:8000/api/v1/simulate/`

Body:
```json
{
  "vsd_resistance": 5.0,
  "flow_dependent_lungs": false,
  "t_span": [0, 200],
  "t_eval_points": 2000
}
```

**Ответ** — временные ряды (mock-данные) и сообщение.

### 9.4. Получить сводку

**GET** `http://localhost:8000/api/v1/simulations/{sim_id}/summary`  
(подставьте реальный ID, который вернулся после симуляции)

### 9.5. Список сценариев

**GET** `http://localhost:8000/api/v1/scenarios/`

---

## 10. Что мы приблизили к реальному проекту HBS

- ✅ Модульная архитектура (папки `schemas`, `api/v1`, `core`, `services`)
- ✅ Pydantic-модели для пациентов, симуляций и сценариев
- ✅ Подготовка к интеграции с `WholeBodyModel` (заглушка в `/simulate`)
- ✅ In-memory хранилище для быстрого прототипирования
- ✅ CORS настроен для взаимодействия с React.js
- ✅ Эндпоинты для управления пациентами, запуска симуляций и получения результатов

---

## 11. Дальнейшие шаги

В следующих уроках (`hbs_fastapi_tutorial_03.md` и далее) мы:
- Подключим реальную модель `WholeBodyModel` вместо mock-данных
- Добавим WebSocket для потоковой передачи промежуточных результатов
- Реализуем сравнение сценариев и калибровку модели
- Настроим сохранение результатов в базе данных (PostgreSQL + TimescaleDB)
- Добавим аутентификацию (JWT) и ролевую модель

---

## Готово!

Теперь у вас есть структурированный бэкенд для HBS, готовый к расширению и интеграции с фронтендом. Вы можете запускать сервер, проверять эндпоинты и постепенно заменять заглушки на реальную физиологическую модель.

**Удачи в разработке HBS!** 🧬🚀