# HBS FastAPI Tutorial 04 — Полноценный эндпоинт `/simulate/full` с Workflow и JWT

Теперь мы приближаемся к **реальному production-коду** HBS.  
В этом уроке мы создадим **главный клинический эндпоинт** — запуск персонализированной симуляции для конкретного пациента с полным оркестрированием через `SimulationWorkflow`: загрузка данных из ЭМК, калибровка модели, запуск расчёта и сохранение результатов в историю.

---

## 1. Цель урока

Научиться:
- Интегрировать несколько сервисов (`EMRClient`, `CalibrationService`, `SimulationService`, `HistoryService`)
- Использовать `SimulationWorkflow` внутри API для оркестрации
- Добавлять JWT-авторизацию с ролями (врач / пациент)
- Создавать полноценный бизнес-эндпоинт `/simulate/full`, который возвращает результаты симуляции и сводку
- Обрабатывать ошибки и компенсационные действия (rollback при сбое)

---

## 2. Установка зависимостей

Добавим библиотеки для JWT и HTTP-клиента (для ЭМК):

```bash
pip install fastapi uvicorn pydantic python-dotenv httpx python-jose[cryptography] passlib[bcrypt]
```

---

## 3. Pydantic модели (`app/schemas/`)

Расширим существующие схемы для полноценного запроса и ответа.

**`app/schemas/simulation.py`** (дополнение):

```python
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class FullSimulationRequest(BaseModel):
    patient_id: str
    # параметры симуляции можно либо взять из данных пациента, либо передать явно
    vsd_resistance: Optional[float] = None   # если None – будет калибровка
    flow_dependent_lungs: bool = False
    t_span: List[float] = [0, 200]
    t_eval_points: int = 2000

class FullSimulationResponse(BaseModel):
    sim_id: str
    patient_id: str
    status: str
    summary: Optional[SimulationSummary] = None
    results: Optional[SimulationResponse] = None
    created_at: datetime
    calibrated_vsd_resistance: Optional[float] = None
```

---

## 4. JWT Auth (`app/core/security.py`)

Реализуем JWT с проверкой роли. В HBS у нас будут роли: `doctor` (врач) и `patient` (пациент). Врач может запускать симуляции для любых пациентов, пациент – только для себя.

```python
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = "hbs-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def create_access_token(subject: str, role: str = "patient"):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        role: str = payload.get("role", "patient")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": user_id, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

def get_current_doctor(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can perform this action")
    return current_user
```

---

## 5. Клиенты внешних сервисов

### 5.1. Клиент ЭМК (`app/clients/emr_client.py`)

Имитирует получение данных пациента из электронной медицинской карты.

```python
import httpx
from typing import Dict, Any

class EMRClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url

    async def get_patient_data(self, patient_id: str) -> Dict[str, Any]:
        # В реальности – HTTP-запрос к сервису ЭМК
        # Здесь заглушка с мок-данными
        # return await httpx.AsyncClient().get(f"{self.base_url}/patients/{patient_id}")
        return {
            "patient_id": patient_id,
            "name": "Иванов Иван Иванович",
            "age": 54,
            "diagnosis": "Дефект межжелудочковой перегородки (ДМЖП), компенсированный",
            "labs": {
                "bilirubin": 18.2,
                "ammonia": 45.0,
                "albumin": 3.2,
                "toxins": 2.1
            },
            "vital_signs": {
                "heart_rate": 82,
                "systolic_bp": 145,
                "diastolic_bp": 90,
                "oxygen_saturation": 0.96
            }
        }
```

### 5.2. Клиент калибровки (`app/clients/calibration_client.py`)

Может быть отдельным микросервисом или функцией внутри бэкенда. Здесь мы создадим заглушку.

```python
class CalibrationClient:
    async def calibrate_vsd(self, patient_data: dict) -> float:
        """
        По данным пациента подбирает сопротивление ДМЖП.
        В реальности – вызов ML-модели или оптимизатора.
        """
        # Простейшая эвристика: если есть диагноз ДМЖП, предполагаем сопротивление 5.0,
        # если нет – бесконечность.
        diagnosis = patient_data.get("diagnosis", "")
        if "ДМЖП" in diagnosis:
            return 5.0
        else:
            return float('inf')
```

---

## 6. Workflow (`app/core/simulation_workflow.py`)

Этот класс оркестрирует весь процесс: получение данных пациента → калибровка → запуск симуляции → сохранение истории.

```python
from app.clients.emr_client import EMRClient
from app.clients.calibration_client import CalibrationClient
from app.services.simulation_service import run_simulation
from app.core.storage import simulations_db, simulation_summaries
from app.schemas.simulation import SimulationSummary, FullSimulationResponse
import uuid
from datetime import datetime
import numpy as np

class SimulationWorkflow:
    def __init__(self):
        self.emr_client = EMRClient()
        self.calibration_client = CalibrationClient()

    async def run_full_simulation(self, patient_id: str, params: dict) -> FullSimulationResponse:
        """
        Полный workflow:
        1. Получить данные пациента из ЭМК.
        2. Если vsd_resistance не указан – калибровать.
        3. Запустить симуляцию.
        4. Сохранить результаты и сводку.
        5. Вернуть ответ.
        """
        sim_id = str(uuid.uuid4())

        try:
            # 1. Получение данных пациента
            patient_data = await self.emr_client.get_patient_data(patient_id)

            # 2. Калибровка (если не передан параметр)
            vsd = params.get("vsd_resistance")
            if vsd is None:
                vsd = await self.calibration_client.calibrate_vsd(patient_data)
                calibrated = vsd
            else:
                calibrated = None

            # 3. Запуск симуляции
            result = run_simulation(
                vsd_resistance=vsd,
                flow_dependent_lungs=params.get("flow_dependent_lungs", False),
                t_span=params.get("t_span", [0, 200]),
                t_eval_points=params.get("t_eval_points", 2000),
                initial_concentrations=patient_data.get("labs")
            )

            # 4. Сохранение
            simulations_db[sim_id] = result

            # Вычисляем summary
            t_arr = np.array(result['time'])
            mask = t_arr >= (params.get("t_span")[1] - 50) if params.get("t_span") else np.ones_like(t_arr, dtype=bool)
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
                summary = SimulationSummary(qp_qs=1.0, mean_sa_pressure=80, mean_pa_pressure=15, blood_volume=5000, gfr=1.2)

            simulation_summaries[sim_id] = summary

            # 5. Формируем ответ
            return FullSimulationResponse(
                sim_id=sim_id,
                patient_id=patient_id,
                status="completed",
                summary=summary,
                results=result,
                created_at=datetime.utcnow(),
                calibrated_vsd_resistance=calibrated
            )

        except Exception as e:
            # Компенсация: удаляем частично сохранённые данные
            simulations_db.pop(sim_id, None)
            simulation_summaries.pop(sim_id, None)
            raise RuntimeError(f"Workflow failed: {str(e)}")
```

---

## 7. Полноценный роутер (`app/api/v1/simulations.py`)

Добавим новый эндпоинт `/simulate/full`.

```python
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.simulation import FullSimulationRequest, FullSimulationResponse
from app.core.simulation_workflow import SimulationWorkflow
from app.core.security import get_current_user, get_current_doctor

router = APIRouter()
workflow = SimulationWorkflow()

@router.post("/simulate/full", response_model=FullSimulationResponse)
async def run_full_simulation(
    request: FullSimulationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Полноценный запуск персонализированной симуляции.
    Доступен только врачам (или пациентам для себя – опционально).
    """
    # Проверка прав: врач может запускать для любого пациента, пациент – только для себя
    if current_user["role"] == "patient" and current_user["user_id"] != request.patient_id:
        raise HTTPException(status_code=403, detail="Patient can only simulate for themselves")

    try:
        params = request.dict(exclude={"patient_id"})
        result = await workflow.run_full_simulation(request.patient_id, params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 8. Обновление `app/main.py`

Добавим импорт нового роутера (если он уже есть, то ничего не меняем). Убедитесь, что роутер `simulations` подключён.

---

## 9. Запуск и тестирование

1. Запустите бэкенд:

```bash
cd hbs/backend
uvicorn app.main:app --reload --port 8000
```

2. Получите JWT-токен (можно создать эндпоинт `/auth/login` или использовать готовый токен).

3. Отправьте POST-запрос на `/api/v1/simulate/full` с заголовком:

```
Authorization: Bearer <your-jwt-token>
```

Body:

```json
{
  "patient_id": "p_001",
  "vsd_resistance": null,
  "flow_dependent_lungs": false,
  "t_span": [0, 200],
  "t_eval_points": 2000
}
```

Если `vsd_resistance` = `null`, будет выполнена калибровка по данным пациента.

**Ответ** – объект `FullSimulationResponse`, содержащий ID симуляции, сводку, полные результаты и калиброванное сопротивление.

---

## Что мы приблизили к реальному проекту HBS

- ✅ Интеграция с внешним сервисом ЭМК (заглушка, но готова к реальному API)
- ✅ Калибровка модели по данным пациента (автоматический подбор параметров)
- ✅ Полноценный Workflow с компенсацией ошибок (rollback)
- ✅ JWT-авторизация с ролями (врач / пациент)
- ✅ Оркестрация нескольких сервисов в одном эндпоинте
- ✅ Сохранение результатов в истории для последующего доступа

---

## Дальнейшие шаги

В следующих уроках (`hbs_fastapi_tutorial_05.md` и далее) мы:

- Добавим **WebSocket** для потоковой передачи промежуточных результатов во время длительной симуляции
- Реализуем **сравнение сценариев** (запуск нескольких симуляций с разными параметрами)
- Настроим **базу данных** (PostgreSQL + TimescaleDB) для постоянного хранения
- Добавим **аутентификацию** с реальной проверкой пользователей
- Подготовим **Docker-контейнеризацию** для развёртывания

---

**Готово!** Теперь ваш бэкенд умеет выполнять сложный клинический сценарий: загружать данные пациента, калибровать модель, запускать симуляцию и возвращать результаты, защищённые JWT. Это ещё один шаг к созданию полноценной системы поддержки принятия врачебных решений.

**Удачи в разработке HBS!** 🧬🚀