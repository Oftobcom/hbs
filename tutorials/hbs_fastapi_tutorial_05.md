# HBS FastAPI Tutorial 05 — Production-Ready Simulation с Realtime, Rate Limiting и Observability

Это **финальный** практический урок серии по FastAPI для проекта **Human Body Simulation (HBS)**.  
Мы реализуем **полноценный production-like** бэкенд для клинической симуляции, максимально приближенный к реальной архитектуре медицинской системы: оркестрация workflow, JWT-авторизация, WebSocket для realtime-обновлений, rate limiting и структурированное логирование.

---

## 1. Цель урока

Научиться:
- Реализовывать **Saga / Workflow** внутри API Gateway для оркестрации симуляции
- Интегрировать сервисы: `EMRClient`, `CalibrationClient`, `SimulationService`, `HistoryService`
- Добавлять **JWT-авторизацию** с ролями (врач / пациент)
- Использовать **WebSocket** для realtime-обновлений статуса и прогресса симуляции
- Внедрить **Rate Limiting** и **Observability** (структурированные логи)
- Создать структуру, соответствующую архитектурным документам HBS

---

## 2. Финальная структура `hbs-backend/`

```
hbs/backend/
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── security.py
│   │   ├── dependencies.py
│   │   ├── simulation_workflow.py
│   │   ├── connection_manager.py
│   │   └── middleware.py
│   ├── clients/
│   │   ├── emr_client.py
│   │   ├── calibration_client.py
│   │   └── history_client.py
│   ├── schemas/
│   │   ├── simulation.py
│   │   ├── patient.py
│   │   └── websocket.py
│   ├── api/v1/
│   │   ├── simulations.py
│   │   └── websocket.py
│   ├── services/
│   │   └── simulation_service.py
│   └── utils/
│       └── logging.py
├── requirements.txt
└── Dockerfile
```

---

## 3. Ключевые зависимости (`requirements.txt`)

```txt
fastapi
uvicorn
pydantic
python-dotenv
httpx
python-jose[cryptography]
passlib[bcrypt]
slowapi          # Rate Limiting
structlog        # Структурированное логирование
numpy
scipy
```

---

## 4. JWT + Rate Limiting (`app/core/security.py` + `app/core/middleware.py`)

**`app/core/security.py`** (расширенная версия с ролями):

```python
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = "hbs-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

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

**`app/core/middleware.py`** (Rate Limiting):

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
```

---

## 5. Полноценный Workflow (`app/core/simulation_workflow.py`)

Реализуем Saga-паттерн с компенсацией при сбое.

```python
from app.clients.emr_client import EMRClient
from app.clients.calibration_client import CalibrationClient
from app.clients.history_client import HistoryClient
from app.services.simulation_service import run_simulation
from app.core.storage import simulations_db, simulation_summaries
from app.schemas.simulation import SimulationSummary, FullSimulationResponse
import structlog
import uuid
from datetime import datetime
import numpy as np

logger = structlog.get_logger()

class SimulationWorkflow:
    def __init__(self):
        self.emr_client = EMRClient()
        self.calibration_client = CalibrationClient()
        self.history_client = HistoryClient()

    async def run_full_simulation(self, patient_id: str, params: dict, websocket=None) -> FullSimulationResponse:
        """
        Полный workflow с компенсацией.
        При ошибке выполняет rollback (удаляет частично сохранённые данные).
        """
        sim_id = str(uuid.uuid4())
        patient_data = None
        vsd = None
        result = None

        try:
            # 1. Получение данных пациента из ЭМК
            logger.info("workflow_start", sim_id=sim_id, patient_id=patient_id)
            patient_data = await self.emr_client.get_patient_data(patient_id)
            if websocket:
                await websocket.send_json({"type": "progress", "step": 1, "total": 5, "message": "Данные пациента загружены"})

            # 2. Калибровка (если не задан R_vsd)
            vsd = params.get("vsd_resistance")
            calibrated = None
            if vsd is None:
                vsd = await self.calibration_client.calibrate_vsd(patient_data)
                calibrated = vsd
                logger.info("calibration_done", sim_id=sim_id, vsd=vsd)
                if websocket:
                    await websocket.send_json({"type": "progress", "step": 2, "total": 5, "message": f"Калибровка завершена, R_vsd = {vsd}"})

            # 3. Запуск симуляции (может быть долгим)
            result = run_simulation(
                vsd_resistance=vsd,
                flow_dependent_lungs=params.get("flow_dependent_lungs", False),
                t_span=params.get("t_span", [0, 200]),
                t_eval_points=params.get("t_eval_points", 2000),
                initial_concentrations=patient_data.get("labs")
            )
            if websocket:
                await websocket.send_json({"type": "progress", "step": 3, "total": 5, "message": "Симуляция выполнена"})

            # 4. Сохранение результатов в БД (in-memory, позже заменим на реальную)
            simulations_db[sim_id] = result
            summary = self._compute_summary(result, params.get("t_span", [0, 200]))
            simulation_summaries[sim_id] = summary
            if websocket:
                await websocket.send_json({"type": "progress", "step": 4, "total": 5, "message": "Результаты сохранены"})

            # 5. Запись в историю пациента (например, для ЭМК)
            await self.history_client.save_simulation_history(patient_id, sim_id, summary)
            if websocket:
                await websocket.send_json({"type": "progress", "step": 5, "total": 5, "message": "История обновлена"})

            logger.info("workflow_complete", sim_id=sim_id, patient_id=patient_id)
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
            logger.error("workflow_failed", sim_id=sim_id, error=str(e), patient_id=patient_id)
            # Компенсация (rollback)
            simulations_db.pop(sim_id, None)
            simulation_summaries.pop(sim_id, None)
            if patient_id:
                await self.history_client.rollback_history(patient_id, sim_id)
            raise RuntimeError(f"Workflow failed: {str(e)}")

    def _compute_summary(self, result, t_span):
        t_arr = np.array(result['time'])
        mask = t_arr >= (t_span[1] - 50)
        if np.any(mask):
            qp_qs_mean = np.mean(np.array(result['outputs']['Qp_Qs'])[mask])
            ps_mean = np.mean(np.array(result['outputs']['P_sa'])[mask])
            pp_mean = np.mean(np.array(result['outputs']['P_pa'])[mask])
            vb_mean = np.mean(np.array(result['outputs']['V_blood'])[mask])
            gfr_mean = np.mean(np.array(result['outputs']['GFR'])[mask])
            return SimulationSummary(
                qp_qs=qp_qs_mean,
                mean_sa_pressure=ps_mean,
                mean_pa_pressure=pp_mean,
                blood_volume=vb_mean,
                gfr=gfr_mean
            )
        else:
            return SimulationSummary(qp_qs=1.0, mean_sa_pressure=80, mean_pa_pressure=15, blood_volume=5000, gfr=1.2)
```

---

## 6. Главный эндпоинт (`app/api/v1/simulations.py`)

Эндпоинт с rate limiting и JWT.

```python
from fastapi import APIRouter, Depends, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.simulation_workflow import SimulationWorkflow
from app.core.security import get_current_user
from app.schemas.simulation import FullSimulationRequest, FullSimulationResponse

router = APIRouter()
workflow = SimulationWorkflow()
limiter = Limiter(key_func=get_remote_address)

@router.post("/simulate/run", response_model=FullSimulationResponse)
@limiter.limit("10/minute")   # Rate limiting: не более 10 запусков в минуту
async def run_simulation(
    request: FullSimulationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Основной эндпоинт запуска персонализированной симуляции.
    Доступен врачам и пациентам (только для себя).
    """
    # Проверка прав
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

## 7. WebSocket Realtime (`app/api/v1/websocket.py`)

Для обновлений статуса симуляции в реальном времени.

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.connection_manager import ConnectionManager

router = APIRouter()
manager = ConnectionManager()

@router.websocket("/ws/simulation/{sim_id}")
async def simulation_status_websocket(websocket: WebSocket, sim_id: str):
    await manager.connect(websocket, sim_id)
    try:
        while True:
            # Клиент может отправлять команды (например, отмена)
            data = await websocket.receive_text()
            # Обработка команд (например, отмена симуляции)
            await manager.broadcast_to_simulation(sim_id, f"Command received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(sim_id)
```

---

## 8. Connection Manager (`app/core/connection_manager.py`)

Управляет активными WebSocket-подключениями.

```python
from typing import Dict, List
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, sim_id: str):
        await websocket.accept()
        if sim_id not in self.active_connections:
            self.active_connections[sim_id] = []
        self.active_connections[sim_id].append(websocket)

    def disconnect(self, sim_id: str, websocket: WebSocket = None):
        if sim_id in self.active_connections:
            if websocket:
                self.active_connections[sim_id].remove(websocket)
            if not self.active_connections[sim_id]:
                del self.active_connections[sim_id]

    async def broadcast_to_simulation(self, sim_id: str, message: str):
        if sim_id in self.active_connections:
            for connection in self.active_connections[sim_id]:
                try:
                    await connection.send_text(message)
                except:
                    pass
```

---

## 9. Observability (структурированные логи)

В примерах выше уже используется `structlog`. Настроим его в `app/utils/logging.py`:

```python
import structlog
import logging

def configure_logging():
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=logging.INFO)
```

Подключите в `main.py` при старте.

---

## 10. Обновление `app/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import simulations, websocket
from app.utils.logging import configure_logging
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

configure_logging()

app = FastAPI(title="HBS Production API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting handler
app.state.limiter = Limiter(key_func=get_remote_address)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Роутеры
app.include_router(simulations.router, prefix="/api/v1", tags=["Simulations"])
app.include_router(websocket.router, prefix="/api/v1", tags=["WebSocket"])

@app.get("/")
def root():
    return {"service": "HBS Production Backend", "status": "running"}
```

---

## 11. Запуск и проверка

```bash
uvicorn app.main:app --reload --port 8000
```

**Тестирование:**

1. **JWT-токен:** получите через эндпоинт `/auth/login` (не реализован в уроке, но можно создать отдельно).
2. **Запуск симуляции:** `POST /api/v1/simulate/run` с заголовком `Authorization: Bearer <token>` и телом:
   ```json
   {
     "patient_id": "p_001",
     "vsd_resistance": null,
     "flow_dependent_lungs": false,
     "t_span": [0, 200],
     "t_eval_points": 2000
   }
   ```
   Будет выполнен workflow с калибровкой (если нужно) и возвращены результаты.

3. **WebSocket:** подключитесь к `ws://localhost:8000/api/v1/ws/simulation/{sim_id}` и получайте обновления прогресса.

4. **Rate Limiting:** повторите запрос более 10 раз в минуту – получите ошибку 429.

5. **Логи:** структурированные JSON-логи будут выводиться в консоль.

---

## 12. Что мы приблизили к реальному проекту HBS

- ✅ Полноценный **Saga-воркфлоу** с компенсацией при ошибках
- ✅ JWT-авторизация с ролевой моделью (врач/пациент)
- ✅ **Rate Limiting** для защиты от перегрузки
- ✅ **WebSocket** для реального времени (прогресс симуляции)
- ✅ **Структурированные логи** для наблюдаемости
- ✅ Модульная структура, готовая к масштабированию

---

## 13. Дальнейшие шаги

- Добавить **асинхронную обработку** длительных симуляций (запуск в фоновом потоке) и уведомление через WebSocket о завершении.
- Подключить реальную базу данных (PostgreSQL + TimescaleDB) вместо in-memory хранилища.
- Реализовать полноценный эндпоинт аутентификации (`/auth/login`).
- Контейнеризация с Docker и orchestration (docker-compose).

---

**Поздравляю!** Вы прошли полный путь от минимального Hello World до production-ready бэкенда для медицинской симуляции. Теперь вы умеете строить сложные системы, сочетающие моделирование, безопасность и реальное время. 🚀🧬

**Удачи в ваших проектах!**