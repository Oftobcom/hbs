# HBS FastAPI Tutorial 01 — Минимальный бэкенд для Human Body Simulation

Это **первый** урок по FastAPI в контексте проекта **Human Body Simulation (HBS)**.  
Мы создадим минимальный веб-сервер, который станет основой для будущего бэкенда, интегрирующего симуляцию физиологии человека с электронной медицинской картой (ЭМК) и React.js интерфейсом.

Цель — быстро запустить API, чтобы в следующих уроках подключить к нему реальные модели органов, расчёты гемодинамики и передачу данных на фронтенд.

---

## 1. Установка зависимостей

```bash
pip install fastapi uvicorn
```

---

## 2. Код (`main.py`)

Создайте файл `main.py` со следующим содержимым:

```python
from fastapi import FastAPI

app = FastAPI(
    title="HBS Backend API",
    description="Минимальный API для Human Body Simulation",
    version="0.1.0"
)

@app.get("/")
@app.get("/hello")
def hello():
    return {"message": "HBS API is running"}

@app.get("/status")
def status():
    return {
        "service": "HBS Backend",
        "status": "healthy",
        "version": "0.1.0",
        "components": ["heart", "lungs", "liver", "kidney", "blood", "brain"]
    }
```

---

## 3. Запуск

```bash
uvicorn main:app --reload --port 8000
```

После запуска сервер будет доступен по адресу:  
`http://localhost:8000`

---

## 4. Проверка через Postman / curl

- **Метод:** `GET`
- **URL:** `http://localhost:8000/hello` (или `http://localhost:8000/`)

**Ответ:**

```json
{
  "message": "HBS API is running"
}
```

Для проверки статуса:

- **Метод:** `GET`
- **URL:** `http://localhost:8000/status`

**Ответ:**

```json
{
  "service": "HBS Backend",
  "status": "healthy",
  "version": "0.1.0",
  "components": ["heart", "lungs", "liver", "kidney", "blood", "brain"]
}
```

---

## 5. Альтернативный вариант с POST (приём данных пациента)

Если нужно передать данные пациента для будущей симуляции:

```python
from pydantic import BaseModel

class PatientData(BaseModel):
    patient_id: str
    name: str
    age: int
    diagnosis: str

@app.post("/patient")
def register_patient(data: PatientData):
    return {
        "message": f"Пациент {data.name} зарегистрирован",
        "patient_id": data.patient_id,
        "status": "ready_for_simulation"
    }
```

**Проверка в Postman:**
- Метод `POST`, URL `http://localhost:8000/patient`
- Вкладка **Body** → **raw** → **JSON**
```json
{
  "patient_id": "p_001",
  "name": "Иванов Иван Иванович",
  "age": 54,
  "diagnosis": "Дефект межжелудочковой перегородки (ДМЖП)"
}
```

**Ответ:**
```json
{
  "message": "Пациент Иванов Иван Иванович зарегистрирован",
  "patient_id": "p_001",
  "status": "ready_for_simulation"
}
```

---

## 6. Что дальше?

В следующих уроках (`hbs_fastapi_tutorial_02.md` и далее) мы:
- Подключим реальную модель `WholeBodyModel` из Python-скриптов
- Создадим эндпоинт `/simulate`, который принимает параметры (сопротивление ДМЖП, длительность) и возвращает временные ряды
- Добавим поддержку CORS для взаимодействия с React.js приложением
- Настроим WebSocket для потоковой передачи промежуточных результатов
- Интегрируем чтение данных из ЭМК (mock или реальный API)

---

## Готово!

Вы создали работающий бэкенд для HBS. Теперь вы можете запускать его и проверять через Postman.  
Это основа, на которой мы построим всю систему моделирования физиологии человека с веб-интерфейсом.

**Удачи в разработке HBS!** 🧬🚀