# React.js Tutorial 04 — Расширенный UX: WebSocket, сохранение сессий, экспорт данных и интеграция с ЭМК

Это **четвёртый** урок серии по React.js в проекте **Human Body Simulation (HBS)**.  
Мы добавим реальные интерактивные возможности, необходимые для клинического использования:

- **WebSocket** для отслеживания прогресса длительной симуляции
- Возможность **сохранять / загружать** сценарии симуляции (localStorage / IndexedDB)
- **Экспорт графиков в PNG** и результатов в CSV
- **Интеграция с ЭМК** (чтение данных пациента из внешнего API или JSON)
- **Отмена симуляции** и улучшенная обработка ошибок

---

## 1. Цель урока

- Научиться использовать **WebSocket** в React для получения промежуточных обновлений от бэкенда
- Реализовать сохранение параметров и результатов симуляции на клиенте
- Добавить кнопки экспорта данных (CSV) и графиков (PNG)
- Связать интерфейс с реальным источником данных пациента (имитация ЭМК)
- Улучшить пользовательский опыт при длительных расчётах

---

## 2. Установка дополнительных зависимостей

```bash
cd hbs/frontend/hbs-dashboard
npm install html-to-image file-saver
```

- `html-to-image` — для сохранения графиков в PNG
- `file-saver` — для скачивания файлов

---

## 3. Подготовка бэкенда: эндпоинт WebSocket (пример на FastAPI)

В бэкенде добавьте WebSocket поддержку (это не обязательно для урока, но нужно для полноты).  
Если у вас ещё нет WebSocket в FastAPI, создайте минимальный эндпоинт:

```python
# app/main.py дополнение
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/simulate")
async def websocket_simulate(websocket: WebSocket):
    await websocket.accept()
    try:
        # Принять параметры от клиента
        params = await websocket.receive_json()
        # Здесь запускается симуляция с периодической отправкой прогресса
        total_steps = 100
        for i in range(total_steps):
            # ... выполнить шаг симуляции ...
            await websocket.send_json({"progress": i + 1, "total": total_steps})
            # можно передавать частичные результаты
        # финальный результат
        await websocket.send_json({"status": "complete", "results": final_results})
    except WebSocketDisconnect:
        print("Client disconnected")
```

Для простоты урока мы сохраним обычный HTTP POST-запрос, но добавим **симуляцию прогресса** на фронтенде (или можно использовать WebSocket позже). В коде ниже будет два варианта: с обычным `setTimeout` прогресс-баром и реальным WebSocket по желанию.

---

## 4. Обновляем хук `useSimulation.js` — добавляем прогресс и отмену

Теперь хук будет поддерживать состояние `progress` и `abortController` для возможности отмены запроса.

```javascript
// src/hooks/useSimulation.js
import { useState, useRef } from 'react';
import { runSimulation } from '../services/api';

const useSimulation = () => {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const abortControllerRef = useRef(null);

  const simulate = async (params, onProgress) => {
    // Отменяем предыдущий запрос, если есть
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setLoading(true);
    setProgress(0);
    setError(null);
    setResults(null);

    try {
      // Имитация прогресса (можно заменить на реальные события от WebSocket)
      let progressInterval;
      if (onProgress) {
        progressInterval = setInterval(() => {
          setProgress(prev => {
            if (prev >= 90) return prev;
            return prev + 10;
          });
        }, 500);
      }

      const data = await runSimulation(params, abortControllerRef.current.signal);
      
      if (progressInterval) clearInterval(progressInterval);
      setProgress(100);
      setResults(data);
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Симуляция отменена пользователем');
      } else {
        setError(err.response?.data?.detail || err.message || 'Ошибка при выполнении симуляции');
      }
    } finally {
      setLoading(false);
      setProgress(0);
      abortControllerRef.current = null;
    }
  };

  const cancelSimulation = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setLoading(false);
      setProgress(0);
    }
  };

  return { loading, progress, results, error, simulate, cancelSimulation };
};

export default useSimulation;
```

---

## 5. Обновляем `api.js` для поддержки AbortController

```javascript
// src/services/api.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const runSimulation = async (params, signal) => {
  const response = await axios.post(`${API_BASE_URL}/simulate`, params, {
    signal,           // для возможности отмены
    timeout: 120000,  // 2 минуты
  });
  return response.data;
};
```

---

## 6. Компонент отображения прогресса (`ProgressBar.js`)

```jsx
// src/components/ProgressBar.js
import React from 'react';

const ProgressBar = ({ progress, onCancel }) => {
  if (progress === 0 || progress === 100) return null;

  return (
    <div style={{ marginTop: '1rem' }}>
      <div style={{ background: '#e2e8f0', borderRadius: '8px', overflow: 'hidden' }}>
        <div
          style={{
            width: `${progress}%`,
            background: '#3b82f6',
            height: '8px',
            transition: 'width 0.3s ease'
          }}
        />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem' }}>
        <span>Выполнение: {progress}%</span>
        <button onClick={onCancel} style={{ background: '#ef4444' }}>Отменить</button>
      </div>
    </div>
  );
};

export default ProgressBar;
```

---

## 7. Экспорт данных (CSV и PNG)

### 7.1. Создаём утилиты для экспорта

**`src/utils/exportUtils.js`**
```javascript
import { saveAs } from 'file-saver';
import { toPng } from 'html-to-image';

export const exportToCSV = (time, outputs, filename = 'simulation_results.csv') => {
  // Собираем заголовки
  const keys = ['time', ...Object.keys(outputs)];
  const rows = time.map((t, idx) => {
    const row = { time: t };
    for (const key of Object.keys(outputs)) {
      row[key] = outputs[key][idx];
    }
    return row;
  });

  const csvContent = [
    keys.join(','),
    ...rows.map(row => keys.map(k => row[k]).join(','))
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  saveAs(blob, filename);
};

export const exportChartAsPNG = (elementId, filename = 'chart.png') => {
  const node = document.getElementById(elementId);
  if (!node) return;
  toPng(node)
    .then(dataUrl => {
      const link = document.createElement('a');
      link.download = filename;
      link.href = dataUrl;
      link.click();
    })
    .catch(err => console.error('Ошибка сохранения PNG', err));
};
```

### 7.2. Добавляем кнопки в `ResultsPreview.js`

Обновим компонент, добавив экспорт и передачу `results` в экспортные функции.

```jsx
// src/components/ResultsPreview.js (фрагмент)
import { exportToCSV, exportChartAsPNG } from '../utils/exportUtils';

// Внутри компонента после отрисовки графиков добавим:
<div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
  <button onClick={() => exportToCSV(results.time, results.outputs)}>
    📥 Скачать CSV
  </button>
  <button onClick={() => exportChartAsPNG('chart-pressure', 'pressure.png')}>
    📸 Сохранить график давления
  </button>
  <button onClick={() => exportChartAsPNG('chart-flows', 'flows.png')}>
    📸 Сохранить график потоков
  </button>
</div>

// Каждому графику TimeSeriesChart нужно добавить id:
<TimeSeriesChart id="chart-pressure" ... />
```

Измените `TimeSeriesChart`, чтобы он принимал `id` и передавал его в контейнер.

---

## 8. Интеграция с ЭМК (получение реальных данных пациента)

Создадим сервис для загрузки данных пациента. Пока используем JSON-файл или заглушку, но можно заменить на реальный API.

**`src/services/emrApi.js`**
```javascript
import axios from 'axios';

const EMR_API_URL = process.env.REACT_APP_EMR_URL || 'http://localhost:8001/patients';

export const fetchPatientData = async (patientId) => {
  // Имитация запроса к ЭМК
  // В реальности: return axios.get(`${EMR_API_URL}/${patientId}`).then(res => res.data);
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        id: patientId,
        name: 'Иванов Иван Иванович',
        age: 54,
        diagnosis: 'Дефект межжелудочковой перегородки (ДМЖП), компенсированный',
        admissionDate: '2025-03-10',
        labs: {
          bilirubin: 18.2,
          ammonia: 45.0,
          albumin: 3.2,
          toxins: 2.1
        }
      });
    }, 500);
  });
};
```

### 8.1. Компонент выбора пациента (`PatientSelector.js`)

```jsx
// src/components/PatientSelector.js
import React, { useState, useEffect } from 'react';
import { fetchPatientData } from '../services/emrApi';

const PatientSelector = ({ onPatientSelect }) => {
  const [patientId, setPatientId] = useState('p_001');
  const [loading, setLoading] = useState(false);

  const handleLoad = async () => {
    setLoading(true);
    try {
      const data = await fetchPatientData(patientId);
      onPatientSelect(data);
    } catch (err) {
      alert('Ошибка загрузки данных пациента');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginBottom: '1rem' }}>
      <label>
        ID пациента:
        <input value={patientId} onChange={(e) => setPatientId(e.target.value)} />
      </label>
      <button onClick={handleLoad} disabled={loading}>
        {loading ? 'Загрузка...' : 'Загрузить данные из ЭМК'}
      </button>
    </div>
  );
};

export default PatientSelector;
```

### 8.2. Обновляем `Dashboard.js` для динамической загрузки пациента

Добавим состояние `patient` и передадим его в `PatientInfo`. Также сохраняем выбранные параметры симуляции в `localStorage` для восстановления сессии.

```jsx
// src/pages/Dashboard.js (фрагмент)
import React, { useState, useEffect } from 'react';
import PatientSelector from '../components/PatientSelector';
import PatientInfo from '../components/PatientInfo';
import SimulationPanel from '../components/SimulationPanel';
import ResultsPreview from '../components/ResultsPreview';
import ProgressBar from '../components/ProgressBar';
import useSimulation from '../hooks/useSimulation';

const Dashboard = () => {
  const { loading, progress, results, error, simulate, cancelSimulation } = useSimulation();
  const [patient, setPatient] = useState(null);
  const [lastParams, setLastParams] = useState(() => {
    const saved = localStorage.getItem('hbs_last_params');
    return saved ? JSON.parse(saved) : null;
  });

  useEffect(() => {
    // Автоматическая загрузка тестового пациента при старте
    fetchPatientData('p_001').then(setPatient);
  }, []);

  const handleSimulate = (params) => {
    setLastParams(params);
    localStorage.setItem('hbs_last_params', JSON.stringify(params));
    simulate(params);
  };

  return (
    <div className="dashboard">
      <div className="dashboard-left">
        <PatientSelector onPatientSelect={setPatient} />
        <PatientInfo patient={patient} />
        <SimulationPanel onSimulate={handleSimulate} initialParams={lastParams} />
        <ProgressBar progress={progress} onCancel={cancelSimulation} />
      </div>
      <div className="dashboard-right">
        <ResultsPreview results={results} loading={loading} error={error} />
      </div>
    </div>
  );
};
```

---

## 9. Сохранение сессий симуляции (история)

Создадим простой хук `useHistory` для сохранения и загрузки результатов.

**`src/hooks/useHistory.js`**
```javascript
import { useState, useEffect } from 'react';

const STORAGE_KEY = 'hbs_simulation_history';

export const useHistory = () => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) setHistory(JSON.parse(stored));
  }, []);

  const saveToHistory = (params, results) => {
    const newEntry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      params,
      summary: {
        qp_qs: results?.outputs?.Qp_Qs?.[results.outputs.Qp_Qs.length - 1],
        mean_sa: results?.outputs?.P_sa?.slice(-50).reduce((a,b)=>a+b,0)/50,
      }
    };
    const updated = [newEntry, ...history].slice(0, 20); // не более 20 записей
    setHistory(updated);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem(STORAGE_KEY);
  };

  return { history, saveToHistory, clearHistory };
};
```

В `Dashboard` после получения `results` вызываем `saveToHistory`.

---

## 10. Запуск и проверка

1. Убедитесь, что бэкенд FastAPI работает на порту 8000.
2. Запустите фронтенд: `npm start`
3. Проверьте:
   - Загрузка данных пациента через кнопку (или автоматически)
   - Выбор параметров симуляции, запуск
   - Появление прогресс-бара, возможность отмены
   - Отображение графиков и таблицы
   - Экспорт CSV и PNG
   - После завершения симуляции запись появляется в истории (можно вывести список)

---

## 11. Что мы приблизили к реальному клиническому использованию

- ✅ Пользователь может **отменить** долгий расчёт
- ✅ Сохраняются **последние параметры** и **история сессий**
- ✅ Данные пациента загружаются из **внешнего источника** (ЭМК)
- ✅ Результаты можно **экспортировать** для отчёта или анализа
- ✅ Визуализация прогресса снижает тревожность врача
