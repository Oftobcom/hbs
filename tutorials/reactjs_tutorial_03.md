# React.js Tutorial 03 — Интеграция с бэкендом HBS (FastAPI) и визуализация результатов

Это **третий** урок серии по React.js в проекте **Human Body Simulation (HBS)**.  
Мы подключим наше приложение к реальному бэкенду на FastAPI (который уже умеет запускать симуляцию `WholeBodyModel`), заменим заглушки на реальные вызовы API, добавим интерактивные графики и таблицу установившихся значений.

Цель — получить полностью работающий веб-интерфейс для настройки и запуска симуляции физиологии человека, отображения временных рядов и сводных показателей.

---

## 1. Цель урока

- Научиться отправлять **POST-запросы** к FastAPI бэкенду из React (axios)
- Обрабатывать асинхронные ответы, ошибки, состояния загрузки
- Визуализировать **временные ряды** с помощью `recharts`
- Вычислять и отображать **установившиеся значения** (средние за последние N секунд)
- Подготовить приложение к развёртыванию вместе с бэкендом (CORS, proxy)

---

## 2. Установка зависимостей

```bash
cd hbs/frontend/hbs-dashboard
npm install axios recharts
```

- `axios` — для HTTP-запросов
- `recharts` — библиотека графиков на основе D3

---

## 3. Запуск бэкенда (FastAPI)

Убедитесь, что бэкенд из предыдущих уроков (`fastapi_tutorial_03/04/05`) запущен:

```bash
cd hbs/backend
uvicorn app.main:app --reload --port 8000
```

**Проверьте** доступность эндпоинта `/simulate` (или `/api/v1/simulate`) через браузер или `curl`.  
Если эндпоинт ещё не реализован — создайте минимальный:

```python
# app/main.py (пример)
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class SimulateRequest(BaseModel):
    vsd_resistance: float = float('inf')
    flow_dependent_lungs: bool = False
    t_span: list = [0, 200]

@app.post("/simulate")
async def simulate(req: SimulateRequest):
    # Здесь должна быть ваша полноценная симуляция (whole_body.py)
    # Пока вернём тестовые данные, но в реальном проекте вызывайте WholeBodyModel
    import numpy as np
    t = np.linspace(0, 200, 1000)
    # mock-данные
    return {
        "time": t.tolist(),
        "outputs": {
            "P_sa": (80 + 20 * np.sin(t * 0.5)).tolist(),
            "P_pa": (15 + 5 * np.sin(t * 0.7)).tolist(),
            "Q_aortic": (80 + 10 * np.sin(t * 0.5)).tolist(),
            "Qp_Qs": [1.0 if req.vsd_resistance == float('inf') else 2.5] * len(t),
        }
    }
```

**Важно:** позже вы замените mock на реальный вызов `WholeBodyModel.simulate()`.

---

## 4. Обновляем API-клиент (`src/services/api.js`)

Заменим заглушку на реальный axios-запрос.

```javascript
// src/services/api.js
import axios from 'axios';

// Базовый URL бэкенда (при разработке используйте proxy или полный URL)
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const runSimulation = async (params) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/simulate`, params, {
      timeout: 60000, // симуляция может быть долгой (60 сек)
    });
    return response.data; // { time: [...], outputs: {...} }
  } catch (error) {
    console.error('Ошибка при вызове симуляции:', error);
    throw error;
  }
};
```

Создайте файл `.env` в корне проекта:
```
REACT_APP_API_URL=http://localhost:8000
```

---

## 5. Компонент графиков (`src/components/TimeSeriesChart.js`)

Создадим универсальный компонент для отображения нескольких линий.

```jsx
// src/components/TimeSeriesChart.js
import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

const TimeSeriesChart = ({ data, lines, xKey = 'time', title }) => {
  if (!data || data.length === 0) return <div>Нет данных для графика</div>;

  return (
    <div style={{ marginBottom: '2rem' }}>
      <h4>{title}</h4>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={xKey} label={{ value: 'Время (с)', position: 'insideBottomRight', offset: -5 }} />
          <YAxis />
          <Tooltip />
          <Legend />
          {lines.map((line) => (
            <Line
              key={line.key}
              type="monotone"
              dataKey={line.key}
              stroke={line.color}
              name={line.name}
              dot={false}
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TimeSeriesChart;
```

---

## 6. Компонент таблицы установившихся значений (`src/components/SteadyStateTable.js`)

Будет вычислять средние значения за последние `t_start` секунд.

```jsx
// src/components/SteadyStateTable.js
import React, { useMemo } from 'react';

const SteadyStateTable = ({ time, outputs, tStart = 150 }) => {
  const steadyState = useMemo(() => {
    if (!time || !outputs || time.length === 0) return null;

    // Находим индекс, начиная с которого время >= tStart
    const startIdx = time.findIndex(t => t >= tStart);
    if (startIdx === -1) return null;

    const metrics = [
      { key: 'P_sa', name: 'АД сист., мм рт. ст.', unit: 'мм рт. ст.', decimals: 0 },
      { key: 'P_pa', name: 'АД лёг., мм рт. ст.', unit: 'мм рт. ст.', decimals: 0 },
      { key: 'Q_aortic', name: 'Сист. выброс, мл/с', unit: 'мл/с', decimals: 1 },
      { key: 'Q_pulmonary', name: 'Лёг. кровоток, мл/с', unit: 'мл/с', decimals: 1 },
      { key: 'Qp_Qs', name: 'Qp/Qs', unit: '', decimals: 2 },
      { key: 'Q_vsd', name: 'Шунт VSD, мл/с', unit: 'мл/с', decimals: 1 },
      { key: 'V_lv', name: 'Объём ЛЖ, мл', unit: 'мл', decimals: 1 },
      { key: 'V_blood', name: 'Объём крови, мл', unit: 'мл', decimals: 0 },
      { key: 'C_bilirubin_blood', name: 'Билирубин, у.е./мл', unit: 'у.е./мл', decimals: 3 },
      { key: 'C_ammonia_blood', name: 'Аммиак, у.е./мл', unit: 'у.е./мл', decimals: 3 },
      { key: 'GFR', name: 'СКФ, мл/с', unit: 'мл/с', decimals: 2 },
    ];

    const result = {};
    for (const metric of metrics) {
      const values = outputs[metric.key]?.slice(startIdx) || [];
      if (values.length > 0) {
        const mean = values.reduce((a,b) => a + b, 0) / values.length;
        result[metric.key] = mean.toFixed(metric.decimals);
      } else {
        result[metric.key] = '—';
      }
    }
    return { result, metrics };
  }, [time, outputs, tStart]);

  if (!steadyState) return <div>Недостаточно данных для расчёта установившихся значений (t ≥ {tStart} с)</div>;

  return (
    <div>
      <h4>📊 Установившиеся значения (t ≥ {tStart} с)</h4>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #ccc' }}>
            <th style={{ textAlign: 'left', padding: '8px' }}>Показатель</th>
            <th style={{ textAlign: 'right', padding: '8px' }}>Значение</th>
          </tr>
        </thead>
        <tbody>
          {steadyState.metrics.map(metric => (
            <tr key={metric.key} style={{ borderBottom: '1px solid #eee' }}>
              <td style={{ padding: '8px' }}>{metric.name}</td>
              <td style={{ textAlign: 'right', padding: '8px' }}>
                {steadyState.result[metric.key]} {metric.unit}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default SteadyStateTable;
```

---

## 7. Обновляем хук `useSimulation.js` для работы с реальными данными

```javascript
// src/hooks/useSimulation.js
import { useState } from 'react';
import { runSimulation } from '../services/api';

const useSimulation = () => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null); // { time, outputs }
  const [error, setError] = useState(null);

  const simulate = async (params) => {
    setLoading(true);
    setError(null);
    try {
      const data = await runSimulation(params);
      setResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Ошибка при выполнении симуляции');
    } finally {
      setLoading(false);
    }
  };

  return { loading, results, error, simulate };
};

export default useSimulation;
```

---

## 8. Обновляем `ResultsPreview.js` — теперь с графиками и таблицей

```jsx
// src/components/ResultsPreview.js
import React from 'react';
import TimeSeriesChart from './TimeSeriesChart';
import SteadyStateTable from './SteadyStateTable';

const ResultsPreview = ({ results, loading, error }) => {
  if (loading) return <div>⏳ Выполняется симуляция (это может занять до 60 секунд)...</div>;
  if (error) return <div className="error">❌ Ошибка: {error}</div>;
  if (!results || !results.time || results.time.length === 0) {
    return <div>Запустите симуляцию, чтобы увидеть результаты.</div>;
  }

  // Подготовим данные для графиков: массив объектов {time, P_sa, P_pa, ...}
  const chartData = results.time.map((t, idx) => {
    const point = { time: t };
    for (const [key, values] of Object.entries(results.outputs)) {
      point[key] = values[idx];
    }
    return point;
  });

  return (
    <div>
      <h3>📈 Результаты симуляции</h3>
      <TimeSeriesChart
        data={chartData}
        lines={[
          { key: 'P_sa', name: 'АД сист. (мм рт. ст.)', color: '#3b82f6' },
          { key: 'P_pa', name: 'АД лёг. (мм рт. ст.)', color: '#ef4444' },
        ]}
        title="Артериальное давление"
      />
      <TimeSeriesChart
        data={chartData}
        lines={[
          { key: 'Q_aortic', name: 'Сист. выброс (мл/с)', color: '#10b981' },
          { key: 'Q_pulmonary', name: 'Лёг. кровоток (мл/с)', color: '#f59e0b' },
          { key: 'Q_vsd', name: 'Шунт (мл/с)', color: '#8b5cf6' },
        ]}
        title="Кровотоки"
      />
      <TimeSeriesChart
        data={chartData}
        lines={[
          { key: 'C_bilirubin_blood', name: 'Билирубин (у.е./мл)', color: '#ec489a' },
          { key: 'C_ammonia_blood', name: 'Аммиак (у.е./мл)', color: '#14b8a6' },
        ]}
        title="Метаболиты"
      />
      <SteadyStateTable time={results.time} outputs={results.outputs} tStart={150} />
    </div>
  );
};

export default ResultsPreview;
```

---

## 9. Обновляем `Dashboard.js` для передачи error в ResultsPreview

```jsx
// src/pages/Dashboard.js
import React, { useState } from 'react';
import PatientInfo from '../components/PatientInfo';
import SimulationPanel from '../components/SimulationPanel';
import ResultsPreview from '../components/ResultsPreview';
import useSimulation from '../hooks/useSimulation';

// mock-данные пациента (позже будут из ЭМК)
const mockPatient = { /* ... как в прошлом уроке ... */ };

const Dashboard = () => {
  const { loading, results, error, simulate } = useSimulation();
  const [lastParams, setLastParams] = useState(null);

  const handleSimulate = (params) => {
    setLastParams(params);
    simulate(params);
  };

  return (
    <div className="dashboard">
      <div className="dashboard-left">
        <PatientInfo patient={mockPatient} />
        <SimulationPanel onSimulate={handleSimulate} />
      </div>
      <div className="dashboard-right">
        <ResultsPreview results={results} loading={loading} error={error} />
        {lastParams && !loading && results && (
          <div className="info">
            Симуляция выполнена с параметрами: R_vsd = {lastParams.vsd_resistance === Infinity ? '∞' : lastParams.vsd_resistance}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
```

---

## 10. Настройка прокси для разработки (чтобы избежать CORS)

В файле `package.json` добавьте:

```json
"proxy": "http://localhost:8000"
```

После этого можно в `api.js` использовать относительные URL: `'/simulate'` вместо полного.  
**Важно:** перезапустите `npm start` после изменения.

---

## 11. Запуск и тестирование

1. Убедитесь, что бэкенд FastAPI запущен на порту 8000.
2. В терминале фронтенда:
   ```bash
   npm start
   ```
3. Откройте `http://localhost:3000`, выберите параметры, нажмите «Запустить симуляцию».
4. Через несколько секунд должны появиться графики и таблица.

---

## 12. Что мы сделали для реального проекта HBS

- ✅ Полноценная интеграция React ↔ FastAPI (axios)
- ✅ Отображение временных рядов с помощью `recharts`
- ✅ Расчёт установившихся значений (средние по хвосту временного ряда)
- ✅ Обработка ошибок и длительных запросов
- ✅ Модульная структура готова к расширению
