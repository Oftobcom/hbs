# React.js Tutorial 05 — Production-Ready HBS: Сравнение сценариев, калибровка модели и развёртывание

Это **заключительный** урок серии по React.js в проекте **Human Body Simulation (HBS)**.  
Мы создадим **полноценное клиническое приложение**, которое позволяет врачу не только запускать симуляцию, но и сравнивать несколько сценариев, калибровать модель под конкретного пациента, а также развернуть приложение в production вместе с бэкендом.

Цель — получить **законченный продукт**, готовый к использованию в реальной клинической практике.

---

## 1. Цель урока

- Реализовать **сравнение сценариев** (здоровый, малый ДМЖП, большой ДМЖП, послеоперационный) на одном графике
- Добавить **калибровку модели** — подбор параметров по клиническим данным пациента
- Использовать **WebSocket** для потоковой передачи промежуточных результатов (чтобы графики строились в реальном времени)
- Настроить **production-сборку** и **Docker-контейнеризацию**
- Добавить **аутентификацию** (JWT) и ролевую модель (врач / пациент)
- Подготовить **документацию** для пользователя

---

## 2. Установка дополнительных зависимостей

```bash
cd hbs/frontend/hbs-dashboard
npm install react-router-dom recharts html-to-image file-saver
# для WebSocket (в браузере уже есть)
# для аутентификации:
npm install jwt-decode
```

---

## 3. WebSocket для потоковой симуляции

### 3.1. Создаём хук `useWebSocketSimulation`

```jsx
// src/hooks/useWebSocketSimulation.js
import { useState, useRef, useEffect } from 'react';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws/simulate';

export const useWebSocketSimulation = () => {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [partialResults, setPartialResults] = useState(null);
  const [finalResults, setFinalResults] = useState(null);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);

  const startSimulation = (params) => {
    setLoading(true);
    setProgress(0);
    setPartialResults(null);
    setFinalResults(null);
    setError(null);

    wsRef.current = new WebSocket(WS_URL);
    wsRef.current.onopen = () => {
      wsRef.current.send(JSON.stringify({ type: 'simulate', params }));
    };
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'progress') {
        setProgress(data.progress);
      } else if (data.type === 'partial') {
        // Добавляем новую точку к временному ряду
        setPartialResults(prev => {
          const newTime = [...(prev?.time || []), data.time];
          const newOutputs = { ...prev?.outputs };
          for (const key of Object.keys(data.outputs)) {
            newOutputs[key] = [...(prev?.outputs?.[key] || []), data.outputs[key]];
          }
          return { time: newTime, outputs: newOutputs };
        });
      } else if (data.type === 'complete') {
        setFinalResults(data.results);
        setLoading(false);
        wsRef.current.close();
      }
    };
    wsRef.current.onerror = (err) => {
      setError('WebSocket error');
      setLoading(false);
    };
    wsRef.current.onclose = () => {};
  };

  const cancelSimulation = () => {
    if (wsRef.current) wsRef.current.close();
    setLoading(false);
  };

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return { loading, progress, partialResults, finalResults, error, startSimulation, cancelSimulation };
};
```

### 3.2. Обновляем `ResultsPreview` для отображения частичных результатов

Теперь можно показывать графики, которые обновляются по мере поступления новых точек.  
Реализуйте анимацию обновления или просто перерисовку.

---

## 4. Сравнение сценариев

Создадим компонент `ScenarioComparison`, который запускает несколько симуляций последовательно или параллельно и отображает их на одном графике.

### 4.1. Компонент `ScenarioComparison.js`

```jsx
// src/components/ScenarioComparison.js
import React, { useState } from 'react';
import { runSimulation } from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const SCENARIOS = {
  healthy: { name: 'Здоровый', params: { vsd_resistance: Infinity, flow_dependent_lungs: false } },
  small_vsd: { name: 'Малый ДМЖП', params: { vsd_resistance: 5.0, flow_dependent_lungs: false } },
  large_vsd: { name: 'Большой ДМЖП', params: { vsd_resistance: 1.0, flow_dependent_lungs: true } },
  post_op: { name: 'После операции', params: { vsd_resistance: Infinity, flow_dependent_lungs: false } }
};

const ScenarioComparison = ({ metric = 'P_sa', label = 'АД сист., мм рт. ст.' }) => {
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState(false);

  const runAll = async () => {
    setLoading(true);
    const newResults = {};
    for (const [key, scenario] of Object.entries(SCENARIOS)) {
      try {
        const data = await runSimulation(scenario.params);
        newResults[key] = data;
      } catch (err) {
        console.error(`Ошибка сценария ${key}`, err);
      }
    }
    setResults(newResults);
    setLoading(false);
  };

  // Преобразование в формат для recharts: { time, healthy_P_sa, small_vsd_P_sa, ... }
  const prepareChartData = () => {
    const firstKey = Object.keys(results)[0];
    if (!firstKey) return [];
    const times = results[firstKey].time;
    return times.map((t, idx) => {
      const point = { time: t };
      for (const [scenarioKey, scenarioData] of Object.entries(results)) {
        point[`${scenarioKey}_${metric}`] = scenarioData.outputs[metric]?.[idx];
      }
      return point;
    });
  };

  const lines = Object.keys(results).map(scenarioKey => ({
    key: `${scenarioKey}_${metric}`,
    name: SCENARIOS[scenarioKey]?.name || scenarioKey,
    color: scenarioKey === 'healthy' ? '#10b981' : scenarioKey === 'small_vsd' ? '#f59e0b' : '#ef4444'
  }));

  return (
    <div>
      <button onClick={runAll} disabled={loading}>Сравнить все сценарии</button>
      {loading && <p>Запуск симуляций...</p>}
      {Object.keys(results).length > 0 && (
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={prepareChartData()}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" label="Время, с" />
            <YAxis label={label} />
            <Tooltip />
            <Legend />
            {lines.map(line => <Line key={line.key} type="monotone" dataKey={line.key} stroke={line.color} dot={false} />)}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};
```

### 4.2. Добавляем вкладку "Сравнение" в Dashboard

Используйте `react-router-dom` для навигации между основным дашбордом и сравнением.

---

## 5. Калибровка модели под пациента

Создадим компонент `ModelCalibration`, который позволяет подобрать параметры модели (например, `R_vsd`, `E_max_LV`, системное сопротивление) так, чтобы результаты симуляции соответствовали измеренным клиническим данным пациента (АД, сердечный выброс, Qp/Qs).

### 5.1. Простая оптимизация на клиенте (или через бэкенд)

В реальном проекте лучше выполнять калибровку на бэкенде (scipy.optimize), но для демонстрации можно сделать минимальную версию.

```jsx
// src/components/ModelCalibration.js
import React, { useState } from 'react';
import { runSimulation } from '../services/api';

const ModelCalibration = ({ patientMeasurements, onCalibrated }) => {
  const [vsdResistance, setVsdResistance] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runCalibration = async () => {
    setLoading(true);
    // Простейший поиск: перебираем несколько значений R_vsd
    const candidates = [1, 2, 5, 10, 20, Infinity];
    let best = null;
    let bestError = Infinity;
    for (const r of candidates) {
      const result = await runSimulation({ vsd_resistance: r, flow_dependent_lungs: false });
      const simulatedPs = result.outputs.P_sa.slice(-50).reduce((a,b)=>a+b,0)/50;
      const error = Math.abs(simulatedPs - patientMeasurements.measured_sa_pressure);
      if (error < bestError) {
        bestError = error;
        best = r;
      }
    }
    onCalibrated(best);
    setLoading(false);
  };

  return (
    <div>
      <h3>Калибровка модели по данным пациента</h3>
      <p>Измеренное АД сист.: {patientMeasurements.measured_sa_pressure} мм рт. ст.</p>
      <button onClick={runCalibration} disabled={loading}>Подобрать R_vsd</button>
      {loading && <p>Калибровка...</p>}
    </div>
  );
};
```

---

## 6. Аутентификация и роли

### 6.1. Создаём контекст авторизации

```jsx
// src/context/AuthContext.js
import React, { createContext, useState, useContext } from 'react';
import { login as apiLogin } from '../services/authApi';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    const token = localStorage.getItem('token');
    if (token) {
      const decoded = jwtDecode(token);
      return { id: decoded.sub, role: decoded.role };
    }
    return null;
  });

  const login = async (username, password) => {
    const data = await apiLogin(username, password);
    localStorage.setItem('token', data.access_token);
    const decoded = jwtDecode(data.access_token);
    setUser({ id: decoded.sub, role: decoded.role });
  };

  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
```

### 6.2. Защита маршрутов

Используйте `react-router-dom` и компонент `PrivateRoute`, который проверяет роль (`doctor` / `patient`). Для пациентов доступен только просмотр результатов, для врачей — полное управление симуляцией.

---

## 7. Production-сборка и Docker

### 7.1. Оптимизация сборки React

```bash
npm run build
```

Создаст оптимизированные статические файлы в папке `build`.

### 7.2. Dockerfile для фронтенда (многоэтапная сборка)

```dockerfile
# Dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### 7.3. docker-compose.yml для всего проекта

```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://...
  frontend:
    build: ./frontend/hbs-dashboard
    ports:
      - "3000:80"
    depends_on:
      - backend
```

---

## 8. Пользовательская документация

Создайте страницу `/help` с инструкцией:

- Как выбрать пациента из ЭМК
- Как запустить базовую симуляцию
- Как интерпретировать графики (нормальные значения давлений, Qp/Qs)
- Как использовать сравнение сценариев для планирования лечения
- Как экспортировать отчёт

Используйте `react-markdown` для удобного редактирования.

---

## 9. Итоговая структура приложения

```
src/
├── components/
│   ├── PatientInfo.js
│   ├── SimulationPanel.js
│   ├── ResultsPreview.js
│   ├── TimeSeriesChart.js
│   ├── SteadyStateTable.js
│   ├── ProgressBar.js
│   ├── ScenarioComparison.js
│   ├── ModelCalibration.js
│   └── HelpPage.js
├── pages/
│   ├── Dashboard.js
│   ├── Compare.js
│   ├── Calibrate.js
│   └── Login.js
├── hooks/
│   ├── useSimulation.js
│   ├── useWebSocketSimulation.js
│   ├── useHistory.js
│   └── useAuth.js
├── services/
│   ├── api.js
│   ├── emrApi.js
│   └── authApi.js
├── utils/
│   └── exportUtils.js
├── context/
│   └── AuthContext.js
├── App.js
├── index.js
└── ...
```

---

## 10. Запуск финальной версии

1. Убедитесь, что бэкенд FastAPI с WebSocket поддержкой запущен.
2. Соберите и запустите фронтенд через Docker или `npm start`.
3. Зайдите как врач (логин: `doctor@hbs.com`, пароль: `test`) или как пациент.
4. Протестируйте все функции: симуляция, сравнение, калибровка, экспорт, история.

---

## 11. Что мы достигли

- ✅ Полноценное **клиническое приложение** для моделирования гемодинамики
- ✅ **Сравнение сценариев** помогает выбрать оптимальную тактику лечения
- ✅ **Калибровка модели** повышает точность прогноза для конкретного пациента
- ✅ **WebSocket** даёт отзывчивый интерфейс при длительных расчётах
- ✅ **Аутентификация и роли** обеспечивают безопасность
- ✅ **Docker-развёртывание** упрощает установку в больнице

---

## 12. Дальнейшее развитие

- **Добавить поддержку большего числа патологий** (клапаны, сердечная недостаточность)
- **Интегрировать с реальной ЭМК через FHIR** (стандарт обмена медицинскими данными)
- **Использовать GPU-ускорение** для сложных симуляций (через TensorFlow.js или внешний сервис)
- **Внедрить машинное обучение** для предсказания исходов на основе истории пациента

---

## Поздравляю!

Вы прошли полный цикл разработки веб-приложения **Human Body Simulation** на React.js — от минимального "Hello World" до production-ready системы, интегрированной с бэкендом на FastAPI и моделью физиологии человека.

Теперь вы можете создавать медицинские симуляторы, которые помогают врачам принимать обоснованные решения. 🧬🚀

**Удачи в ваших проектах!**