# React.js Tutorial 02 — Структурированный дашборд HBS с компонентами и состоянием

Теперь мы делаем **важный шаг** к реальному проекту **Human Body Simulation (HBS)**.  
В этом уроке мы создадим **структурированное React-приложение**, которое будет содержать отдельные компоненты для отображения данных пациента, настройки параметров симуляции и отображения результатов (пока заглушки).

---

## 1. Цель урока

Научиться:
- Организовывать код React в компонентную структуру (как в реальном проекте `hbs/frontend/src/`)
- Использовать **хуки** (`useState`, `useEffect`) для управления состоянием
- Создавать **форму управления симуляцией** (выбор типа пациента, параметры ДМЖП)
- Подготавливать **HTTP клиент** для будущей интеграции с FastAPI бэкендом
- Декомпозировать интерфейс на переиспользуемые компоненты

---

## 2. Обновление структуры проекта

```bash
cd hbs/frontend/hbs-dashboard

# Создаём структуру папок
mkdir -p src/components
mkdir -p src/pages
mkdir -p src/services
mkdir -p src/hooks
mkdir -p src/utils

# Удаляем ненужные файлы (опционально)
rm src/App.test.js src/setupTests.js src/logo.svg
```

**Финальная структура (после урока):**
```
hbs-dashboard/
├── src/
│   ├── components/
│   │   ├── PatientInfo.js
│   │   ├── SimulationPanel.js
│   │   └── ResultsPreview.js
│   ├── pages/
│   │   └── Dashboard.js
│   ├── services/
│   │   └── api.js         (клиент для связи с бэкендом)
│   ├── hooks/
│   │   └── useSimulation.js
│   ├── App.js
│   ├── App.css
│   └── index.js
├── public/
├── package.json
└── README.md
```

---

## 3. Установка дополнительных зависимостей

```bash
npm install axios react-router-dom
# recharts для графиков добавим в следующем уроке
```

---

## 4. Компонент информации о пациенте (`PatientInfo.js`)

Пока используем **mock-данные**, в будущем они будут приходить из ЭМК или API.

**`src/components/PatientInfo.js`**
```jsx
import React from 'react';

const PatientInfo = ({ patient }) => {
  if (!patient) {
    return <div>Выберите пациента</div>;
  }

  return (
    <div className="patient-info">
      <h3>📋 Информация о пациенте</h3>
      <p><strong>ФИО:</strong> {patient.name}</p>
      <p><strong>Возраст:</strong> {patient.age} лет</p>
      <p><strong>Диагноз:</strong> {patient.diagnosis}</p>
      <p><strong>Дата поступления:</strong> {patient.admissionDate}</p>
      
      <h4>🧪 Лабораторные показатели</h4>
      <ul>
        <li>Билирубин: {patient.labs.bilirubin} мкмоль/л</li>
        <li>Аммиак: {patient.labs.ammonia} мкмоль/л</li>
        <li>Альбумин: {patient.labs.albumin} г/л</li>
        <li>Токсины: {patient.labs.toxins} у.е.</li>
      </ul>
    </div>
  );
};

export default PatientInfo;
```

---

## 5. Компонент панели управления симуляцией (`SimulationPanel.js`)

Позволяет выбрать тип пациента или вручную задать параметры ДМЖП.

**`src/components/SimulationPanel.js`**
```jsx
import React, { useState } from 'react';

const SimulationPanel = ({ onSimulate }) => {
  const [preset, setPreset] = useState('healthy');
  const [vsdResistance, setVsdResistance] = useState(Infinity);
  const [flowDependentLungs, setFlowDependentLungs] = useState(false);
  const [duration, setDuration] = useState(200);

  const handlePresetChange = (e) => {
    const value = e.target.value;
    setPreset(value);
    if (value === 'healthy') setVsdResistance(Infinity);
    else if (value === 'small_vsd') setVsdResistance(5.0);
    else if (value === 'large_vsd') setVsdResistance(1.0);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSimulate({
      vsd_resistance: vsdResistance,
      flow_dependent_lungs: flowDependentLungs,
      t_span: [0, duration],
      t_eval_points: 2000
    });
  };

  return (
    <div className="simulation-panel">
      <h3>⚙️ Параметры симуляции</h3>
      <form onSubmit={handleSubmit}>
        <label>
          Тип пациента / дефекта:
          <select value={preset} onChange={handlePresetChange}>
            <option value="healthy">Здоровый</option>
            <option value="small_vsd">Малый ДМЖП (R=5.0)</option>
            <option value="large_vsd">Большой ДМЖП (R=1.0)</option>
            <option value="custom">Произвольный</option>
          </select>
        </label>

        {preset === 'custom' && (
          <label>
            Сопротивление ДМЖП (R_vsd):
            <input
              type="number"
              step="0.5"
              value={vsdResistance === Infinity ? '' : vsdResistance}
              onChange={(e) => setVsdResistance(parseFloat(e.target.value) || Infinity)}
            />
            (бесконечность = нет шунта)
          </label>
        )}

        <label>
          <input
            type="checkbox"
            checked={flowDependentLungs}
            onChange={(e) => setFlowDependentLungs(e.target.checked)}
          />
          Учитывать адаптацию лёгких (повышение сопротивления при перегрузке)
        </label>

        <label>
          Длительность симуляции (секунд):
          <input
            type="number"
            value={duration}
            onChange={(e) => setDuration(parseInt(e.target.value))}
          />
        </label>

        <button type="submit">▶ Запустить симуляцию</button>
      </form>
    </div>
  );
};

export default SimulationPanel;
```

---

## 6. Компонент предпросмотра результатов (`ResultsPreview.js`)

Пока только заглушка, в следующем уроке добавим графики и таблицы.

**`src/components/ResultsPreview.js`**
```jsx
import React from 'react';

const ResultsPreview = ({ results, loading }) => {
  if (loading) return <div>⏳ Выполняется симуляция...</div>;
  if (!results) return <div>Здесь появятся результаты расчёта</div>;

  // Выводим несколько ключевых показателей (заглушка)
  return (
    <div className="results-preview">
      <h3>📊 Результаты симуляции</h3>
      <p><strong>Соотношение Qp/Qs:</strong> {results.qp_qs?.toFixed(2) ?? '—'}</p>
      <p><strong>Системное давление (среднее):</strong> {results.mean_sa_pressure?.toFixed(0)} мм рт. ст.</p>
      <p><strong>Лёгочное давление (среднее):</strong> {results.mean_pa_pressure?.toFixed(0)} мм рт. ст.</p>
      <p><strong>Объём крови:</strong> {results.blood_volume?.toFixed(0)} мл</p>
      <p><em>Графики и полная таблица появятся в следующем уроке.</em></p>
    </div>
  );
};

export default ResultsPreview;
```

---

## 7. Сервис для взаимодействия с бэкендом (`api.js`)

Пока создаём заглушку, которая имитирует задержку и возвращает тестовые данные.

**`src/services/api.js`**
```js
// В будущем здесь будет реальный вызов к FastAPI бэкенду
// POST /simulate

export const runSimulation = async (params) => {
  console.log('Отправка параметров на бэкенд:', params);
  
  // Имитация асинхронного запроса
  return new Promise((resolve) => {
    setTimeout(() => {
      // Mock-результаты (в реальности придёт от solve_ivp)
      resolve({
        qp_qs: params.vsd_resistance === Infinity ? 1.0 : 2.3,
        mean_sa_pressure: 98,
        mean_pa_pressure: params.vsd_resistance === Infinity ? 18 : 42,
        blood_volume: 5100,
        message: 'Симуляция завершена (тестовые данные)'
      });
    }, 1500);
  });
};
```

---

## 8. Пользовательский хук `useSimulation.js`

Управляет состоянием симуляции (загрузка, результаты, ошибки).

**`src/hooks/useSimulation.js`**
```js
import { useState } from 'react';
import { runSimulation } from '../services/api';

const useSimulation = () => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const simulate = async (params) => {
    setLoading(true);
    setError(null);
    try {
      const data = await runSimulation(params);
      setResults(data);
    } catch (err) {
      setError(err.message || 'Ошибка при выполнении симуляции');
    } finally {
      setLoading(false);
    }
  };

  return { loading, results, error, simulate };
};

export default useSimulation;
```

---

## 9. Главная страница дашборда (`Dashboard.js`)

Объединяет все компоненты, использует хук для запуска симуляции.

**`src/pages/Dashboard.js`**
```jsx
import React, { useState } from 'react';
import PatientInfo from '../components/PatientInfo';
import SimulationPanel from '../components/SimulationPanel';
import ResultsPreview from '../components/ResultsPreview';
import useSimulation from '../hooks/useSimulation';

// Временные mock-данные пациента
const mockPatient = {
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
};

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
        <ResultsPreview results={results} loading={loading} />
        {error && <div className="error">❌ Ошибка: {error}</div>}
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

## 10. Обновление `App.js` и стилей

**`src/App.js`**
```jsx
import React from 'react';
import Dashboard from './pages/Dashboard';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>🫀 Human Body Simulation (HBS)</h1>
        <p>Модель целостной физиологии человека + ДМЖП</p>
      </header>
      <main>
        <Dashboard />
      </main>
      <footer>
        <p>© 2025 HBS — интеграция с ЭМК и реальной клинической практикой</p>
      </footer>
    </div>
  );
}

export default App;
```

**`src/App.css`** (добавим стили для двухколоночной сетки)

```css
.App {
  text-align: center;
  min-height: 100vh;
  background-color: #f5f7fa;
}

.App-header {
  background-color: #1e3a5f;
  padding: 1rem;
  color: white;
}

main {
  padding: 2rem;
}

.dashboard {
  display: flex;
  flex-wrap: wrap;
  gap: 2rem;
  justify-content: center;
}

.dashboard-left {
  flex: 1;
  min-width: 300px;
  background-color: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  padding: 1.5rem;
}

.dashboard-right {
  flex: 2;
  min-width: 400px;
  background-color: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  padding: 1.5rem;
}

.patient-info, .simulation-panel, .results-preview {
  text-align: left;
}

.simulation-panel form {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 1rem;
}

.simulation-panel label {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.simulation-panel input, .simulation-panel select {
  margin-left: 1rem;
  padding: 0.25rem;
}

button {
  background-color: #3b82f6;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 0.5rem 1rem;
  cursor: pointer;
}

button:hover {
  background-color: #2563eb;
}

.error {
  color: red;
  margin-top: 1rem;
}

footer {
  background-color: #e2e8f0;
  padding: 1rem;
  margin-top: 2rem;
  font-size: 0.875rem;
}
```

---

## 11. Запуск и тестирование

```bash
npm start
```

**Проверьте в браузере:**
- Отображается информация о пациенте (mock)
- Можно выбрать тип пациента, изменить параметры
- При нажатии «Запустить симуляцию» появляется индикатор загрузки (1.5 сек), затем отображаются тестовые результаты
- Переключение между здоровым / ДМЖП меняет поле сопротивления (в custom)

---

## 12. Что мы приблизили к реальному проекту HBS

- ✅ Модульная архитектура (компоненты, хуки, сервисы)
- ✅ Управление состоянием через `useState` и пользовательский хук
- ✅ Подготовка к HTTP-запросам (заглушка `api.js`)
- ✅ Форма с различными типами входных данных (пресеты, чекбокс, число)
- ✅ Разделение на левую панель (пациент + настройки) и правую (результаты)
