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