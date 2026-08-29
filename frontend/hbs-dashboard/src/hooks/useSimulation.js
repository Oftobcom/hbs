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