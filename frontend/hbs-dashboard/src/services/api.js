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