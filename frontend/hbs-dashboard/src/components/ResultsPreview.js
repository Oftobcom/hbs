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