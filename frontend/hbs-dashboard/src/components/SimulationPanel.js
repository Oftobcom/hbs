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