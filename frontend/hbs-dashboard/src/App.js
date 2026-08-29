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