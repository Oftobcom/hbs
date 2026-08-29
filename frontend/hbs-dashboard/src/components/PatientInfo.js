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