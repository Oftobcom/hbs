# hbs
# HBS – Human Body Simulation

Human Body Simulation (HBS) is a modular Python framework for multi-organ physiological modeling.
It supports compartmental ODE-based simulations and extended disease-specific scenarios such as:

* VSD (Ventricular Septal Defect) circulation models
* Multi-organ pharmacokinetic and metabolic simulations

The project is designed for:

* Computational physiology research
* Mathematical biology
* Differential equation modeling
* Whole-body system simulations
* Educational and experimental biomedical modeling

---

# Core Design Philosophy

HBS follows a **modular organ-based architecture**:

* Each organ is represented as an independent computational module
* All organs inherit from a shared `organ_base.py`
* Interactions are handled via blood exchange and whole-body coupling
* Systems can be simulated via ODE integration

This allows:

* Plug-and-play organ extensions
* Disease-specific overrides
* Numerical scheme experimentation

---

# Implemented Physiological Modules

Current organ-level modules include:

* Baroreflex
* Blood compartment
* Brain
* Gastrointestinal tract
* Heart
* Jugular vein
* Kidney
* Liver
* Lungs

Each organ defines:

* State variables
* Exchange dynamics
* Internal metabolic or physiological processes
* Interface with systemic circulation

---

# How to Run

```bash
cd vsd_rus
python run_simulation_parallel.py
```

---

# Mathematical Foundation

The framework is based on:

* Systems of Ordinary Differential Equations (ODE)
* Compartmental modeling
* Mass balance principles
* Organ-to-organ exchange fluxes
* Coupled nonlinear dynamics

---

# Extending the Framework

To add a new organ:

1. Create a new file inheriting from `organ_base.py`
2. Define:

   * State variables
   * Update equations
   * Exchange interface
3. Register organ in `whole_body.py`

To add a new disease model:

* Create a new directory under `hbs/`
* Reuse organ modules
* Override disease-specific dynamics
* Add a new `run_simulation.py`

---

# Research Use Cases

HBS can be used for:

* Cardiac defect modeling
* Multi-organ pharmacokinetics
* Metabolic disorder simulation
* Hepatitis B viral dynamics research
* Educational demonstrations
* Numerical method comparison

---

# License

MIT License

```
MIT License

Copyright (c) 2026

Rahmatjon I. Hakimov
---

# Author

Rahmatjon I. Hakimov
