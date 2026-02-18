# hbs
# HBS – Human Body Simulation

Human Body Simulation (HBS) is a modular Python framework for multi-organ physiological modeling.
It supports compartmental ODE-based simulations and extended disease-specific scenarios such as:

* Hepatitis B systemic modeling
* VSD (Ventricular Septal Defect) circulation models
* Multi-organ pharmacokinetic and metabolic simulations

The project is designed for:

* Computational physiology research
* Mathematical biology
* Differential equation modeling
* Whole-body system simulations
* Educational and experimental biomedical modeling

---

# Project Structure

```
hbs/
│
├── hepatitis_b/
│   ├── prompts/
│   └── Python/
│       ├── organ_base.py
│       ├── blood.py
│       ├── liver.py
│       ├── lungs.py
│       ├── heart.py
│       ├── kidney.py
│       ├── brain.py
│       ├── gitract.py
│       ├── whole_body.py
│       └── run_simulation.py
│
├── vsd/
│   ├── prompts/
│   └── Python/
│       ├── HBS_FDG_v01/
│       └── HBS_ODE_v01/
│
└── README.md
```

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
* Model comparison (ODE vs FDG versions)

---

# Implemented Physiological Modules

Current organ-level modules include:

* Blood compartment
* Liver
* Lungs
* Heart
* Kidney
* Brain
* Gastrointestinal tract

Each organ defines:

* State variables
* Exchange dynamics
* Internal metabolic or physiological processes
* Interface with systemic circulation

---

# Model Types

## 1️⃣ Hepatitis B Model

Located in:

```
hepatitis_b/Python/
```

Focus:

* Liver-centered viral dynamics
* Systemic interaction
* Multi-organ coupling
* Whole-body disease progression simulation

Entry point:

```bash
python run_simulation.py
```

---

## 2️⃣ VSD Models

Located in:

```
vsd/Python/
```

Two implementations:

### • HBS_ODE_v01

Classic ODE-based systemic simulation.

### • HBS_FDG_v01

Extended version (e.g., fractional/differential variants or enhanced circulation dynamics).

---

# Requirements

Python 3.9+

Recommended packages:

```bash
pip install numpy scipy matplotlib
```

(Adjust depending on solver usage in your implementation.)

---

# How to Run

Example (Hepatitis B simulation):

```bash
cd hepatitis_b/Python
python run_simulation.py
```

Example (VSD ODE version):

```bash
cd vsd/Python/HBS_ODE_v01
python run_simulation.py
```

---

# Mathematical Foundation

The framework is based on:

* Systems of Ordinary Differential Equations (ODE)
* Compartmental modeling
* Mass balance principles
* Organ-to-organ exchange fluxes
* Coupled nonlinear dynamics

Future extensions may include:

* Fractional differential equations
* Delay differential equations
* Control-theoretic intervention modeling
* Parameter estimation modules
* Sensitivity analysis
* Optimization and inverse problems

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

* Hepatitis B viral dynamics research
* Cardiac defect modeling
* Multi-organ pharmacokinetics
* Metabolic disorder simulation
* Educational demonstrations
* Numerical method comparison

---

# Roadmap (Suggested)

* Parameter calibration module
* YAML/JSON configuration system
* CLI interface
* Visualization dashboard
* Unit tests
* Jupyter notebook examples
* Docker support
* Documentation site

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