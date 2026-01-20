# MECTMT11 - Smets & Wouters (2007) Python Replication

Replicación en Python del modelo DSGE de Smets & Wouters (2007) "Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach" para su aplicación a datos macroeconómicos argentinos.

## Descripción del Proyecto

Este proyecto es parte del examen final de la Maestría en Econometría (Macroeconometría). El objetivo es:

1. **Replicar** el modelo DSGE New Keynesian de Smets & Wouters (2007) traduciendo el código MATLAB/Dynare original a Python modular
2. **Validar** la implementación comparando resultados con el paper original usando datos de EE.UU.
3. **Aplicar** el modelo a datos macroeconómicos de Argentina
4. **Evaluar** el desempeño del modelo en una economía emergente

## Estructura del Repositorio

```
MECTMT11/
├── README.md                   # Este archivo
├── memory.md                   # Documentación del proyecto
├── requirements.txt            # Dependencias de Python
├── idea/                       # Propuesta del trabajo
├── papers/                     # Papers de referencia
├── repo/                       # Código original MATLAB/Dynare
│   ├── usmodel.mod             # Especificación del modelo
│   └── usmodel_data.xls        # Datos de EE.UU.
└── replication/                # Implementación en Python
    ├── priors.py               # Distribuciones a priori
    ├── solver.py               # Solver QZ
    ├── kalman.py               # Filtro de Kalman
    ├── model.py                # Especificación del modelo
    ├── estimation.py           # Estimación bayesiana
    ├── bvar.py                 # VAR bayesiano
    └── ... (otros módulos)
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso Básico

Ver `replication.ipynb` para ejemplos completos y `memory.md` para documentación técnica detallada.

## Referencias

- Smets, F. y Wouters, R. (2007). American Economic Review, 97(3), 586-606
- DeJong, D. y Dave, C. (2011). Structural Macroeconometrics, Capítulo 3
- Chari, V. V., Kehoe, P. y McGrattan, E. (2009). AEJ: Macroeconomics, 1(1), 242-266

