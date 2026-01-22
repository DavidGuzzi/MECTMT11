# Direct Replication - Smets & Wouters (2007)

Replicación directa del paper usando Dynare vía oct2py.

## Descripción

Esta carpeta contiene la infraestructura para replicar los resultados de Smets & Wouters (2007) "Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach" llamando directamente a las funciones MATLAB/Octave desde Python.

El objetivo es verificar que los resultados del paper pueden ser replicados antes de cambiar las fuentes de datos.

## Estructura

```
direct_replication/
├── __init__.py                           # Módulo Python
├── requirements.txt                      # Dependencias Python
├── setup_instructions.md                 # Instrucciones de instalación
├── README.md                             # Este archivo
│
├── dynare_interface.py                   # Wrapper para Dynare vía oct2py
├── sims_bvar_interface.py               # Wrapper para funciones BVAR de Sims
├── results_extractor.py                 # Extracción de resultados de Dynare
├── verification.py                       # Tests de verificación
│
└── notebooks/
    └── replication_notebook.ipynb       # Notebook principal de replicación
```

## Requisitos Previos

### Software Externo

1. **GNU Octave** (v6.x o 7.x)
   - Descargar: https://www.gnu.org/software/octave/download
   - Debe estar en PATH del sistema

2. **Dynare** (v5.x o 6.x)
   - Descargar: https://www.dynare.org/download/
   - Anotar la ruta al folder `matlab/` (ej: `C:\dynare\6.2\matlab`)

3. **Sims VARtools**
   - Descargar de: http://sims.princeton.edu/yftp/VARtools/
   - Archivos necesarios: `varprior.m`, `rfvar3.m`, `matrictint.m`
   - Colocar en carpeta `../repo/`

### Python Dependencies

```bash
pip install -r requirements.txt
```

Dependencias principales:
- `oct2py>=5.6.0` - Bridge Python-Octave
- `numpy`, `scipy`, `pandas` - Cómputo científico
- `matplotlib` - Gráficos
- `jupyter` - Notebooks

## Instalación

Ver instrucciones detalladas en [setup_instructions.md](setup_instructions.md).

**Resumen rápido:**

1. Instalar Octave y agregar a PATH
2. Instalar Dynare (anotar ruta)
3. Descargar Sims VARtools a `repo/`
4. Instalar dependencias Python: `pip install -r requirements.txt`
5. Verificar instalación ejecutando el notebook

## Uso

### Notebook Principal

El notebook [replication_notebook.ipynb](notebooks/replication_notebook.ipynb) ejecuta la replicación completa:

1. **Setup y Configuración** - Test de conexión Octave/Dynare
2. **Cargar Datos** - Datos US 1955-2005
3. **Ejecutar Dynare** - Estimar modelo DSGE
4. **Extraer Resultados** - Parámetros, IRFs, likelihood
5. **Análisis IRFs** - Generar gráficos
6. **BVAR Marginal Likelihoods** - BVAR(1) a BVAR(4)
7. **Forecasting Recursivo** - Horizontes 1, 2, 4, 8, 12
8. **Verificación** - Comparar con paper

### Uso Programático

```python
from direct_replication import DynareInterface, SimsBVAR

# Inicializar Dynare
di = DynareInterface(
    dynare_path=r'C:\dynare\6.2\matlab',
    model_path=r'C:\path\to\repo'
)

# Ejecutar modelo
di.run_model('usmodel.mod')

# Extraer resultados
params = di.get_parameters()
irfs = di.get_irfs(periods=20)
ss = di.get_state_space()

# BVAR
bvar = SimsBVAR(di.oc, repo_path)
mlik = bvar.mgnldnsty(data, lags=4)
```

## Archivos de Salida

El notebook genera:

- `parameter_estimates.csv` - Parámetros estimados (posterior mode)
- `irfs_figure.png` - Gráfico de impulse responses
- `bvar_marginal_likelihoods.csv` - Marginal likelihoods BVAR(1-4)
- `forecast_rmse.csv` - RMSE de forecasts por horizonte

## Verificación

La clase `ReplicationVerification` implementa 6 criterios de verificación:

| Criterio | Tolerancia |
|----------|------------|
| 1. State-space matrices (T, R, Z) | Frobenius norm < 1e-4 |
| 2. Log-likelihood at mode | < 1e-4 |
| 3. Parámetros estimados | < 1% |
| 4. IRFs | RMSE < 5% del pico |
| 5. BVAR marginal likelihoods | < 1% |
| 6. Forecast RMSE | < 1% |

## Módulos

### `dynare_interface.py`

Clase `DynareInterface`:
- `run_model(mod_file)` - Ejecutar Dynare
- `get_state_space()` - Extraer matrices T, R, Z
- `get_irfs(periods)` - Extraer IRFs
- `get_parameters()` - Extraer parámetros
- `get_likelihood()` - Extraer log-likelihood

### `sims_bvar_interface.py`

Clase `SimsBVAR`:
- `mgnldnsty(ydata, lags)` - Marginal log density
- `mgnldnsty_fcast(ydata, lags, start_for, horizon)` - Forecasts
- `estimate_bvar(ydata, lags)` - Estimar BVAR completo
- `compute_forecast_rmse(errors)` - Calcular RMSE

### `results_extractor.py`

Funciones utilitarias:
- `extract_state_space_matrices(oc)` - Extraer T, R, Z, steady state
- `extract_parameter_estimates(oc)` - Extraer parámetros con std errors
- `convert_dr_ordering_to_declaration(oc, matrix)` - Convertir ordenamiento Dynare
- `extract_forecast_results(oc, horizon)` - Extraer forecasts
- `get_model_info(oc)` - Metadata del modelo

### `verification.py`

Clase `ReplicationVerification`:
- `verify_likelihood(computed, reference)` - Criterio 2
- `verify_parameters(computed, reference)` - Criterio 3
- `verify_state_space(T_computed, T_reference)` - Criterio 1
- `verify_irfs(computed, reference)` - Criterio 4
- `verify_bvar_marginal_lik(computed, reference)` - Criterio 5
- `verify_forecast_rmse(computed, reference)` - Criterio 6
- `run_all_tests(results)` - Ejecutar todos los tests

## Solución de Problemas

### Error: "octave-cli not found"
- Verificar que Octave está en PATH
- Reiniciar terminal después de modificar PATH
- O configurar: `os.environ['OCTAVE_EXECUTABLE'] = r'C:\...\octave-cli.exe'`

### Error: "dynare command not found"
- Verificar DYNARE_PATH en el notebook
- Asegurar que `addpath()` se ejecuta antes de llamar a Dynare

### Error: "undefined function varprior"
- Descargar VARtools de http://sims.princeton.edu/yftp/VARtools/
- Colocar archivos en `repo/`

### Dynare tarda mucho
- Normal, la estimación puede tardar 5-10 minutos
- Se está ejecutando optimización Bayesiana completa

## Referencias

- **Paper Original**: Smets, F., & Wouters, R. (2007). Shocks and frictions in US business cycles: A Bayesian DSGE approach. *American Economic Review*, 97(3), 586-606.

- **Dynare**: https://www.dynare.org/
- **Oct2Py**: https://oct2py.readthedocs.io/
- **Sims VARtools**: http://sims.princeton.edu/yftp/VARtools/

## Licencia

Este código es para uso académico como parte del proyecto MECTMT11.

## Autor

David Guzzi - Propuesta Examen de Macroeconometría
