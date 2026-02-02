# Modelo DSGE Smets & Wouters - Argentina

Replicación del modelo DSGE New Keynesian de Smets & Wouters (2007) aplicado a datos macroeconómicos de Argentina.

## Referencia

> Smets, F. & Wouters, R. (2007). "Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach"
> *American Economic Review*, 97(3), 586-606.

## Datos

- **Período**: 2004Q2 - 2025Q3 (86 observaciones trimestrales)
- **Variables observables** (7):
  - `dy`: Crecimiento del PIB
  - `dc`: Crecimiento del consumo
  - `dinve`: Crecimiento de la inversión
  - `labobs`: Horas trabajadas
  - `pinfobs`: Inflación trimestral
  - `dw`: Crecimiento de salarios reales
  - `robs`: Tasa de interés

## Estructura del Repositorio

```
DG/
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias Python
│
├── data/
│   ├── Arg.xlsx                 # Datos de Argentina (original)
│   └── argmodel_data.mat        # Datos en formato Dynare (generado)
│
├── model/
│   ├── argmodel.mod             # Modelo Dynare para estimación
│   └── argmodel_figures.mod     # Modelo para generar figuras
│
├── src/
│   ├── __init__.py              # Módulo Python
│   ├── dynare_interface.py      # Interfaz Python-Dynare
│   ├── results_extractor.py     # Extracción de resultados
│   └── data_preparation.py      # Preparación de datos
│
├── notebooks/
│   ├── tables_argentina.ipynb   # Genera Tablas 1A y 1B
│   └── figures_argentina.ipynb  # Genera Figuras 1 y 2
│
└── output/
    ├── tables/                  # CSVs de tablas generadas
    └── figures/                 # PNGs de figuras generadas
```

## Requisitos

### Software
- Python 3.8+
- GNU Octave 6.x o superior
- Dynare 5.x o 6.x

### Dependencias Python
```bash
pip install -r requirements.txt
```

## Uso

### 1. Preparar datos
Ejecutar desde la carpeta `src/`:
```bash
python data_preparation.py
```
Esto genera `argmodel_data.mat` a partir de `Arg.xlsx`.

### 2. Generar Tablas 1A y 1B
Ejecutar el notebook `notebooks/tables_argentina.ipynb`:
- Estima el modelo DSGE con datos de Argentina
- Genera tablas de parámetros estructurales y shocks
- Exporta resultados a `output/tables/`

### 3. Generar Figuras 1 y 2
Ejecutar el notebook `notebooks/figures_argentina.ipynb`:
- Genera FEVD (Descomposición de Varianza del Error de Pronóstico)
- Genera IRFs (Funciones Impulso-Respuesta)
- Exporta figuras a `output/figures/`

## Outputs

### Tablas
- **Tabla 1A**: Distribución prior y posterior de 19 parámetros estructurales
- **Tabla 1B**: Distribución prior y posterior de 17 parámetros de shocks

### Figuras
- **Figura 1**: FEVD para crecimiento del PIB, inflación y tasa de interés
- **Figura 2**: IRFs a shocks de demanda (prima de riesgo, gasto gobierno, inversión)

## Configuración

Modificar las rutas en los notebooks según tu instalación:
```python
os.environ['OCTAVE_EXECUTABLE'] = r'C:\Program Files\GNU Octave\Octave-10.3.0\mingw64\bin\octave-cli.exe'
DYNARE_PATH = r'C:\dynare\6.5\matlab'
```

## Notas

- La estimación MCMC puede tardar varios minutos dependiendo de `mh_replic`
- El modelo usa los mismos priors que el paper original de Smets & Wouters (2007)
- Los datos de Argentina tienen mayor volatilidad que US, especialmente en inflación
