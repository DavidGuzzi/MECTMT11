# Transformaciones de Datos - Smets & Wouters (2007)

Documentación de las transformaciones aplicadas a los datos originales de FRED para obtener las variables observables del modelo DSGE.

**Fuente:** `repo/readme.pdf`

---

## Variables Observables del Modelo (7 variables)

Las siguientes variables son las que observa el modelo DSGE en la estimación Bayesiana:

| Variable | Nombre | Descripción |
|----------|--------|-------------|
| `dy` | Crecimiento del PIB | Δ output (trimestral) |
| `dc` | Crecimiento del consumo | Δ consumption (trimestral) |
| `dinve` | Crecimiento de la inversión | Δ investment (trimestral) |
| `labobs` | Horas trabajadas | hours (log per cápita) |
| `pinfobs` | Inflación | inflation (trimestral) |
| `dw` | Crecimiento salarial | Δ real wage (trimestral) |
| `robs` | Tasa de interés | interest rate (trimestral) |

---

## Transformaciones desde Series Originales

### 1. Consumo Real Per Cápita (log × 100)
```
consumption = LN((PCEC / GDPDEF) / LNSindex) × 100
```

Donde:
- **PCEC**: Personal Consumption Expenditures (nominal, billones de dólares)
- **GDPDEF**: Deflactor implícito del PIB (1996=100)
- **LNSindex**: Población civil no institucional normalizada (1992:3 = 1)

**dc** = Δ consumption (primera diferencia)

---

### 2. Inversión Real Per Cápita (log × 100)
```
investment = LN((FPI / GDPDEF) / LNSindex) × 100
```

Donde:
- **FPI**: Fixed Private Investment (nominal, billones de dólares)
- **GDPDEF**: Deflactor implícito del PIB (1996=100)
- **LNSindex**: Población civil no institucional normalizada

**dinve** = Δ investment (primera diferencia)

---

### 3. PIB Real Per Cápita (log × 100)
```
output = LN(GDPC96 / LNSindex) × 100
```

Donde:
- **GDPC96**: Real GDP (chain-weighted 1996 dollars, billones)
- **LNSindex**: Población civil no institucional normalizada

**dy** = Δ output (primera diferencia)

---

### 4. Horas Trabajadas Per Cápita (log × 100)
```
hours = LN((PRS85006023 × CE16OV / 100) / LNSindex) × 100
```

Donde:
- **PRS85006023**: Nonfarm Business Average Weekly Hours (índice 1992=100)
- **CE16OV**: Civilian Employment 16+ (miles de personas)
- **LNSindex**: Población civil no institucional normalizada

**labobs** = hours (sin primera diferencia, ya en niveles)

---

### 5. Inflación (log × 100)
```
inflation = LN(GDPDEF / GDPDEF₍₋₁₎) × 100
```

Donde:
- **GDPDEF**: Deflactor implícito del PIB (1996=100)

**pinfobs** = inflation (tasa de cambio trimestral del deflactor)

---

### 6. Salario Real (log × 100)
```
real wage = LN(PRS85006103 / GDPDEF) × 100
```

Donde:
- **PRS85006103**: Nonfarm Business Hourly Compensation (índice 1992=100)
- **GDPDEF**: Deflactor implícito del PIB (1996=100)

**dw** = Δ real wage (primera diferencia)

---

### 7. Tasa de Interés Nominal (%)
```
interest rate = Federal Funds Rate / 4
```

Donde:
- **Federal Funds Rate**: Tasa de fondos federales (% anual)
- División por 4: Convierte tasa anual → tasa trimestral

**robs** = interest rate (ya en trimestral)

---

## Series FRED Utilizadas

### Cuentas Nacionales
- **GDPC96**: Real Gross Domestic Product (Chained 1996 Dollars)
  - Fuente: U.S. Bureau of Economic Analysis
- **GDPDEF**: GDP Implicit Price Deflator (1996=100)
  - Fuente: U.S. Bureau of Economic Analysis
- **PCEC**: Personal Consumption Expenditures (Nominal)
  - Fuente: U.S. Bureau of Economic Analysis
- **FPI**: Fixed Private Investment (Nominal)
  - Fuente: U.S. Bureau of Economic Analysis

### Mercado Laboral
- **CE16OV**: Civilian Employment 16+ (Miles, Seasonally Adjusted)
  - Fuente: U.S. Bureau of Labor Statistics
- **PRS85006023**: Nonfarm Business Average Weekly Hours (Index 1992=100)
  - Fuente: U.S. Department of Labor
- **PRS85006103**: Nonfarm Business Hourly Compensation (Index 1992=100)
  - Fuente: U.S. Department of Labor
- **LNS10000000**: Civilian Noninstitutional Population 16+ (Miles)
  - Fuente: U.S. Bureau of Labor Statistics
  - Antes de 1976: LFU800000000

### Política Monetaria
- **FEDFUNDS**: Federal Funds Rate (% Anual)
  - Fuente: Board of Governors of the Federal Reserve System
  - Antes de 1954: 3-Month Treasury Bill Rate

---

## Índices de Normalización

### LNSindex
```
LNSindex = LNS10000000 / LNS10000000(1992:3)
```
Normaliza la población civil no institucional para que 1992:3 = 1

### CE16OV index
```
CE16OV index = CE16OV / CE16OV(1992:3)
```
Normaliza el empleo civil para que 1992:3 = 1

---

## Período de Datos

- **Datos originales**: 1947:1 - 2004:4 (238 observaciones trimestrales)
- **Datos transformados**: 1947:3 - 2004:4 (231 observaciones)
  - Se pierden 2 observaciones por el cálculo de primeras diferencias y lag de inflación
- **Período de estimación principal**: 1966:1 - 2004:4 (156 observaciones)
  - Especificado en `usmodel.mod`: `first_obs=71, presample=4`

---

## Ubicación en usmodel_data.xls

El archivo Excel tiene dos secciones:

1. **Columnas 1-19**: Datos originales de FRED
   - GDPC96, GDPDEF, PCEC, FPI, CE16OV, etc.
   - Comienzan en fila 5 (1947:1)

2. **Columnas 22-28**: Variables observables transformadas
   - counter, dc, dinve, dy, labobs, pinfobs, dw, robs
   - Comienzan en fila 7 (1947:3, counter=1)

**Para leer en Python:**
```python
data = pd.read_excel(
    'usmodel_data.xls',
    skiprows=6,        # Saltar headers
    usecols='V:AB',    # Columnas 21-28
    names=['counter', 'dc', 'dinve', 'dy', 'labobs', 'pinfobs', 'dw', 'robs']
)
```

---

## Referencias

- **Paper**: Smets, F., & Wouters, R. (2007). "Shocks and frictions in US business cycles: A Bayesian DSGE approach." *American Economic Review*, 97(3), 586-606.
- **Código original**: `repo/readme.pdf`
- **Archivo de datos**: `repo/usmodel_data.xls`
