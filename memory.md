# MECTMT11 - Project Memory

## Project Overview

**Objective**: Replicate Smets & Wouters (2007) "Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach" by translating MATLAB/Dynare code to modular Python, then apply to Argentine macroeconomic data.

**Academic Context**: Master's thesis in Econometrics (Macroeconometría). The goal is to evaluate DSGE New Keynesian model performance in an emerging economy (Argentina) and analyze results in light of methodological critiques by Chari, Kehoe & McGrattan (2009).

## Repository Structure

```
MECTMT11/
├── memory.md                       # This file - project context
├── idea/                           # Project proposal
│   └── David Guzzi - Propuesta Examen de Macroeconometría.pdf
├── papers/                         # Reference papers
│   ├── Shocks and Frictions in US Business Cycles. A Bayesian DSGE Approach AER.pdf
│   ├── New Keynesian Models. Not Yet Useful for Policy Analysis AEJ.pdf
│   ├── Technology Shocks in the New Keynesian Model. RES.pdf
│   ├── Time to Build and Aggregate Fluctuations E.pdf
│   └── Structural Macroeconometrics.pdf
├── repo/                          # Original MATLAB/Dynare code
│   ├── usmodel.mod                # Dynare model specification (MAIN)
│   ├── usmodel_stst.m             # Steady-state calculations
│   ├── BVAR_statistics_usmodel_data.m  # BVAR analysis
│   ├── mgnldnsty_fcast.m          # Marginal likelihood & forecasting
│   ├── sims_fcast.m               # Simulation functions
│   ├── usmodel_data.xls           # US quarterly data 1955-2005
│   ├── usmodel_data.mat           # MATLAB data format
│   ├── usmodel_mode.mat           # Estimated posterior mode
│   └── usmodel_hist_dsge_f19_7_*_mode.mat  # Subsample estimates
└── replication/                   # Python replication (TO BE CREATED)
    ├── __init__.py
    ├── model.py                   # DSGE model class
    ├── solver.py                  # QZ decomposition rational expectations solver
    ├── kalman.py                  # Kalman filter
    ├── estimation.py              # Bayesian mode-finding
    ├── bvar.py                    # Bayesian VAR with Minnesota prior
    ├── forecast.py                # Forecast evaluation
    ├── data_loader.py             # Data I/O utilities
    ├── priors.py                  # Prior distribution classes
    ├── utils.py                   # General utilities
    └── replication.ipynb          # Main replication notebook
```

## Original Model Specification (Smets & Wouters 2007)

### Model Type
New Keynesian DSGE model with:
- Nominal rigidities (sticky prices and wages - Calvo pricing)
- Real frictions (habit formation, investment adjustment costs, variable capacity utilization)
- 7 structural shocks (productivity, risk premium, government spending, investment-specific, monetary policy, price markup, wage markup)

### Key Variables (7 Observed)
1. **dy**: Output growth
2. **dc**: Consumption growth
3. **dinve**: Investment growth
4. **labobs**: Hours worked (levels)
5. **pinfobs**: Inflation
6. **dw**: Wage growth
7. **robs**: Interest rate (Federal Funds Rate/4)

### Data Sources (US Data)
- GDPC96: Real GDP (1996 dollars)
- GDPDEF: GDP deflator (1996=100)
- PCEC: Personal consumption expenditures
- FPI: Fixed private investment
- CE16OV: Civilian employment 16+
- Federal Funds Rate
- PRS85006023: Average weekly hours (nonfarm business)
- PRS85006103: Hourly compensation (nonfarm business)
- Population: LNS10000000 (1992:3=1 normalization)

### Data Transformation
```
consumption = LN((PCEC / GDPDEF) / LNSindex) * 100
investment = LN((FPI / GDPDEF) / LNSindex) * 100
output = LN(GDPC96 / LNSindex) * 100
hours = LN((PRS85006023 * CE16OV / 100) / LNSindex) * 100
inflation = LN(GDPDEF / GDPDEF(-1)) * 100
real wage = LN(PRS85006103 / GDPDEF) * 100
interest rate = Federal Funds Rate / 4
```

All variables are **per capita** and in **logs × 100** (except interest rate).

### Estimation Period
- **Main estimation**: 1966Q1 - 2004Q4 (first_obs=71, with 40-quarter training sample from 1955)
- **Training sample**: 1955Q2 - 1965Q1 (40 observations for BVAR priors)
- **Total data**: 1955Q1 - 2005Q4 (~200 observations)

### Model Parameters (26 Estimated)

**Structural parameters:**
- calfa: Capital share (α)
- csigma: Risk aversion (σ)
- chabb: Habit formation (λ)
- csigl: Labor supply elasticity (σ_l)
- csadjcost: Investment adjustment cost (φ)
- czcap: Capacity utilization cost (ψ)
- cprobp: Price stickiness (Calvo probability)
- cprobw: Wage stickiness (Calvo probability)
- cindp: Price indexation (ι_p)
- cindw: Wage indexation (ι_w)
- cfc: Fixed cost (Φ, determines markup)

**Policy rule (Taylor rule):**
- crpi: Response to inflation (r_π)
- crr: Interest rate smoothing (ρ)
- cry: Response to output gap (r_y)
- crdy: Response to output gap change (r_Δy)

**Shock persistence (AR(1) coefficients):**
- crhoa: Productivity shock
- crhob: Risk premium shock
- crhog: Government spending shock
- crhoqs: Investment-specific shock
- crhoms: Monetary policy shock
- crhopinf: Price markup shock
- crhow: Wage markup shock

**MA terms:**
- cmap: Price markup MA coefficient
- cmaw: Wage markup MA coefficient

**Trend/constants:**
- ctrend: Quarterly trend growth rate
- constepinf: Steady-state inflation
- constebeta: Discount factor adjustment
- constelab: Labor hours constant
- cgy: Government spending response to productivity

### Prior Distributions

Priors specified in [repo/usmodel.mod](repo/usmodel.mod:164-203):
- **Beta** priors: Probabilities, persistence parameters (0-1 bounded)
- **Gamma** priors: Positive parameters with shape constraints
- **Normal** priors: Unbounded parameters (elasticities, policy coefficients)
- **Inverse-Gamma** priors: Shock standard deviations

Example:
```
csigma ~ Normal(1.50, 0.375)  [bounded: 0.25-3]
chabb ~ Beta(0.7, 0.1)        [bounded: 0.001-0.99]
crhoa ~ Beta(0.5, 0.20)       [bounded: 0.01-0.9999]
stderr(ea) ~ InvGamma(0.1, 2) [bounded: 0.01-3]
```

## Model Equations (Log-linearized)

### Flexible Economy (Benchmark without frictions)
- Production function (Cobb-Douglas with technology)
- Capital utilization decision
- Capital rental rate
- Investment Euler equation (Tobin's Q)
- Consumption Euler equation (with habit)
- Resource constraint
- Labor supply
- Capital accumulation

### Sticky Price-Wage Economy (Actual economy)
- All flexible economy equations +
- **New Keynesian Phillips Curve** (price inflation dynamics)
- **Wage Phillips Curve** (wage inflation dynamics)
- **Monetary policy rule** (Taylor rule with smoothing)
- Markup shocks (price and wage)

### Measurement Equations
Link model variables to observables:
```
dy = y - y(-1) + ctrend
dc = c - c(-1) + ctrend
dinve = inve - inve(-1) + ctrend
dw = w - w(-1) + ctrend
pinfobs = pinf + constepinf
robs = r + conster
labobs = lab + constelab
```

## Technical Implementation Details

### DSGE Solution Method
**Approach**: Custom implementation of Sims' QZ decomposition
- Transform model to canonical form: Γ0*s_t = Γ1*s_{t-1} + Ψ*ε_t + Π*η_t
- Apply generalized Schur (QZ) decomposition
- Select stable eigenvalues (saddle-path)
- Construct state-space representation: s_t = T*s_{t-1} + R*ε_t

### Kalman Filter
Standard Kalman filter for likelihood evaluation:
```python
# Prediction
s_t|t-1 = T * s_{t-1|t-1}
P_t|t-1 = T * P_{t-1|t-1} * T' + R * Q * R'

# Update
y_t|t-1 = Z * s_t|t-1 + D
v_t = y_t - y_t|t-1  # Innovation
F_t = Z * P_t|t-1 * Z' + H
K_t = P_t|t-1 * Z' * inv(F_t)
s_t|t = s_t|t-1 + K_t * v_t
P_t|t = P_t|t-1 - K_t * Z * P_t|t-1

# Log-likelihood
logL = -0.5 * sum_t [log|F_t| + v_t' * inv(F_t) * v_t]
```

### Bayesian Estimation
Mode-finding only (no MCMC):
```python
# Posterior = Prior × Likelihood
log_posterior = log_prior(θ) + log_likelihood(θ | data)

# Optimization
θ_mode = argmax_θ log_posterior(θ)
```

Using scipy.optimize with bounds from prior support.

### BVAR with Minnesota Prior
From [repo/BVAR_statistics_usmodel_data.m](repo/BVAR_statistics_usmodel_data.m:16-25):
```
tau = 10        # Tightness
decay = 1       # Lag decay
train = 40      # Training sample observations
```

Prior mean: Random walk (β_i,i = 1, others = 0)
Prior variance: Minnesota specification with presample estimate

## Verification Criteria

Python replication is successful if:
1. State-space matrices (T, R, Z) match Dynare
2. Log-likelihood at mode matches within 1e-4
3. Parameter estimates match within 1%
4. IRFs visually match Dynare output
5. BVAR marginal likelihoods match within 1%
6. Forecast RMSE match within 1%

## Python Dependencies

```
numpy>=1.24.0       # Matrix operations
scipy>=1.10.0       # Optimization, linalg, stats
pandas>=2.0.0       # Data manipulation
matplotlib>=3.7.0   # Plotting
openpyxl>=3.1.0     # Excel reading
```

Optional:
```
sympy>=1.12.0       # Symbolic math (if needed)
jupyter>=1.0.0      # Notebook environment
```

## Implementation Status (Updated: 2026-01-19)

### ✅ Phase 1: Setup - COMPLETED
- [x] memory.md created with full technical documentation
- [x] replication/ directory structure created
- [x] requirements.txt with Python dependencies
- [x] README.md with project overview

### ✅ Phase 2: Utility Modules - COMPLETED
- [x] **priors.py** (368 lines)
  - Beta, Gamma, Normal, InvGamma prior classes
  - Log-density evaluation and random sampling
  - Factory function for creating priors
  - All classes tested and working

- [x] **utils.py** (379 lines)
  - Matrix operations (Lyapunov, vectorization, Kronecker)
  - Stability checks for eigenvalues
  - Autocorrelation and detrending functions
  - HP filter implementation
  - Plotting utilities for IRFs and forecast errors
  - RMSE/MAE computation

- [x] **data_loader.py** (264 lines)
  - Excel and MATLAB file loading (openpyxl, scipy.io)
  - Smets-Wouters data loader function
  - Data transformation utilities
  - VAR lag matrix creation
  - Estimation sample extraction (matches Dynare conventions)
  - Placeholder for Argentine data

### ✅ Phase 3: Core DSGE - COMPLETED
- [x] **solver.py** (368 lines)
  - Full QZ (generalized Schur) decomposition implementation
  - Sims' (2002) method for rational expectations models
  - Blanchard-Kahn condition checking
  - Solution matrices T, R construction
  - Impulse response functions
  - Variance decomposition
  - Stability verification

- [x] **kalman.py** (363 lines)
  - Standard Kalman filter for state-space models
  - Forward filtering with prediction/update steps
  - Log-likelihood computation for DSGE estimation
  - Rauch-Tung-Striebel smoother (backward pass)
  - Diffuse initialization via Lyapunov equation
  - Handles measurement errors

- [x] **model.py** (415 lines)
  - Base DSGEModel class with solver integration
  - SmetsWoutersModel class with parameter specification
  - Prior specification function matching paper
  - Steady-state parameter computation
  - **NOTE**: Equation matrices are PLACEHOLDER - need full translation from usmodel.mod

- [x] **estimation.py** (269 lines)
  - BayesianEstimator class for posterior mode-finding
  - Log-prior, log-likelihood, log-posterior computation
  - Numerical optimization (L-BFGS-B, TNC, SLSQP)
  - Numerical Hessian and variance-covariance matrix
  - Standard errors and t-statistics
  - Results printing utilities

### ✅ Phase 4: BVAR - COMPLETED
- [x] **bvar.py** (329 lines)
  - Bayesian VAR with Minnesota prior
  - OLS and Bayesian estimation
  - Marginal likelihood computation (Laplace approximation)
  - Prior variance from training sample
  - Multi-step forecasting
  - Model comparison across lag orders

- [x] **forecast.py** (304 lines)
  - Recursive forecasting framework
  - RMSE, MAE, bias computation
  - Multivariate forecast statistics (log determinants)
  - Diebold-Mariano test for model comparison
  - Cumulative error calculation for growth rates
  - Results printing utilities

### 🔨 Phase 5: Replication - PENDING
- [ ] **replication.ipynb** - Main notebook demonstrating:
  - [ ] Load Smets-Wouters US data
  - [ ] Solve model with default parameters
  - [ ] Estimate posterior mode (simplified subset)
  - [ ] Generate IRFs for 20 periods
  - [ ] Run BVAR(1) vs BVAR(4) comparison
  - [ ] Recursive forecasts at horizons 1,2,4,8,12
  - [ ] Compare results with paper tables

- [ ] **Complete model.py equations**:
  - [ ] Translate 48 equations from repo/usmodel.mod:78-143
  - [ ] Build Γ0, Γ1, Ψ, Π matrices correctly
  - [ ] Verify solution matches Dynare output
  - [ ] Test with known parameter values

### 📋 Phase 6: Future Work
- [ ] Argentine data acquisition and processing
- [ ] Simplified model version if full estimation fails
- [ ] Robustness checks and sensitivity analysis
- [ ] Comparison with Chari et al. (2009) critiques
- [ ] Final report and documentation

## Key References

1. **Main paper**: Smets & Wouters (2007) - AER
2. **Critique**: Chari, Kehoe & McGrattan (2009) - AEJ Macro
3. **Textbook**: DeJong & Dave (2011) - Structural Macroeconometrics Chapter 3
4. **Related**: Kydland & Prescott (1982), Ireland (2004)

## Notes for Future Sessions

- The model is **already log-linearized** in usmodel.mod - no need to derive first-order conditions
- Fixed parameters (lines 16-20): ctou=0.025, clandaw=1.5, cg=0.18, curvp=10, curvw=10
- Steady-state relationships (lines 56-70) derive additional parameters from estimated ones
- Model has 48 endogenous variables but only 7 are observed
- Presample=4 means first 4 observations initialize Kalman filter
- lik_init=2 means diffuse initialization for non-stationary variables
- Training sample is used for BVAR prior estimation but not DSGE

## Session Summary (2026-01-19)

### What Was Accomplished

This session completed the **foundational infrastructure** for the DSGE replication:

1. **Project Structure**: Created complete directory structure with documentation
2. **9 Python Modules**: Implemented all core functionality (~2,800 lines of code)
3. **Testing**: Each module has basic tests to verify functionality
4. **Documentation**: Comprehensive memory.md and README.md

### Code Statistics

```
replication/
├── __init__.py           (26 lines)
├── priors.py            (368 lines) - Prior distributions
├── utils.py             (379 lines) - Matrix operations, plotting
├── data_loader.py       (264 lines) - Data I/O
├── solver.py            (368 lines) - QZ decomposition
├── kalman.py            (363 lines) - Kalman filter
├── model.py             (415 lines) - DSGE model class
├── estimation.py        (269 lines) - Bayesian estimation
├── bvar.py              (329 lines) - Bayesian VAR
└── forecast.py          (304 lines) - Forecast evaluation
─────────────────────────────────────
Total: ~2,800 lines of documented Python code
```

### Critical Next Steps

1. **Complete model.py** - The BIGGEST remaining task:
   - Current implementation is a **placeholder**
   - Must translate 48 equations from [repo/usmodel.mod](repo/usmodel.mod:78-143)
   - Build proper Γ0, Γ1, Ψ, Π matrices
   - This is complex and requires careful attention to:
     - Index ordering of variables
     - Lead/lag notation (y_t vs y_{t-1} vs E_t[y_{t+1}])
     - Expectational errors vs structural shocks
   - Consider using symbolic computation (sympy) to help

2. **Create replication.ipynb**:
   - Once model equations are complete, demonstrate full workflow
   - Start with simple checks (model solves, IRFs look reasonable)
   - Then attempt estimation (may need to fix parameter subset initially)

3. **Validation**:
   - Compare solution matrices with Dynare output
   - Verify log-likelihood values match
   - Check IRF shapes against paper figures

### Known Issues & Limitations

1. **model.py equations**: Placeholder implementation - will fail on real data
2. **No notebook yet**: Need to create replication.ipynb
3. **Argentine data**: Not yet sourced or processed
4. **Testing**: Basic tests only, need comprehensive validation
5. **Performance**: Not optimized, may be slow for full MCMC (but we're only doing mode-finding)

### Tips for Next Session

**To continue implementation:**

1. Start with model.py equations:
   ```python
   # Read the original Dynare code
   from replication import model

   # Look at equations in repo/usmodel.mod lines 78-143
   # Each equation needs to be mapped to matrix elements
   ```

2. Use the existing infrastructure:
   ```python
   from replication import data_loader, solver, kalman

   # Load data
   data = data_loader.load_smets_wouters_data()

   # Once model is complete:
   # model.solve() -> calls solver.DSGESolver
   # model.log_likelihood(data) -> uses kalman.KalmanFilter
   ```

3. Reference materials:
   - Original equations: [repo/usmodel.mod](repo/usmodel.mod:78-143)
   - Parameter values: [repo/usmodel.mod](repo/usmodel.mod:22-70)
   - Data structure: [repo/readme.pdf](repo/readme.pdf)

**Common pitfalls to avoid:**
- Indexing errors (Python 0-indexed vs Dynare 1-indexed)
- Sign conventions in equations
- Timing of variables (t vs t-1 vs t+1)
- Units (percentages vs decimals, logs vs levels)

### File Locations

- **Main documentation**: [memory.md](memory.md) (this file)
- **Project README**: [README.md](README.md)
- **Python package**: [replication/](replication/)
- **Original code**: [repo/](repo/)
- **Papers**: [papers/](papers/)
- **Proposal**: [idea/David Guzzi - Propuesta Examen de Macroeconometría.pdf](idea/David Guzzi - Propuesta Examen de Macroeconometría.pdf)

## Contact & Context

- **Student**: David Guzzi
- **Course**: Maestría en Econometría
- **Subject**: Macroeconometría
- **Repo**: C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECTMT11
- **Language**: Spanish (documents), Python (code), English (code comments)
- **Last Updated**: 2026-01-19

---

**Ready for next session**: All foundational modules complete. Next priority is completing the model equation specification in model.py, then creating the replication notebook.
