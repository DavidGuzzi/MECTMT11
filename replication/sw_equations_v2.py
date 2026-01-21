"""
Smets-Wouters (2007) Model Equations - Version 2 (CORRECTED)
================================================================

CRITICAL FIX: Equations must be placed in rows corresponding to their variable index.
For example, the equation defining 'pinf' must be in row var_idx['pinf'].

Canonical form: Γ0·y_t = Γ1·y_{t-1} + Ψ·ε_t + Π·η_t

Variable ordering (from model.py):
0-6:   labobs, robs, pinfobs, dy, dc, dinve, dw  (observed/measurement)
7-8:   ewma, epinfma  (MA terms)
9-13:  zcapf, rkf, kf, pkf, cf  (flexible economy 1)
14-18: invef, yf, labf, wf, rrf  (flexible economy 2)
19-27: mc, zcap, rk, k, pk, c, inve, y, lab  (sticky economy)
28-30: pinf, w, r  (sticky economy cont.)
31-37: a, b, g, qs, ms, spinf, sw  (shocks)
38-39: kpf, kp  (predetermined capital)
"""

import numpy as np
from typing import Tuple, Dict


def build_smets_wouters_matrices(params: Dict[str, float],
                                 var_names: list,
                                 shock_names: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build canonical form matrices with equations in correct rows."""

    p = params
    n_vars = len(var_names)  # 40
    n_shocks = len(shock_names)  # 7

    # Variable and shock indices
    var_idx = {name: i for i, name in enumerate(var_names)}
    shock_idx = {name: i for i, name in enumerate(shock_names)}

    # Forward-looking variables
    forward_vars = ['invef', 'rkf', 'pkf', 'cf', 'labf',
                    'inve', 'rk', 'pk', 'c', 'lab', 'pinf', 'w']
    forward_idx = {name: i for i, name in enumerate(forward_vars)}
    n_forward = len(forward_vars)

    # Initialize matrices
    Gamma0 = np.zeros((n_vars, n_vars))
    Gamma1 = np.zeros((n_vars, n_vars))
    Psi = np.zeros((n_vars, n_shocks))
    Pi = np.zeros((n_vars, n_forward))

    # Common coefficients
    h_gamma = p['chabb'] / p['cgamma']
    betabar_gamma = p['cbetabar'] * p['cgamma']
    coef_inv = 1 / (1 + betabar_gamma)

    # ========================================================================
    # MEASUREMENT EQUATIONS (rows 0-6, variables: labobs, robs, pinfobs, dy, dc, dinve, dw)
    # ========================================================================

    # Row 0: labobs = lab + constelab (line 141)
    eq = var_idx['labobs']
    Gamma0[eq, var_idx['labobs']] = 1.0
    Gamma0[eq, var_idx['lab']] = -1.0

    # Row 1: robs = r + conster (line 140)
    eq = var_idx['robs']
    Gamma0[eq, var_idx['robs']] = 1.0
    Gamma0[eq, var_idx['r']] = -1.0

    # Row 2: pinfobs = pinf + constepinf (line 139)
    eq = var_idx['pinfobs']
    Gamma0[eq, var_idx['pinfobs']] = 1.0
    Gamma0[eq, var_idx['pinf']] = -1.0

    # Row 3: dy = y - y(-1) + ctrend (line 135)
    eq = var_idx['dy']
    Gamma0[eq, var_idx['dy']] = 1.0
    Gamma0[eq, var_idx['y']] = -1.0
    Gamma1[eq, var_idx['y']] = 1.0

    # Row 4: dc = c - c(-1) + ctrend (line 136)
    eq = var_idx['dc']
    Gamma0[eq, var_idx['dc']] = 1.0
    Gamma0[eq, var_idx['c']] = -1.0
    Gamma1[eq, var_idx['c']] = 1.0

    # Row 5: dinve = inve - inve(-1) + ctrend (line 137)
    eq = var_idx['dinve']
    Gamma0[eq, var_idx['dinve']] = 1.0
    Gamma0[eq, var_idx['inve']] = -1.0
    Gamma1[eq, var_idx['inve']] = 1.0

    # Row 6: dw = w - w(-1) + ctrend (line 138)
    eq = var_idx['dw']
    Gamma0[eq, var_idx['dw']] = 1.0
    Gamma0[eq, var_idx['w']] = -1.0
    Gamma1[eq, var_idx['w']] = 1.0

    # ========================================================================
    # MA TERMS (rows 7-8)
    # ========================================================================

    # Row 7: ewma = ew (line 130)
    eq = var_idx['ewma']
    Gamma0[eq, var_idx['ewma']] = 1.0
    Psi[eq, shock_idx['ew']] = -1.0

    # Row 8: epinfma = epinf (line 128)
    eq = var_idx['epinfma']
    Gamma0[eq, var_idx['epinfma']] = 1.0
    Psi[eq, shock_idx['epinf']] = -1.0

    # ========================================================================
    # FLEXIBLE ECONOMY (rows 9-18)
    # ========================================================================

    # Row 9: zcapf = (1/(czcap/(1-czcap))) * rkf (line 85)
    eq = var_idx['zcapf']
    Gamma0[eq, var_idx['zcapf']] = 1.0
    Gamma0[eq, var_idx['rkf']] = -(1 / (p['czcap'] / (1 - p['czcap'])))

    # Row 10: rkf = wf + labf - kf (line 86)
    eq = var_idx['rkf']
    Gamma0[eq, var_idx['rkf']] = 1.0
    Gamma0[eq, var_idx['wf']] = -1.0
    Gamma0[eq, var_idx['labf']] = -1.0
    Gamma0[eq, var_idx['kf']] = 1.0

    # Row 11: kf = kpf(-1) + zcapf (line 87)
    eq = var_idx['kf']
    Gamma0[eq, var_idx['kf']] = 1.0
    Gamma1[eq, var_idx['kpf']] = 1.0
    Gamma0[eq, var_idx['zcapf']] = -1.0

    # Row 12: pkf asset price (line 89) - HAS EXPECTATIONS
    eq = var_idx['pkf']
    coef_b = 1 / ((1 - h_gamma) / (p['csigma'] * (1 + h_gamma)))
    coef_rk_lead = p['crk'] / (p['crk'] + (1 - p['ctou']))
    coef_pk_lead = (1 - p['ctou']) / (p['crk'] + (1 - p['ctou']))
    Gamma0[eq, var_idx['pkf']] = 1.0
    Gamma0[eq, var_idx['rrf']] = 1.0
    Gamma0[eq, var_idx['b']] = -coef_b
    Pi[eq, forward_idx['rkf']] = -coef_rk_lead
    Pi[eq, forward_idx['pkf']] = -coef_pk_lead

    # Row 13: cf consumption Euler (line 90) - HAS EXPECTATION
    eq = var_idx['cf']
    coef_c_lag = h_gamma / (1 + h_gamma)
    coef_c_lead = 1 / (1 + h_gamma)
    coef_lab = (p['csigma'] - 1) * p['cwhlc'] / (p['csigma'] * (1 + h_gamma))
    coef_r = (1 - h_gamma) / (p['csigma'] * (1 + h_gamma))
    Gamma0[eq, var_idx['cf']] = 1.0
    Gamma1[eq, var_idx['cf']] = coef_c_lag
    Pi[eq, forward_idx['cf']] = -coef_c_lead
    Gamma0[eq, var_idx['labf']] = -coef_lab
    Pi[eq, forward_idx['labf']] = coef_lab
    Gamma0[eq, var_idx['rrf']] = coef_r
    Gamma0[eq, var_idx['b']] = -1.0

    # Row 14: invef investment (line 88) - HAS EXPECTATION
    eq = var_idx['invef']
    Gamma0[eq, var_idx['invef']] = 1.0
    Gamma1[eq, var_idx['invef']] = -coef_inv
    Pi[eq, forward_idx['invef']] = -coef_inv * betabar_gamma
    Gamma0[eq, var_idx['pkf']] = -coef_inv * (1 / (p['cgamma']**2 * p['csadjcost']))
    Gamma0[eq, var_idx['qs']] = -1.0

    # Row 15: yf production function (line 92)
    eq = var_idx['yf']
    Gamma0[eq, var_idx['yf']] = 1.0
    Gamma0[eq, var_idx['kf']] = -p['cfc'] * p['calfa']
    Gamma0[eq, var_idx['labf']] = -p['cfc'] * (1 - p['calfa'])
    Gamma0[eq, var_idx['a']] = -p['cfc']

    # Row 16: labf is defined by resource constraint (line 91)
    # yf = ccy*cf + ciy*invef + g + crkky*zcapf
    eq = var_idx['labf']
    # But wait - line 91 defines yf, not labf
    # Let me use the alternative: a = calfa*rkf + (1-calfa)*wf (line 84)
    # Rearranged for labf using rkf = wf + labf - kf
    # Actually, I need to use a different equation. Let me use the wage equation (line 93):
    # wf = csigl*labf + (1/(1-h/γ))*cf - (h/γ)/(1-h/γ)*cf(-1)
    coef_c_cur = 1 / (1 - h_gamma)
    coef_c_lag_w = h_gamma / (1 - h_gamma)
    Gamma0[eq, var_idx['labf']] = p['csigl']
    Gamma0[eq, var_idx['wf']] = -1.0
    Gamma0[eq, var_idx['cf']] = coef_c_cur
    Gamma1[eq, var_idx['cf']] = coef_c_lag_w

    # Row 17: wf is defined by flexible price condition (line 84)
    # a = calfa*rkf + (1-calfa)*wf
    eq = var_idx['wf']
    Gamma0[eq, var_idx['a']] = -1.0
    Gamma0[eq, var_idx['rkf']] = p['calfa']
    Gamma0[eq, var_idx['wf']] = (1 - p['calfa'])

    # Row 18: rrf is determined by resource constraint (line 91)
    # yf = ccy*cf + ciy*invef + g + crkky*zcapf
    eq = var_idx['rrf']
    Gamma0[eq, var_idx['yf']] = 1.0
    Gamma0[eq, var_idx['cf']] = -p['ccy']
    Gamma0[eq, var_idx['invef']] = -p['ciy']
    Gamma0[eq, var_idx['g']] = -1.0
    Gamma0[eq, var_idx['zcapf']] = -p['crkky']

    # ========================================================================
    # STICKY ECONOMY (rows 19-30)
    # ========================================================================

    # Row 19: mc = calfa*rk + (1-calfa)*w - a (line 98)
    eq = var_idx['mc']
    Gamma0[eq, var_idx['mc']] = 1.0
    Gamma0[eq, var_idx['rk']] = -p['calfa']
    Gamma0[eq, var_idx['w']] = -(1 - p['calfa'])
    Gamma0[eq, var_idx['a']] = 1.0

    # Row 20: zcap = (1/(czcap/(1-czcap))) * rk (line 99)
    eq = var_idx['zcap']
    Gamma0[eq, var_idx['zcap']] = 1.0
    Gamma0[eq, var_idx['rk']] = -(1 / (p['czcap'] / (1 - p['czcap'])))

    # Row 21: rk = w + lab - k (line 100)
    eq = var_idx['rk']
    Gamma0[eq, var_idx['rk']] = 1.0
    Gamma0[eq, var_idx['w']] = -1.0
    Gamma0[eq, var_idx['lab']] = -1.0
    Gamma0[eq, var_idx['k']] = 1.0

    # Row 22: k = kp(-1) + zcap (line 101)
    eq = var_idx['k']
    Gamma0[eq, var_idx['k']] = 1.0
    Gamma1[eq, var_idx['kp']] = 1.0
    Gamma0[eq, var_idx['zcap']] = -1.0

    # Row 23: pk asset price (line 103) - HAS EXPECTATIONS
    eq = var_idx['pk']
    Gamma0[eq, var_idx['pk']] = 1.0
    Gamma0[eq, var_idx['r']] = 1.0
    Pi[eq, forward_idx['pinf']] = -1.0
    Gamma0[eq, var_idx['b']] = -coef_b
    Pi[eq, forward_idx['rk']] = -coef_rk_lead
    Pi[eq, forward_idx['pk']] = -coef_pk_lead

    # Row 24: c consumption Euler (line 104) - HAS EXPECTATION
    eq = var_idx['c']
    Gamma0[eq, var_idx['c']] = 1.0
    Gamma1[eq, var_idx['c']] = coef_c_lag
    Pi[eq, forward_idx['c']] = -coef_c_lead
    Gamma0[eq, var_idx['lab']] = -coef_lab
    Pi[eq, forward_idx['lab']] = coef_lab
    Gamma0[eq, var_idx['r']] = coef_r
    Pi[eq, forward_idx['pinf']] = -coef_r
    Gamma0[eq, var_idx['b']] = -1.0

    # Row 25: inve investment (line 102) - HAS EXPECTATION
    eq = var_idx['inve']
    Gamma0[eq, var_idx['inve']] = 1.0
    Gamma1[eq, var_idx['inve']] = -coef_inv
    Pi[eq, forward_idx['inve']] = -coef_inv * betabar_gamma
    Gamma0[eq, var_idx['pk']] = -coef_inv * (1 / (p['cgamma']**2 * p['csadjcost']))
    Gamma0[eq, var_idx['qs']] = -1.0

    # Row 26: y production function (line 106)
    eq = var_idx['y']
    Gamma0[eq, var_idx['y']] = 1.0
    Gamma0[eq, var_idx['k']] = -p['cfc'] * p['calfa']
    Gamma0[eq, var_idx['lab']] = -p['cfc'] * (1 - p['calfa'])
    Gamma0[eq, var_idx['a']] = -p['cfc']

    # Row 27: lab is determined by resource constraint (line 105)
    # y = ccy*c + ciy*inve + g + crkky*zcap
    eq = var_idx['lab']
    Gamma0[eq, var_idx['y']] = 1.0
    Gamma0[eq, var_idx['c']] = -p['ccy']
    Gamma0[eq, var_idx['inve']] = -p['ciy']
    Gamma0[eq, var_idx['g']] = -1.0
    Gamma0[eq, var_idx['zcap']] = -p['crkky']

    # Row 28: pinf Phillips curve (lines 107-108) - HAS EXPECTATION
    eq = var_idx['pinf']
    denom_p = 1 + betabar_gamma * p['cindp']
    coef_pinf_lead = betabar_gamma / denom_p
    coef_pinf_lag = p['cindp'] / denom_p
    kappa_p = ((1 - p['cprobp']) * (1 - betabar_gamma * p['cprobp']) / p['cprobp']) / ((p['cfc'] - 1) * p['curvp'] + 1)
    coef_mc = kappa_p / denom_p
    Gamma0[eq, var_idx['pinf']] = 1.0
    Pi[eq, forward_idx['pinf']] = -coef_pinf_lead
    Gamma1[eq, var_idx['pinf']] = coef_pinf_lag
    Gamma0[eq, var_idx['mc']] = -coef_mc
    Gamma0[eq, var_idx['spinf']] = -1.0

    # Row 29: w wage Phillips curve (lines 109-116) - HAS EXPECTATION
    eq = var_idx['w']
    denom_w = 1 + betabar_gamma
    coef_w_lag = 1 / denom_w
    coef_w_lead = betabar_gamma / denom_w
    coef_pinf_lag_w = p['cindw'] / denom_w
    coef_pinf_cur_w = -(1 + betabar_gamma * p['cindw']) / denom_w
    coef_pinf_lead_w = betabar_gamma / denom_w
    kappa_w = (1 - p['cprobw']) * (1 - betabar_gamma * p['cprobw']) / (denom_w * p['cprobw'])
    coef_wage_gap = kappa_w * (1 / ((p['clandaw'] - 1) * p['curvw'] + 1))

    Gamma0[eq, var_idx['w']] = 1.0 + coef_wage_gap
    Gamma1[eq, var_idx['w']] = coef_w_lag
    Pi[eq, forward_idx['w']] = -coef_w_lead
    Gamma1[eq, var_idx['pinf']] = coef_pinf_lag_w
    Gamma0[eq, var_idx['pinf']] = -coef_pinf_cur_w
    Pi[eq, forward_idx['pinf']] = -coef_pinf_lead_w
    Gamma0[eq, var_idx['lab']] = -coef_wage_gap * p['csigl']
    Gamma0[eq, var_idx['c']] = -coef_wage_gap * (1 / (1 - h_gamma))
    Gamma1[eq, var_idx['c']] = -coef_wage_gap * (h_gamma / (1 - h_gamma))
    Gamma0[eq, var_idx['sw']] = -1.0

    # Row 30: r Taylor rule (lines 117-121)
    eq = var_idx['r']
    Gamma0[eq, var_idx['r']] = 1.0
    Gamma0[eq, var_idx['pinf']] = -p['crpi'] * (1 - p['crr'])
    Gamma0[eq, var_idx['y']] = -(p['cry'] * (1 - p['crr']) + p['crdy'])
    Gamma0[eq, var_idx['yf']] = p['cry'] * (1 - p['crr']) + p['crdy']
    Gamma1[eq, var_idx['y']] = -p['crdy']
    Gamma1[eq, var_idx['yf']] = p['crdy']
    Gamma1[eq, var_idx['r']] = p['crr']
    Gamma0[eq, var_idx['ms']] = -1.0

    # ========================================================================
    # SHOCK PROCESSES (rows 31-37)
    # ========================================================================

    # Row 31: a = crhoa*a(-1) + ea (line 122)
    eq = var_idx['a']
    Gamma0[eq, var_idx['a']] = 1.0
    Gamma1[eq, var_idx['a']] = p['crhoa']
    Psi[eq, shock_idx['ea']] = -1.0

    # Row 32: b = crhob*b(-1) + eb (line 123)
    eq = var_idx['b']
    Gamma0[eq, var_idx['b']] = 1.0
    Gamma1[eq, var_idx['b']] = p['crhob']
    Psi[eq, shock_idx['eb']] = -1.0

    # Row 33: g = crhog*g(-1) + eg + cgy*ea (line 124)
    eq = var_idx['g']
    Gamma0[eq, var_idx['g']] = 1.0
    Gamma1[eq, var_idx['g']] = p['crhog']
    Psi[eq, shock_idx['eg']] = -1.0
    Psi[eq, shock_idx['ea']] = -p['cgy']

    # Row 34: qs = crhoqs*qs(-1) + eqs (line 125)
    eq = var_idx['qs']
    Gamma0[eq, var_idx['qs']] = 1.0
    Gamma1[eq, var_idx['qs']] = p['crhoqs']
    Psi[eq, shock_idx['eqs']] = -1.0

    # Row 35: ms = crhoms*ms(-1) + em (line 126)
    eq = var_idx['ms']
    Gamma0[eq, var_idx['ms']] = 1.0
    Gamma1[eq, var_idx['ms']] = p['crhoms']
    Psi[eq, shock_idx['em']] = -1.0

    # Row 36: spinf = crhopinf*spinf(-1) + epinfma - cmap*epinfma(-1) (line 127)
    eq = var_idx['spinf']
    Gamma0[eq, var_idx['spinf']] = 1.0
    Gamma1[eq, var_idx['spinf']] = p['crhopinf']
    Gamma0[eq, var_idx['epinfma']] = -1.0
    Gamma1[eq, var_idx['epinfma']] = -p['cmap']

    # Row 37: sw = crhow*sw(-1) + ewma - cmaw*ewma(-1) (line 129)
    eq = var_idx['sw']
    Gamma0[eq, var_idx['sw']] = 1.0
    Gamma1[eq, var_idx['sw']] = p['crhow']
    Gamma0[eq, var_idx['ewma']] = -1.0
    Gamma1[eq, var_idx['ewma']] = -p['cmaw']

    # ========================================================================
    # CAPITAL ACCUMULATION (rows 38-39)
    # ========================================================================

    # Row 38: kpf = (1-cikbar)*kpf(-1) + cikbar*invef + cikbar*γ²*adj*qs (line 94)
    eq = var_idx['kpf']
    Gamma0[eq, var_idx['kpf']] = 1.0
    Gamma1[eq, var_idx['kpf']] = (1 - p['cikbar'])
    Gamma0[eq, var_idx['invef']] = -p['cikbar']
    Gamma0[eq, var_idx['qs']] = -p['cikbar'] * p['cgamma']**2 * p['csadjcost']

    # Row 39: kp = (1-cikbar)*kp(-1) + cikbar*inve + cikbar*γ²*adj*qs (line 131)
    eq = var_idx['kp']
    Gamma0[eq, var_idx['kp']] = 1.0
    Gamma1[eq, var_idx['kp']] = (1 - p['cikbar'])
    Gamma0[eq, var_idx['inve']] = -p['cikbar']
    Gamma0[eq, var_idx['qs']] = -p['cikbar'] * p['cgamma']**2 * p['csadjcost']

    return Gamma0, Gamma1, Psi, Pi


if __name__ == '__main__':
    print("Smets-Wouters equations module v2 (corrected indexing)")
    print("This version ensures each equation is in the correct row for its variable.")
