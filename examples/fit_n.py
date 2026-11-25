# -*- coding: utf-8 -*-

# =============================================================================
# Fit of the exponent n of the pp cross section parametrisation of Arleo-Peigne (JHEP03(2013)122, Eq. 3.10)
#     dsigma/dxF  ~  N (1 - x')^n / x'    with    x' = x1 + x2 = sqrt(xF^2 + 4 M_T^2/s) = 2 M_T/sqrt(s) cosh(y)
# on the E789 data (J/psi, p-Be and p-Cu, 800 GeV/c). The normalisation N is free (profiled analytically).
# Last modified: 22/10/2026
# =============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.optimize import minimize_scalar
from src import Jpsi as jpsi
from src import Collision, Var_choice

### Kinematics ###
m_p = 0.938 # GeV
rs = np.sqrt(2*m_p*800.+2*m_p**2) # 38.8 GeV (the paper quotes 38.7)
s = rs**2
p_T = 1. # GeV, as in Arleo-Peigne (M = 3 GeV, p_T = 1 GeV)
M_T = np.sqrt(jpsi.m_jpsi**2 + p_T**2) # same M_T as in SigmaJpsi.M_jpsi_T

### Fit settings ###
col_type = 'pA'
var_type = 'xF'
is_pp = Collision(col_type)
is_y = Var_choice(var_type)
if is_pp or is_y:
	raise ValueError("This script is for pA data as a function of x_F")
target = 'Be' # 'Be' (used by Arleo-Peigne as a pp-like reference) or 'Cu'
xF_fit_min, xF_fit_max = 0.3, 0.8
use_syst = True # add the diagonal systematic error in quadrature to the statistical one
n_bounds = (0.5, 15.)
n_AP = 4.5 # Arleo-Peigne Table 1 (E789 p-Be): n = 4.5 +/- 0.05, for comparison
f_size = 15
a = 5

fit_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fit_dir'))
os.makedirs(fit_dir, exist_ok=True)
data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'E789_Jpsi_dsigma_dxF.csv'))

### Data ###
def load_E789(path, target):
	"""returns xF, (1/A)dsigma/dxF [nb/nucleon], total error (the Be upper-limit bin, nan, is dropped)"""
	d = np.loadtxt(path, comments='#', ndmin=2)
	col = {'Cu': 1, 'Be': 4}[target]
	xF, y, stat, syst = d[:,0], d[:,col], d[:,col+1], d[:,col+2]
	err = np.sqrt(stat**2 + (syst**2 if use_syst else 0.))
	ok = np.isfinite(y)
	return xF[ok], y[ok], err[ok]

xF_all, y_all, err_all = load_E789(data_file, target)
mask = (xF_all >= xF_fit_min) & (xF_all <= xF_fit_max)
xF_d, y_d, err_d = xF_all[mask], y_all[mask], err_all[mask]
N_data = len(xF_d)
dof = N_data - 2 # free parameters: n and the normalisation

### Model ###
def x_prime(xF):
	return np.sqrt(np.asarray(xF)**2 + 4.*M_T**2/s)

def shape(xF, n):
	xp = x_prime(xF)
	return (1.-xp)**n/xp

def norm_hat(n):
	"""best normalisation for a given n (linear least squares)"""
	f = shape(xF_d, n)
	return np.sum(y_d*f/err_d**2)/np.sum(f**2/err_d**2)

def chi2(n):
	return float(np.sum(((y_d - norm_hat(n)*shape(xF_d, n))/err_d)**2))

### Fit ###
res = minimize_scalar(chi2, bounds=n_bounds, method='bounded', options={'xatol': 1e-6})
n_fit, chi2_min = res.x, res.fun
h = 0.05
curv = (chi2(n_fit+h) - 2*chi2_min + chi2(n_fit-h))/h**2
dn = np.sqrt(2./curv) if curv > 0 else np.nan # Delta chi2 = 1 (N profiled)
N_fit = norm_hat(n_fit)
print(f"E789 p-{target}, xF in [{xF_fit_min}, {xF_fit_max}], {N_data} points")
print(f"n = {n_fit:.3f} +/- {dn:.3f} | N = {N_fit:.3g} nb/nucleon | chi2/dof = {chi2_min:.2f}/{dof} = {chi2_min/dof:.2f}")
print(f"(Arleo-Peigne, p-Be: n = {n_AP} +/- 0.05; chi2/dof with n = {n_AP} and refitted N: {chi2(n_AP)/dof:.2f})")
np.savetxt(os.path.join(fit_dir, f'E789_n_fit_{target}_xF{xF_fit_min}-{xF_fit_max}.txt'), [[n_fit, dn, N_fit, chi2_min, chi2_min/dof]], header="n dn N[nb/nucleon] chi2 chi2_red (dof=%d)" % dof)

### Plot ###
xF_plot = np.linspace(0.25, 0.95, 200)
fig, (ax, axr) = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]})
ax.errorbar(xF_all[~mask], y_all[~mask], yerr=err_all[~mask], fmt='o', mfc='none', color='grey', ms=4, label='E789 p-'+target+' (not fitted)')
ax.errorbar(xF_d, y_d, yerr=err_d, fmt='o', color='k', ms=4, label='E789 p-'+target+' (fitted)')
ax.plot(xF_plot, N_fit*shape(xF_plot, n_fit), color='crimson', lw=2, label=r'fit: $n$ = %.2f $\pm$ %.2f' % (n_fit, dn))
ax.fill_between(xF_plot, N_fit*shape(xF_plot, n_fit+dn), N_fit*shape(xF_plot, n_fit-dn), color='crimson', alpha=0.3)
ax.plot(xF_plot, norm_hat(n_AP)*shape(xF_plot, n_AP), color='tab:blue', ls='--', label=r'$n$ = %.1f (Arleo-Peigné)' % n_AP)
ax.set_yscale('log')
ax.set_ylabel(r'$(1/A)\,d\sigma/dx_F$ [nb/nucleon]', fontsize=f_size-a)
ax.legend(frameon=False, fontsize=f_size-a-2, loc='upper right')
ax.tick_params(labelsize=f_size-2)
axr.errorbar(xF_all, (y_all - N_fit*shape(xF_all, n_fit))/err_all, fmt='o', color='k', ms=4)
axr.axhline(0, color='grey', alpha=0.5)
axr.set_ylabel('pull', fontsize=f_size-a)
axr.set_xlabel(r'$x_F$', fontsize=f_size)
axr.xaxis.set_major_locator(MultipleLocator(0.1))
axr.tick_params(labelsize=f_size-2)
plt.savefig(os.path.join(plots_dir, f'E789_Jpsi_fit_n_{target}.pdf'), bbox_inches="tight")
plt.show()