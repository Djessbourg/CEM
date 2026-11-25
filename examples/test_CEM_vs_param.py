# -*- coding: utf-8 -*-

# =============================================================================
# Test the different ways of reproducing the cross section
# Last modified: 05/11/2026
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
import lhapdf as lha

### Kinematics ###
m_p = 0.938 # GeV
rs = np.sqrt(2*m_p*800.+2*m_p**2) # 38.8 GeV (the paper quotes 38.7)
s = rs**2
p_T = 1. # GeV, as in Arleo-Peigne (M = 3 GeV, p_T = 1 GeV)
M_T = np.sqrt(jpsi.m_jpsi**2 + p_T**2) # same M_T as in SigmaJpsi.M_jpsi_T

### Fit settings ###
col_type = 'pp'
var_type = 'xF'
is_pp = Collision(col_type)
is_y = Var_choice(var_type)
if is_pp or is_y:
	print("Caution! This script is for pA data as a function of x_F")
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
N_fit,n_fit,dn = 609,4.56,0.05 # results from previous fit 

### CEM ###
proton ={"nNNPDF30 noLHCbD" : "nNNPDF30_nlo_as_0118_noLHCbD_p",
         "NNPDF40" : "NNPDF40_nlo_as_01180",
         "EPPS16": "CT14nlo"
         } 
nuclear_names = {'Be': {"nNNPDF30 noLHCbD":"nNNPDF30_nlo_as_0118_noLHCbD_A9_Z4",
                        "NNPDF40":"nNNPDF30_nlo_as_0118_noLHCbD_A9_Z4", #Pas le bon
                        "EPPS16":"EPPS16nlo_CT14nlo_Be9"}
                 }
atom = 'Be'
nuclei = nuclear_names[atom]
Atom = jpsi.Atom
A,Z = Atom[atom]['A'],Atom[atom]['Z']

p_set = {prt:lha.getPDFSet(proton[prt]) for prt in proton.keys()}
A_set = {nuc:lha.getPDFSet(nuclei[nuc]) for nuc in nuclei.keys()}

p_list = {p:p_set[p].mkPDFs() for p in p_set.keys()}
A_list = {At:A_set[At].mkPDFs() for At in A_set.keys()}

sigma = {pdf_name:jpsi.SigmaJpsi(p_set[pdf_name],p_list[pdf_name],A_set[pdf_name],A_list[pdf_name],s,Z,A) for pdf_name in proton.keys()}

X_F = np.linspace(0.25, 0.95, 100)

mu_fact = 2
mu_f_fact = mu_fact

CEM = {pdf_name:np.array([sigma[pdf_name].dsigma_tot_dvar(xf,p_T,0,mu_factor=mu_fact,mu_f_factor=mu_f_fact,Var_y=is_y,is_pp=is_pp)[0] for xf in X_F]) for pdf_name in proton.keys()}
# CEM_gg = {pdf_name:np.array([sigma[pdf_name].dsigma_gg_dvar(xf,p_T,0,mu_factor=mu_fact,mu_f_factor=mu_f_fact,Var_y=is_y,is_pp=is_pp)[0] for xf in X_F]) for pdf_name in proton.keys()}
# CEM_qqbar= {pdf_name:np.array([sigma[pdf_name].dsigma_qqbar_dvar(xf,p_T,0,mu_factor=mu_fact,mu_f_factor=mu_f_fact,Var_y=is_y,is_pp=is_pp)[0] for xf in X_F]) for pdf_name in proton.keys()}
### We rescale it to the param one ###
i = 0 # index of the xf to scale

scl = {pdf_name:N_fit*shape(X_F[i], n_fit)/np.copy(CEM[pdf_name][i]) for pdf_name in proton.keys()}

CEM_scl = {pdf_name:CEM[pdf_name]*scl[pdf_name] for pdf_name in proton.keys()}
# CEM_gg_scl = {pdf_name:CEM_gg[pdf_name]*scl[pdf_name] for pdf_name in proton.keys()}
# CEM_qqbar_scl = {pdf_name:CEM_qqbar[pdf_name]*scl[pdf_name] for pdf_name in proton.keys()}

# print(CEM['nNNPDF30 noLHCbD'])

CEM_colors ={"nNNPDF30 noLHCbD":"dodgerblue",
             "NNPDF40": "orangered",
             "EPPS16":"olivedrab"
			 }

### Plot ###
xF_plot = X_F
fig,ax = plt.subplots()
ax.errorbar(xF_all[~mask], y_all[~mask], yerr=err_all[~mask], fmt='o', mfc='none', color='grey', ms=4, label='E789 p-'+target+' (not fitted)')
ax.errorbar(xF_d, y_d, yerr=err_d, fmt='o', color='k', ms=4, label='E789 p-'+target+' (fitted)')
ax.plot(xF_plot, N_fit*shape(xF_plot, n_fit), color='crimson', lw=2, label=r'fit: $n$ = %.2f $\pm$ %.2f' % (n_fit, dn))
ax.fill_between(xF_plot, N_fit*shape(xF_plot, n_fit+dn), N_fit*shape(xF_plot, n_fit-dn), color='crimson', alpha=0.3)
for pdf_name in list(proton.keys()):
	ax.plot(xF_plot, CEM_scl[pdf_name], color=CEM_colors[pdf_name], lw=2, label=f'CEM {pdf_name} ( /{np.round(1/scl[pdf_name])})')
	# if pdf_name == "nNNPDF30 noLHCbD":
	# 	ax.plot(xF_plot, CEM_gg_scl[pdf_name],linestyle = 'dashed', color=CEM_colors[pdf_name], lw=2, label=f'gg')
	# 	ax.plot(xF_plot, CEM_qqbar_scl[pdf_name],linestyle= 'dotted',color=CEM_colors[pdf_name], lw=2, label=f'qqbar')
# ax.plot(xF_plot, norm_hat(n_AP)*shape(xF_plot, n_AP), color='tab:blue', ls='--', label=r'$n$ = %.1f (Arleo-Peigné)' % n_AP)
ax.set_yscale('log')
# ax.set_ylim(0,10)
ax.set_ylabel(r'$(1/A)\,d\sigma/dx_F$ [nb/nucleon]', fontsize=f_size-a)
# ax.set_xlim(0.7,1)
ax.legend(frameon=False, fontsize=f_size-a-2, loc='upper right')
ax.tick_params(labelsize=f_size-2)
ax.set_xlabel(r'$x_F$')
plt.savefig(os.path.join(plots_dir, f'E789_Jpsi_fit_n_vs_CEM_{target}.pdf'), bbox_inches="tight")
plt.show()