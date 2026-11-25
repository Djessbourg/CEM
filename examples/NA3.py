# -*- coding: utf-8 -*-

# =============================================================================
# Cross-check of F.Arleo and S.Peigné 2013 paper for the NA3 data with their 
# pp and pion-p fit (it works well for all the configurations)
# Last modified: 20/11/2026
# =============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import lhapdf as lha
from scipy.optimize import minimize_scalar
from scipy.special import erfinv
from src import Jpsi as jpsi
from src import Collision, Var_choice, read_info_file
lha_pdf_path = '/opt/homebrew/Cellar/lhapdf/6.5.4/share/LHAPDF' # maybe find an 'os-friendly' way here
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

### Kinematics: 4 diff kin choice for NA3 ###
m_p = 0.938 # GeV
rs = np.array([19.4,16.8,19.4,22.9]) #  GeV, the order is proton 19.4, pion 16.8, pion 19.4, pion 22.9 GeV
s = rs**2 # GeV^2
print("s", s)
p_T = 1 # GeV, mean p_T used for the pT-integrated data (to be adjusted)

### PDF sets ###
proton_name = "CT14nlo"
nuclear_names = {'W': "EPPS16nlo_CT14nlo_W184", 'Be': "EPPS16nlo_CT14nlo_Be9",'Pt': "EPPS16nlo_CT14nlo_Pt195"} # heavy / light target (alpha = W/Be)
n_nuc = 40 # number of nuclear (EPPS16) error members: the members above are the CT14 ones
members = range(0, n_nuc+1)
# members = [0] # quick test
set_label = "EPPS16"

atom_heavy= 'Pt'
Atom = jpsi.Atom
Z_h, A_h = Atom[atom_heavy]['Z'], Atom[atom_heavy]['A']

### Fit settings ###
col_type = 'pp'
var_type = 'xF'
is_pp = Collision(col_type)
is_y = Var_choice(var_type)
if is_y:
	print("Caution ! This script is for x_F, not y")
if is_pp:
	print("pp mode: no nPDF effect, FCEL only (proton set "+proton_name+" for both targets)")
	members = [0] # in pp mode `mem` selects a CT14 member (pdf_p(num)), not an EPPS16 one: 1 fit is enough
	set_label = proton_name
q0_par = 0.075 # GeV^2/fm, fixed q0 for the Arleo-Peigne parametrization (1-x')^n/x'
N_tab = 100 # points of the x_F table of dsigma0/dxF (per member and nucleus)
cl = 68 # confidence level for the hessian uncertainty on q0
### Arleo-Peigne parametrization dsigma/dxF = N (1-x')^n/x' (the normalisation N cancels in R and alpha) ###
use_param = True
n_param = np.array([4.3,1.4,1.4,1.5]) # exponent n in the same order as rs array
f_size = 15
alph = 0.3
a = 5
xF_plot = np.linspace(0, 0.8, 31)

### Initialisation of the sigma objects ###
for name in list(nuclear_names.values()):
	info = read_info_file(os.path.join(lha_pdf_path, name, name+'.info'))
	print(name, '->', info["ErrorType"])
p_set = lha.getPDFSet(proton_name)
p_list = p_set.mkPDFs()
h_set = lha.getPDFSet(nuclear_names[atom_heavy]); h_list = h_set.mkPDFs()
Sigma_h = np.array([jpsi.SigmaJpsi(p_set, p_list, h_set, h_list, cm_energy , Z_h, A_h) for cm_energy in s])

### Model for one member ###
def R_model(model, xF_list, q0,i):
	"""R(x_F) = R_W/R_Be ; q0 = 0 gives the nPDF-only curve"""
	sig_h, sig_l = model
	out = []
	for x in xF_list:
		if q0 > 0 :
			sh = Sigma_h[i].FCEL_integration_xF(x, p_T, sig_h, q0=q0, B=1)[0] # is François using B=1 for the pion case ?
			sl = sig_l(x)
		else:
			sh, sl = sig_h(x), sig_l(x)
		out.append(sh/sl if (sh > 0 and sl > 0) else np.nan)
	return np.array(out)

### Parametrization (1-x')^n/x': no member dependence ###
if use_param: 
	model_param = [[Sigma_h[i].sigma_xF_param(p_T, n_param[i]), Sigma_h[i].sigma_xF_param(p_T, n_param[i])] for i in range(len(n_param))]
	curve_param = [R_model(model, xF_plot, q0_par, i) for i, model in enumerate(model_param)]

### plotting settings ###
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 12,
    "mathtext.fontset": "cm",
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})

q0 = 0.075 # GeV^2/fm, fixed q0 for the Arleo-Peigne parametrization (1-x')^n/x'

# --- À REMPLACER PAR VOS VECTEURS -------------------------------------------
panels = [
    dict(label=r"NA3  $\sqrt{s}=19.4$ GeV",
         xcurve=xF_plot, ycurve=curve_param[0]),
    dict(label=r"NA3  $\sqrt{s}=16.7$ GeV ($\pi^-$ beam)",
         xcurve=xF_plot, ycurve=curve_param[1]),
    dict(label=r"NA3  $\sqrt{s}=19.4$ GeV ($\pi^-$ beam)",
         xcurve=xF_plot, ycurve=curve_param[2]),
    dict(label=r"NA3  $\sqrt{s}=22.9$ GeV ($\pi^-$ beam)",
         xcurve=xF_plot, ycurve=curve_param[3]),
]
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(7, 5.4), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0, wspace=0, left=0.11, right=0.89, top=0.91, bottom=0.11)
fig.suptitle(rf"$\hat{{q}}_0 = {q0}\ \mathrm{{GeV^2/fm}}$", fontsize=14)

ylab = r"$R_{\mathrm{Pt/p}}(x_F)$"

for i, (ax, p) in enumerate(zip(axes.flat, panels)):
    row, col = divmod(i, 2)

    ax.axhline(1, color="k", ls="--", lw=1)
    ax.plot(p["xcurve"], p["ycurve"], color="red", lw=1.3)

    ax.text(0.5, 0.06, p["label"], transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11)

    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1.2)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks(np.arange(0.2, 1.21, 0.2) if row == 0 else np.arange(0, 1.01, 0.2))

    if col == 0:   # colonne de gauche : axe y à gauche
        ax.set_ylabel(ylab, fontsize=13)
    else:          # colonne de droite : axe y à droite
        ax.tick_params(labelleft=False, labelright=True)
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(ylab, fontsize=13, rotation=270, labelpad=18)
    if row == 1:
        ax.set_xlabel(r"$x_F$", fontsize=13)
plt.show()

# ### Plot 1: alpha(x_F), data and fitted curves ###
# fig, ax = plt.subplots(constrained_layout=True)
# ax.xaxis.set_minor_locator(MultipleLocator(0.1))
# ax.xaxis.set_major_locator(MultipleLocator(0.2))
# ax.yaxis.set_minor_locator(MultipleLocator(0.025))
# ax.yaxis.set_major_locator(MultipleLocator(0.1))
# for k in range(1, len(curves)):
# 	plt.plot(xF_plot, curves[k], color='tab:blue', alpha=alph, lw=0.8, label=set_label+' members (fit)' if k == 1 else None)
# plt.plot(xF_plot, curves[0], color='crimson', lw=2, label=set_label+' central (fit), $q_0$ = %.3f' % q0_all[0])
# plt.plot(xF_plot, npdf, color='k', ls='--', lw=1.2, label='no FCEL ($q_0=0$)' if is_pp else 'nPDF only')
# if comparison:
# 	for ratio in ratio_dict.keys():	
# 		plt.plot(xF_plot,ratio_dict[ratio],lw=2, ls='-', label=ratio+r", $(1-x')^n/x'$, $n$ = %.1f, fixed $q_0$ = %.3f" % (n_param, q_0))
# if use_param:
# 	plt.plot(xF_plot, curve_param, color='tab:green', lw=2, ls='-.', label=r"$(1-x')^n/x'$, $n$ = %.1f, $q_0$ = %.3f" % (n_param, q0_par))
# plt.errorbar(xF_all[~mask], data_all[~mask], yerr=err_all[~mask], fmt='o', mfc='none', color='grey', ms=4, label='E866 (not fitted)')
# plt.errorbar(xF_data, data, yerr=err_data, fmt='o', color='k', ms=4, label='E866 (fitted)')
# if Obs == 'alpha':
# 	plt.ylabel(r'$\alpha_{J/\psi}(x_F)$', fontsize=f_size-a)
# elif Obs == 'R':
# 	plt.ylabel(r'$R_\text{W/Be}^{J/\psi}(x_F)$', fontsize=f_size-a)
# plt.xlabel(r'$x_F$', fontsize=f_size)
# plt.xlim(-0.2, 1)
# plt.legend(frameon=False, fontsize=f_size-a-2, loc='lower left')
# plt.tick_params(labelsize=f_size)
# plt.text(0.97, 0.93, r'$\sqrt{s} =$'+'%.1f' % rs + ' GeV', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=f_size-a)
# plt.savefig(os.path.join(plots_dir, set_label+'_E866_Jpsi_'+Obs+'_xF_fit_q0'+'_xF'+str(xF_fit_min)+'-'+str(xF_fit_max)+'.pdf'), bbox_inches="tight")

# ### Plot 2: q0 and chi2_red per member ###
# fig2, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(6, 5))
# ax1.errorbar(res_fit[:, 0], res_fit[:, 1], yerr=res_fit[:, 2], fmt='o', ms=3)
# ax1.set_ylabel(r'$q_0$ [GeV$^2$/fm]', fontsize=f_size-a)
# ax2.plot(res_fit[:, 0], res_fit[:, 4], 'o', ms=3)
# ax2.axhline(1, color='grey', alpha=alph)
# ax2.set_ylabel(r'$\chi^2/{\rm dof}$', fontsize=f_size-a)
# ax2.set_xlabel('member', fontsize=f_size)
# plt.savefig(os.path.join(plots_dir, set_label+'_E866_Jpsi_q0_chi2_members.pdf'), bbox_inches="tight")
# plt.show()