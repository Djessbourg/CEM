# -*- coding: utf-8 -*-

# =============================================================================
# Fit of the transport coefficient q0 on the E866 J/psi data (alpha(x_F), p+W / p+Be at 800 GeV)
# For each member of a nPDF set (here EPPS16): best q0 and reduced chi2, and plot of all the fitted curves.
# FCEL at LO in x_F (see Jpsi.SigmaJpsi.FCEL_integration_xF)
# Cross-check: same fit with the Arleo-Peigne parametrization dsigma/dxF ~ (1-x')^n/x' (JHEP03(2013)122, Eq. 3.10), see use_param
# Last modified: 16/11/2026
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

### Kinematics: E866 / NuSea, 800 GeV proton beam ###
m_p = 0.938 # GeV
p_beam= 800 # GeV
s = 2*m_p*p_beam+2*m_p**2
rs = np.sqrt(s) # = 38.8 GeV
p_T = 1 # GeV, mean p_T used for the pT-integrated data (to be adjusted)

### PDF sets ###
proton_name = "CT14nlo"
nuclear_names = {'W': "EPPS16nlo_CT14nlo_W184", 'Be': "EPPS16nlo_CT14nlo_Be9"} # heavy / light target (alpha = W/Be)
n_nuc = 40 # number of nuclear (EPPS16) error members: the members above are the CT14 ones
members = range(0, n_nuc+1)
# members = [0] # quick test
set_label = "EPPS16"

atom_heavy, atom_light = 'W', 'Be'
Atom = jpsi.Atom
Z_h, A_h = Atom[atom_heavy]['Z'], Atom[atom_heavy]['A']
Z_l, A_l = Atom[atom_light]['Z'], Atom[atom_light]['A']

### Fit settings ###
# col_type = 'pp'
col_type = 'pA'
var_type = 'xF'
Obs = 'R' #or "alpha"
is_pp = Collision(col_type)
is_y = Var_choice(var_type)
if is_y:
	print("Caution ! This script is for x_F, not y")
if is_pp:
	print("pp mode: no nPDF effect, FCEL only (proton set "+proton_name+" for both targets)")
	members = [0] # in pp mode `mem` selects a CT14 member (pdf_p(num)), not an EPPS16 one: 1 fit is enough
	set_label = proton_name
mu = 2
mu_f = mu
q0_scan = np.geomspace(0.01, 0.3, 10) # coarse scan (GeV^2/fm), then bounded refinement around the minimum
N_tab = 100 # points of the x_F table of dsigma0/dxF (per member and nucleus)
cl = 68 # confidence level for the hessian uncertainty on q0
### Arleo-Peigne parametrization dsigma/dxF = N (1-x')^n/x' (the normalisation N cancels in R and alpha) ###
use_param = True
n_param = 4.5 # exponent n: default value of Jpsi.sigma_Jpsi, to be replaced by the value fitted on the pp data
use_norm_in_chi2 = True # add the 'norm' column (relative, 3%) in quadrature to the errors of the fit
f_size = 15
alph = 0.3
a = 5

fit_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fit_dir')) # the directory to save data from this file
os.makedirs(fit_dir, exist_ok=True)
data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'E866_Jpsi_alpha_xF.csv'))
data_R_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'e866_pw_pbe.dat'))

### Data ###
# xF_fit_min, xF_fit_max = -0.2, 0.95
xF_fit_min, xF_fit_max = 0.2, 0.8
if Obs =="alpha":
# alpha version
	def load_E866(path):
		"""columns: x_F, alpha, d_alpha_stat [, d_alpha_syst]; stat and syst added in quadrature"""
		if not os.path.exists(path):
			raise FileNotFoundError("Data file not found: "+path)
		d = np.loadtxt(path, comments='#', ndmin=2)
		if d.size == 0:
			raise ValueError("The data file is empty: fill "+path+" with E866 alpha(x_F) (x_F, alpha, stat, [syst])")
		err = np.sqrt(d[:,2]**2 + (d[:,3]**2 if d.shape[1] > 3 else 0.))
		return d[:,0], d[:,1], err

	xF_all, data_all, err_all = load_E866(data_file)
	mask = (xF_all >= xF_fit_min) & (xF_all <= xF_fit_max)
	xF_data, data, err_data ,norm = xF_all[mask], data_all[mask], err_all[mask], np.zeros(shape=int(mask.sum()))

elif Obs =="R":
	# R = pW/pBe version
	xF_all, data_all, err_all, norm_all = np.loadtxt(data_R_file, skiprows=1, unpack=True)
	mask = (xF_all >= xF_fit_min) & (xF_all <= xF_fit_max)
	xF_data, data, err_data ,norm = xF_all[mask], data_all[mask], err_all[mask], norm_all[mask]
	if use_norm_in_chi2:
		err_data = np.hypot(err_data, norm*data)

N_data = len(xF_data)
dof = N_data - 1 # one free parameter: q0
xF_plot = np.linspace(-0.2, 0.95, 21)
xF_tab_min = min(xF_data.min(), xF_plot.min()) - 0.03

### Hessian tools (same as RpA_Jpsi.py) ###
def ConfLevel_to_sigma(b):
	"b en % (genre 90% => b = 90)"
	return erfinv(b/100.)*(2**0.5)

def hess_error(H, cl=90): # H matrice [x_id][mem], asymmetric
	factor = ConfLevel_to_sigma(cl)/ConfLevel_to_sigma(90)
	Nx = len(H)
	Neig = (len(H[0])-1)//2
	cen = [H[i][0] for i in range(Nx)]
	Oplus = [[max(cen[i], H[i][2*k-1], H[i][2*k]) for i in range(Nx)] for k in range(1, Neig+1)]
	Ominus = [[min(cen[i], H[i][2*k-1], H[i][2*k]) for i in range(Nx)] for k in range(1, Neig+1)]
	delta_plus = [np.sqrt(sum([(cen[i]-Oplus[k][i])**2 for k in range(Neig)])) for i in range(Nx)]
	delta_minus = [np.sqrt(sum([(cen[i]-Ominus[k][i])**2 for k in range(Neig)])) for i in range(Nx)]
	return [np.array(cen), factor*np.array(delta_plus), factor*np.array(delta_minus)]

### Initialisation of the sigma objects ###
for name in list(nuclear_names.values()):
	info = read_info_file(os.path.join(lha_pdf_path, name, name+'.info'))
	print(name, '->', info["ErrorType"])
p_set = lha.getPDFSet(proton_name)
p_list = p_set.mkPDFs()
h_set = lha.getPDFSet(nuclear_names[atom_heavy]); h_list = h_set.mkPDFs()
l_set = lha.getPDFSet(nuclear_names[atom_light]); l_list = l_set.mkPDFs()
Sigma_h = jpsi.SigmaJpsi(p_set, p_list, h_set, h_list, s, Z_h, A_h)
Sigma_l = jpsi.SigmaJpsi(p_set, p_list, l_set, l_list, s, Z_l, A_l)

### Model for one member ###
def build_member(mem):
	"""tables of dsigma0/dxF (nPDF only, proton at its central member) for the heavy and light targets"""
	kw = dict(mu_factor=mu, mu_f_factor=mu_f, is_pp=is_pp, xF_min=xF_tab_min, n_pts=N_tab)
	return [Sigma_h.sigma_xF_interp(p_T, mem, **kw), Sigma_l.sigma_xF_interp(p_T, mem, **kw)]

def alpha_model(model, xF_list, q0):
	"""alpha(x_F) = 1 + ln(R_W/R_Be)/ln(A_W/A_Be); q0 = 0 gives the nPDF-only curve"""
	sig_h, sig_l = model
	out = []
	for x in xF_list:
		if q0 > 0:
			sh = Sigma_h.FCEL_integration_xF(x, p_T, sig_h, q0=q0)[0]
			sl = Sigma_l.FCEL_integration_xF(x, p_T, sig_l, q0=q0)[0]
		else:
			sh, sl = sig_h(x), sig_l(x)
		out.append(jpsi.alpha_from_R(sh, sl, A_h, A_l) if (sh > 0 and sl > 0) else np.nan)
	return np.array(out)

def R_model(model, xF_list, q0):
	"""R(x_F) = R_W/R_Be ; q0 = 0 gives the nPDF-only curve"""
	sig_h, sig_l = model
	out = []
	for x in xF_list:
		if q0 > 0:
			sh = Sigma_h.FCEL_integration_xF(x, p_T, sig_h, q0=q0, B=A_l)[0]
			sl = sig_l(x)
		else:
			sh, sl = sig_h(x), sig_l(x)
		out.append(sh/sl if (sh > 0 and sl > 0) else np.nan)
	return np.array(out)

def chi2(model, q0):
	if Obs == 'alpha':
		th = alpha_model(model, xF_data, q0)
	elif Obs == 'R':
		th = R_model(model,xF_data, q0)
	else:
		raise ValueError("Obs has to be equal to 'alpha' or 'R'")
	if not np.all(np.isfinite(th)):
		return 1e12
	return float(np.sum(((th-data)/err_data)**2))


def fit_q0(model):
	"""returns q0, dq0 (parabolic, Delta chi2 = 1), chi2_min"""
	c = np.array([chi2(model, q) for q in q0_scan])
	i = int(np.argmin(c))
	lo, hi = q0_scan[max(i-1, 0)], q0_scan[min(i+1, len(q0_scan)-1)]
	res = minimize_scalar(lambda q: chi2(model, q), bounds=(lo, hi), method='bounded', options={'xatol': 1e-4})
	q_best, c_best = (res.x, res.fun) if res.fun < c[i] else (q0_scan[i], c[i])
	if i in (0, len(q0_scan)-1):
		print("   WARNING: minimum at the edge of the scan range, enlarge q0_scan")
	h = 0.05*q_best
	curv = (chi2(model, q_best+h)-2*c_best+chi2(model, q_best-h))/h**2
	dq = np.sqrt(2./curv) if curv > 0 else np.nan
	return q_best, dq, c_best

### Loop on the members (results are cached like in RpA_Jpsi.py) ###
f_fit = os.path.join(fit_dir, set_label+'_'+col_type+'_'+Obs+'_q0fit_E866_Jpsi_'+str(rs)[:5]+'GeV_pT'+str(p_T)+'GeV_mu'+str(mu)+'_xF'+str(xF_fit_min)+'-'+str(xF_fit_max)+'.txt')
f_curves = f_fit.replace('_q0fit_', '_q0fit_curves_')
if os.path.exists(f_fit) and os.path.exists(f_curves):
	print(f"The file '{os.path.basename(f_fit)}' already exists. It is loaded.")
	res_fit = np.loadtxt(f_fit, ndmin=2)
	curves = np.loadtxt(f_curves, ndmin=2)
	npdf = curves[-1]; curves = curves[:-1]
else:
	res_fit, curves = [], []
	for mem in members:
		print(f"member {mem}/{members[-1]}")
		model = build_member(mem)
		q0, dq0, c = fit_q0(model)
		print(f"   q0 = {q0:.4f} +/- {dq0:.4f} GeV^2/fm | chi2/dof = {c:.2f}/{dof} = {c/dof:.3f}")
		res_fit.append([mem, q0, dq0, c, c/dof])
		if Obs=='alpha':
			curves.append(alpha_model(model, xF_plot, q0))
		elif Obs =='R':
			curves.append(R_model(model, xF_plot, q0))
		if mem == 0:
			if Obs=='alpha':
				npdf = alpha_model(model, xF_plot, 0.)
			elif Obs =='R':
				npdf = R_model(model, xF_plot, 0.)
			
			print(f"   nPDF only (q0=0): chi2/dof = {chi2(model, 0.)/dof:.3f}")
	res_fit, curves = np.array(res_fit), np.array(curves)
	np.savetxt(f_fit, res_fit, header=" member q0[GeV2/fm] dq0 chi2 chi2_red (dof=%d)" % dof)
	np.savetxt(f_curves, np.vstack([curves, npdf]))
	print(f" '{os.path.basename(f_fit)}' has been created")

### Parametrization (1-x')^n/x': same fit, no member dependence ###
if use_param:
	model_param = [Sigma_h.sigma_xF_param(p_T, n_param), Sigma_l.sigma_xF_param(p_T, n_param)]
	q0_par, dq0_par, c_par= fit_q0(model_param)
	print(f"\nParametrization (1-x')^n/x', n = {n_param}: q0 = {q0_par:.4f} +/- {dq0_par:.4f} GeV^2/fm | chi2/dof = {c_par:.2f}/{dof} = {c_par/dof:.3f}")
	if Obs == 'alpha':
		curve_param = alpha_model(model_param, xF_plot, q0_par)
	else:
		curve_param = R_model(model_param, xF_plot, q0_par)


q0_all = res_fit[:, 1]
print("\nmember  q0      chi2_red")
for r in res_fit:
	print(f"{int(r[0]):4d}  {r[1]:.4f}  {r[4]:.3f}")
if len(q0_all) > 2:
	q_c, q_p, q_m = hess_error([q0_all], cl=cl)
	print(f"\nq0 (central, hessian {cl}% CL) = {q_c[0]:.4f} +{q_p[0]:.4f} -{q_m[0]:.4f} GeV^2/fm")

### Plot 1: alpha(x_F), data and fitted curves ###
fig, ax = plt.subplots(constrained_layout=True)
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
for k in range(1, len(curves)):
	plt.plot(xF_plot, curves[k], color='tab:blue', alpha=alph, lw=0.8, label=set_label+' members (fit)' if k == 1 else None)
plt.plot(xF_plot, curves[0], color='crimson', lw=2, label=set_label+' central (fit), $q_0$ = %.3f' % q0_all[0])
plt.plot(xF_plot, npdf, color='k', ls='--', lw=1.2, label='no FCEL ($q_0=0$)' if is_pp else 'nPDF only')
if use_param:
	plt.plot(xF_plot, curve_param, color='tab:green', lw=2, ls='-.', label=r"$(1-x')^n/x'$, $n$ = %.1f, $q_0$ = %.3f" % (n_param, q0_par))
plt.errorbar(xF_all[~mask], data_all[~mask], yerr=err_all[~mask], fmt='o', mfc='none', color='grey', ms=4, label='E866 (not fitted)')
plt.errorbar(xF_data, data, yerr=err_data, fmt='o', color='k', ms=4, label='E866 (fitted)')
if Obs == 'alpha':
	plt.ylabel(r'$\alpha_{J/\psi}(x_F)$', fontsize=f_size-a)
elif Obs == 'R':
	plt.ylabel(r'$R_\text{W/Be}^{J/\psi}(x_F)$', fontsize=f_size-a)
plt.xlabel(r'$x_F$', fontsize=f_size)
plt.xlim(-0.2, 1)
plt.legend(frameon=False, fontsize=f_size-a-2, loc='lower left')
plt.tick_params(labelsize=f_size)
plt.text(0.97, 0.93, r'$\sqrt{s} =$'+'%.1f' % rs + ' GeV', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=f_size-a)
plt.savefig(os.path.join(plots_dir, set_label+'_E866_Jpsi_'+Obs+'_xF_fit_q0'+'_xF'+str(xF_fit_min)+'-'+str(xF_fit_max)+'.pdf'), bbox_inches="tight")

### Plot 2: q0 and chi2_red per member ###
fig2, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(6, 5))
ax1.errorbar(res_fit[:, 0], res_fit[:, 1], yerr=res_fit[:, 2], fmt='o', ms=3)
ax1.set_ylabel(r'$q_0$ [GeV$^2$/fm]', fontsize=f_size-a)
ax2.plot(res_fit[:, 0], res_fit[:, 4], 'o', ms=3)
ax2.axhline(1, color='grey', alpha=alph)
ax2.set_ylabel(r'$\chi^2/{\rm dof}$', fontsize=f_size-a)
ax2.set_xlabel('member', fontsize=f_size)
plt.savefig(os.path.join(plots_dir, set_label+'_E866_Jpsi_q0_chi2_members.pdf'), bbox_inches="tight")
plt.show()