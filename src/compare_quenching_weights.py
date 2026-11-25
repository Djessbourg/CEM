# -*- coding: utf-8 -*-

# =============================================================================
# Compare the quenching weights P_{W|p} and P_{W|Be} (Arleo-Peigne, JHEP03(2013)122, Eq. 3.23),
# as a function of x = eps/E, at fixed x_F (E866 kinematics, FCEL only).
# Also tests the composition P_{W|p} =? P_{W|Be} (x) P_{Be|p} (independent emissions, Monte Carlo).
# The cumulative of P is exactly F(x) = exp(-int_x^inf dI) = prob.Exp_u, which is used to sample eps.
# Needs Probability.py in the same folder (or in the python path).
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import Probability as prob

### Settings (same as Fit_q0_E866_1.py) ###
m_p, p_beam = 0.938, 800. # GeV
rs = np.sqrt(2*m_p*p_beam+2*m_p**2) # 38.8 GeV
p_T, m_jpsi = 1., 3.0 # GeV
M_T = np.hypot(m_jpsi, p_T)
alpha_s, Nc, Xi = 0.5, 3., 0.5
q0 = 0.075 # GeV^2/fm
A_h, A_l = 184, 9 # W, Be
xF_list = [0.3, 0.5, 0.7]
n_mc = 400000
rng = np.random.default_rng(1)
trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

### Weights in the variable x = eps/E ###
def sh_chi(A, B, xF):
	"""sigma_hat = l_A/M_T and chi = (Lambda_B/l_A)^2 of the spectrum of A relative to B"""
	y0 = np.arcsinh(xF*rs/(2*M_T))
	pr = prob.proba(A, B, rs, p_T, y0, alpha_s, Nc, m_jpsi, q0, photon=False)
	return pr.sigma_hat(Xi), pr.Chi(Xi)

def P_x(x, sh, chi):
	"""density of x = eps/E: P(x) dx = p_tilde(u) du, u = x/sigma_hat"""
	return prob.p_tilde_u(x/sh, chi, Nc, alpha=alpha_s)/sh

def F_x(x, sh, chi):
	"""cumulative of P: probability that eps/E < x"""
	return prob.Exp_u(x/sh, chi, Nc, alpha=alpha_s)

def sample(sh, chi, n):
	"""inverse-transform sampling of x = eps/E"""
	lx = np.linspace(-16, 3, 6000)
	return np.exp(np.interp(rng.random(n), F_x(np.exp(lx), sh, chi), lx))

### Figure + numbers ###
lx = np.linspace(np.log(1e-12), np.log(10.), 6000)
x = np.exp(lx)
bins = np.logspace(-6, 0.3, 60)
xc = np.sqrt(bins[1:]*bins[:-1])
fig, axs = plt.subplots(1, len(xF_list), sharey=True, figsize=(12, 3.8), constrained_layout=True)
print("x = eps/E ; <x>_{x<1} = int_{x<1} x P dx ; P(x>1) = 1-F(1) is the part cut by the kinematic limit")
for ax, xF in zip(axs, xF_list):
	sh_p, chi_p = sh_chi(A_h, 1., xF) # W | p
	sh_b, chi_b = sh_chi(A_h, A_l, xF) # W | Be
	sh_l, chi_l = sh_chi(A_l, 1., xF) # Be | p
	comp = sample(sh_b, chi_b, n_mc) + sample(sh_l, chi_l, n_mc) # P_{W|Be} (x) P_{Be|p}
	print(f"\nx_F = {xF}: chi(W|p) = {chi_p:.3f}, chi(W|Be) = {chi_b:.3f}, sigma_hat(W) = {sh_p:.3f}")
	for name, sh, chi in (("W|p", sh_p, chi_p), ("W|Be", sh_b, chi_b),("Be|p", sh_l, chi_l)):
		dens = x*P_x(x, sh, chi) # density in ln x
		F = F_x(x, sh, chi)
		m = x < 1
		print(f"   {name:5s} median x = {np.interp(0.5, F, x):.4f} | <x>_(x<1) = {trap(x[m]*dens[m], lx[m]):.4f} | P(x>1) = {1-np.interp(1., x, F):.4f}")
		ax.plot(x, dens, lw=2, label=r'$P_{%s}$' % name.replace('|', r'|'))
	# print(f"   W|Be (x) Be|p (MC) median x = {np.median(comp):.4f} | <x>_(x<1) = {np.mean(comp*(comp < 1)):.4f} | P(x>1) = {np.mean(comp > 1):.4f}")
	# h = np.histogram(comp, bins=bins)[0]/(len(comp)*np.diff(np.log(bins)))
	# ax.plot(xc, h, 'k--', lw=1.2, label=r'$P_{W|Be}\otimes P_{Be|p}$ (MC)')
	ax.set_xscale('log')
	ax.set_xlim(1e-6, 2)
	ax.set_xlabel(r'$x=\varepsilon/E$')
	ax.set_title(r'$x_F=%.1f$,  $q_0=%.3f$ GeV$^2$/fm' % (xF, q0), fontsize=10)
axs[0].set_ylabel(r'$x\,P(x)$  (density in $\ln x$)')
axs[0].legend(frameon=False, fontsize=9, loc='upper left')
plt.savefig('quenching_weights_WBe_vs_Wp.pdf', bbox_inches='tight')
plt.show()
