# -*- coding: utf-8 -*-

# =============================================================================
# Study the first impacts of compton QCD diagrams and annihilation 
# to parton luminosity with the sigma class.
# Study on RpA and isospin effects.
# last modified: 25/02/2026
# =============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
from src import Jpsi as jpsi
from src import Collision, Var_choice

rs = 7000
# rs = 200
# rs =5000
# rs =38.7
# rs =15
s = (rs)**2 # CM energy in Gev2
proton = "NNPDF40_nlo_as_01180"
# proton= "NNPDF40_lo_as_01180"
# proton = "CT10"
# proton = "cteq61" # This one, with B and a new F gives a good approximation of phenix data at 200 GeV
# proton = "MRSTMCal"
Pb = "nNNPDF30_nlo_as_0118_A208_Z82"
coll_type = 'pp'
Var = 'y'
# Var = 'xf'
cen = 0 
# f_mu = 2
f_mu=1
# B = 5.97/100 #Branching ratio of J/psi -> mu+mu-
B = 1 
F = 0.025 # CEM constant in R.Vogt paper for proton induced interactions
# F = 0.034 # CEM constant in R.Vogt paper for pion induced interations
# F = 0.0565# From my pov, i found F \in [0.050;0.063] so i did the mean

# F=1

p_T = 0 
# p_T = 1
Atom = jpsi.Atom

atom = 'Pb'

Z = Atom[atom]['Z']
A = Atom[atom]['A']

LHC_Jpsi = jpsi.SigmaJpsi(proton,Pb,s,Z,A)

if Var_choice(Var):
    L = jpsi.Y_list(rs)
else:
    L = jpsi.X_F_list(rs)

# print('for xf:')
# print(F*B*LHC_Jpsi.sigma(cen,mu_factor=f_mu,mu_f_factor=f_mu,is_pp=Collision(coll_type),Var_y=False)[0])
# print('for y:')
# print(F*B*LHC_Jpsi.sigma(cen,mu_factor=f_mu,mu_f_factor=f_mu,is_pp=Collision(coll_type),Var_y=True)[0])


# sig_gg, err_gg = LHC_Jpsi.dsigma_gg_Y_or_x_F(p_T,cen,mu_factor=f_mu,mu_f_factor=f_mu,is_pp=Collision(coll_type),Var_y=Var_choice(Var))
# sig_qqbar, err_qqbar = LHC_Jpsi.dsigma_qqbar_Y_or_x_f(p_T,cen,mu_factor=f_mu,mu_f_factor=f_mu,is_pp=Collision(coll_type),Var_y=Var_choice(Var))
sig_tot, err_tot = LHC_Jpsi.dsigma_tot_Y_or_x_f(p_T,cen,mu_factor=f_mu,mu_f_factor=f_mu,is_pp=Collision(coll_type),Var_y=Var_choice(Var))

# n=8.3
n=19.2
n=32.3
sig_test = lambda y: LHC_Jpsi.sigma_Jpsi(y,p_T,n=n)
sig_test_plus = lambda y: LHC_Jpsi.sigma_Jpsi(y,p_T,n=n+7.5)
sig_test_minus = lambda y: LHC_Jpsi.sigma_Jpsi(y,p_T,n=n-7.5)

norm = LHC_Jpsi.dsigma_tot_dvar(0,p_T,cen,mu_factor=f_mu,mu_f_factor=f_mu,is_pp=Collision(coll_type),Var_y=Var_choice(Var))[0]/sig_test(0)
norm_plus = LHC_Jpsi.dsigma_tot_dvar(0,p_T,cen,mu_factor=f_mu,mu_f_factor=f_mu,is_pp=Collision(coll_type),Var_y=Var_choice(Var))[0]/sig_test_plus(0)
norm_minus=LHC_Jpsi.dsigma_tot_dvar(0,p_T,cen,mu_factor=f_mu,mu_f_factor=f_mu,is_pp=Collision(coll_type),Var_y=Var_choice(Var))[0]/sig_test_minus(0)
# sig_test_FCEL, err_sig_test_FCEL = LHC_Jpsi.FCEL_integration_dy_test(cen)


fig, ax = plt.subplots(constrained_layout=True)
# ax.xaxis.set_minor_locator(MultipleLocator(0.125))
# ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.set_major_locator(MultipleLocator(2))
# plt.plot(L,B*F*sig_gg/1000, label ='gg')
# plt.plot(L,B*F*sig_qqbar/1000, label= 'qqbar')
plt.plot(L,B*F*sig_tot/1000,label='tot')
plt.plot(L,norm*B*F*sig_test(L)/1000,color='blue',label='param')
plt.fill_between(L,norm_minus*B*F*sig_test_minus(L)/1000,norm_plus*B*F*sig_test_plus(L)/1000,alpha = 0.3,color='blue')
# plt.plot(L,sig_test_FCEL,label='FCEL')
plt.legend()
plt.grid()
# plt.xlim(0,1)
# plt.ylim(10**(-2),10**4)
# plt.yscale('log')
plt.xlabel(Var)
plt.ylabel('cross section in nb')
plt.text(0.01,0.9,proton, horizontalalignment='left', verticalalignment='center',transform=ax.transAxes)
plt.text(0.01,0.85,r'$n = $ ' +str(n) +r' $\pm 5$',horizontalalignment='left', verticalalignment='center',transform=ax.transAxes)
plt.text(0.01,0.80,r'$\sqrt{s} = $ '+str(rs/1000)+' TeV',horizontalalignment='left', verticalalignment='center',transform=ax.transAxes)
plt.text(0.01,0.75,r'$p_\bot = $ '+str(p_T)+' GeV',horizontalalignment='left', verticalalignment='center',transform=ax.transAxes)
# plt.ylabel('cross section no unitiy')
plt.show()
