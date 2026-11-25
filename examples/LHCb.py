# -*- coding: utf-8 -*-

# =============================================================================
# File for computing reweighting on LHCb datas
# Last modified: 26/06/2026
# =============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
omega_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'omega'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lhapdf as lha
from src import Jpsi as jpsi
from src import reweighting as rw

### DATA ###

rs = 8160 # GeV
x_type = "y"
x_value  = [-4.25,-3.75,-3.25,-2.75,1.75,2.25,2.75,3.25,3.75] # central values, bin width of 0.5
bin_width = 0.5
Data = [0.86,0.84,0.87,0.90,0.68,0.71,0.62,0.59,0.57]
x_unc = [0.10,0.09,0.10,0.13,0.09,0.07,0.06,0.05,0.05]

d_info = rw.Data_info_dict(rs,x_type,x_value,x_unc)

# Creation du reweighting object
s = rs**2 

# put EPPS21 here 
proton = "CT18ANLO" # baseline for EPPS21 
Pb = "EPPS21nlo_CT18Anlo_Pb208"
n_max = 48 # members for nuclear uncertainties

p_set = lha.getPDFSet(proton)
A_set = lha.getPDFSet(Pb)

p_list = p_set.mkPDFs()
A_list = A_set.mkPDFs()

atom = 'Pb'
# atom = 'Au'

Atom = jpsi.Atom #dictionary where we stock important atomic aspect (now just Z and A) to plot and put into file names

Z = Atom[atom]['Z']
A = Atom[atom]['A']

LHC_Jpsi = jpsi.SigmaJpsi(p_set,p_list,A_set,A_list,s,Z,A)
mu = 2
p_T = 1
Obs = lambda y,i:LHC_Jpsi.RpA(y,p_T,i,mu_factor=mu,mu_f_factor=mu)
Obs.__name__ = 'RpA'
Obs2 = lambda y,i:LHC_Jpsi.vanilla_RpA(y,p_T,i,mu_factor=mu,mu_f_factor=mu)
Obs2.__name__ = 'vanilla_RpA'


# EPPS21 alone

for i in range(n_max+1):
    L  = [Obs2(y,i) for y in x_value]
    if i == 0:
        plt.plot(x_value,L,'.:b',alpha = 1, label = "EPPS21")
    else:
        plt.plot(x_value,L,'.:b',alpha = 0.5)

# L  = [Obs2(y,0) for y in x_value]

# plt.plot(x_value,L,marker='o',color='blue',alpha = 1, label = "EPPS21")

# EPPS21 + FCEL

for i in range(n_max+1):
    L  = [Obs(y,i) for y in x_value]
    if i == 0:
        plt.plot(x_value,L,'.:g',alpha = 1, label = "EPPS21 + FCEL")
    else:
        plt.plot(x_value,L,'.:g',alpha = 0.5)

# L  = [Obs(y,0) for y in x_value]    
# plt.plot(x_value,L,marker='o',color='green',alpha = 1, label = "EPPS21 + FCEL")

#Data plot
plt.errorbar(x_value,Data,xerr=bin_width/2,yerr=x_unc,fmt='+',color='red',label='Data')

plt.legend()
plt.xlabel(x_type)
plt.ylabel(Obs.__name__+r' $\sqrt{s} = $'+ str(rs/1000)+' TeV')
plt.savefig(os.path.join(plots_dir,'LHCb_EPPS21_FCEL_rs_'+str(rs)+'GeV.pdf'))
plt.show()