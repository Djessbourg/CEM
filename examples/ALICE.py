# -*- coding: utf-8 -*-

# =============================================================================
# File for computing reweighting on ALICE datas
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
from scipy.special import erfinv

### DATA ###
x_type = "y"

# rs = 5020  # GeV
# x_value  = [-3.71,2.78] 
# bin_width = 1.5
# Data = [1.08,0.7]
# x_stat = [0.01,0.01]
# x_sys_unco = [0.09,0.05]
# x_unc = [(x_stat[i]**2+x_sys_unco[i]**2)**0.5 for i in range(len(x_stat))]


rs = 8160 # why they've writen 8016.0 GeV in the dataset ?
x_value = [-4.335,-4.085,-3.835,-3.585,-3.335,-3.085,2.155,2.405,2.655,2.905,3.155,3.405]
bin_width = 0.25
Data = [0.99,1.04,0.99,0.94,0.94,0.95,0.74,0.74,0.69,0.66,0.62,0.57]
x_stat = [0.02,0.01,0.01,0.01,0.01,0.02,0.01,0.01,0.01,0.01,0.01,0.01]
x_sys_unco = [0.09,0.07,0.06,0.06,0.06,0.06,0.05,0.04,0.04,0.04,0.04,0.04]
x_sys_global = [0.07,0.07,0.07,0.07,0.07,0.07,0.05,0.05,0.05,0.05,0.05,0.04,]
x_unc = [(x_stat[i]**2+x_sys_unco[i]**2+x_sys_global[i]**2)**0.5 for i in range(len(x_stat))]

# d_info = rw.Data_info_dict(rs,x_type,x_value,x_unc)

# Creation du reweighting object
s = rs**2 

# put EPPS21 here 
proton = "CT18ANLO" # baseline for EPPS21 
Pb = "EPPS21nlo_CT18Anlo_Pb208"
# proton = "nCTEQ15WZSIH_1_1"
# Pb = "nCTEQ15WZSIH_208_82"
n_max = 48

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
# mu = 2
mu = 1
p_T = 0
Obs = lambda y,i:LHC_Jpsi.RpA(y,p_T,i,mu_factor=mu,mu_f_factor=mu)
Obs.__name__ = 'RpA'
Obs2 = lambda y,i:LHC_Jpsi.vanilla_RpA(y,p_T,i,mu_factor=mu,mu_f_factor=mu)
Obs2.__name__ = 'vanilla_RpA'

member = False # Put true if you want to plot every member

def ConfLevel_to_sigma(b):
    "b en % (genre 90% => b = 90)"
    return erfinv(b/100.)*(2**0.5)

def hess_error(H,sym = False, cl=90): # H matrice [x_id][mem]
    factor = ConfLevel_to_sigma(cl)
    f_90_to_68 = ConfLevel_to_sigma(90)
    factor = factor/f_90_to_68
    Nx = len(H)
    Neig = (len(H[0])-1)//2
    cen = [H[i][0] for i in range(len(H))]
    if sym:
        central = np.mean(H,axis=1)
        delta = [sum([(H[i][2*k-1]-H[i][2*k])**2 for k in range(1,Neig+1)])/2 for i in range(Nx)]
        delta = np.array(delta)
        return [central,delta,delta]
    Oplus = [[max(cen[i],H[i][2*k-1],H[i][2*k])for i in range(Nx)]for k in range(1,Neig+1)]
    Ominus= [[min(cen[i],H[i][2*k-1],H[i][2*k])for i in range(Nx)]for k in range(1,Neig+1)]
    delta_plus = [np.sqrt(sum([(cen[i]-Oplus[k][i])**2 for k in range(Neig)]))for i in range(Nx)]
    delta_minus = [np.sqrt(sum([(cen[i]-Ominus[k][i])**2 for k in range(Neig)]))for i in range(Nx)]
    return [np.array(cen),factor*np.array(delta_plus),factor*np.array(delta_minus)]



Val = []
Val2 = []
Val_FCEL = []
X = np.linspace(-5,5,30)
sym= False
cl = 90

lha_cen,lha_plus,lha_minus=[],[],[]

for i,y in enumerate(X):
    print(f"{i}/{len(X)-1}")
    A = [Obs2(y,mem) for mem in range(0,n_max+1)]
    B = [Obs2(y,mem) for mem in range(n_max+1,len(A_list))]
    # Val.append(np.array([Obs2(y,mem) for mem in range(n_max+1)]))
    # Val_FCEL.append(np.array([Obs(y,mem) for mem in range(n_max+1)]))
    Val.append(A)
    # Val2.append(A+B)
    Val2.append(A)
    A_unc = A_set.uncertainty(A+B,alternative = True)
    lha_cen.append(A_unc.central);lha_plus.append(A_unc.errplus);lha_minus.append(A_unc.errminus)

    # Val_FCEL.append([Obs(y,mem) for mem in range(n_max+1)])

lha_cen,lha_plus,lha_minus = np.array(lha_cen),np.array(lha_plus),np.array(lha_minus)
# Val = np.array(Val)
# Val_FCEL = np.array(Val_FCEL)

if member:
    # EPPS21 alone
    for i in range(n_max + 1):
    # for i in range(len(A_list)):
        L  = [Obs2(y,i) for y in x_value]
        if i == 0:
            plt.plot(x_value,L,'.:b',alpha = 1, label = "EPPS21")
            # plt.plot(x_value,L,'.:b',alpha = 1, label = "nCTEQ15WZ+SIH")
        else:
            plt.plot(x_value,L,'.:b',alpha = 0.5)

    # EPPS21 + FCEL
    # for i in range(n_max + 1):
    # # for i in range(len(A_list)):
    #     L  = [Obs(y,i) for y in x_value]
    #     if i == 0:
    #         plt.plot(x_value,L,'.:g',alpha = 1, label = "EPPS21 + FCEL")
    #         # plt.plot(x_value,L,'.:g',alpha = 1, label = "nCTEQ15WZ+SIH + FCEL")
    #     else:
    #         plt.plot(x_value,L,'.:g',alpha = 0.5)
else:
    RpA,D_plus,D_minus = hess_error(Val,sym = False,cl= cl)
    RpA2,D_plus2,D_minus2 = hess_error(Val2,sym = True,cl= cl)
    # RpA_FCEL,D_FCEL_plus,D_FCEL_minus = hess_error(Val_FCEL)
    # plt.plot(X,RpA,color='red')
    # plt.fill_between(X,RpA-D_minus,RpA+D_plus,color='red',alpha = 0.3,label = 'assym EPPS21')
    # plt.plot(X,RpA2,color='blue')
    # plt.fill_between(X,RpA2-D_minus2,RpA2+D_plus2,color='blue',alpha = 0.3,label = 'sym EPPS21')
    # plt.plot(X,RpA,color='blue')
    # plt.fill_between(X,RpA-D_plus,RpA+D_plus,color='blue',alpha = 0.3,label = 'sym 1 EPPS21')
    DT = (D_minus+D_plus)/2
    plt.fill_between(X,RpA-DT,RpA+DT,color='green',alpha = 0.3,label = 'sym 2 EPPS21')
    DT2 = np.sqrt(D_minus**2+D_plus**2)/2
    plt.fill_between(X,RpA-DT2,RpA+DT2,color='orange',alpha = 0.3,label = 'sym 3 EPPS21')
    # plt.plot(X,lha_cen,color='green')
    # plt.fill_between(X,lha_cen-lha_minus,lha_cen+lha_plus,color='green',alpha = 0.3,label = 'lhapdf EPPS21')
    # plt.plot(X,RpA_FCEL,color='blue')
    # plt.fill_between(X,RpA_FCEL-D_FCEL_minus,RpA_FCEL+D_FCEL_plus,color='blue',alpha=0.3,label='EPPS21 + FCEL')



#Data plot
plt.axhline(y=1, color='grey',alpha=0.3)
plt.errorbar(x_value,Data,xerr=bin_width/2,yerr=x_unc,fmt='+',color='red',label='ALICE 8.16 TeV')
plt.ylim(0,1.6)
plt.xlim()
plt.legend()
plt.xlabel(x_type)
plt.ylabel(Obs.__name__+r' $\sqrt{s} = $'+ str(rs/1000)+' TeV')
plt.savefig(os.path.join(plots_dir,str(mu)+'mu_and_pt0_ALICE_EPPS21_FCEL_rs_'+str(rs)+'GeV.pdf'))
# plt.savefig(os.path.join(plots_dir,'ALICE_nCTEQ15WZSIH_FCEL_rs_'+str(rs)+'GeV.pdf'))
plt.show()