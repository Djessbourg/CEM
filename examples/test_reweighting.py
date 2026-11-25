# -*- coding: utf-8 -*-

# =============================================================================
# File for testing the reweighting module
# Last modified: 25/08/2026
# =============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
omega_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'omega'))

lha_new_pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'myPDFs'))
import lhapdf as lha
lha.pathsAppend(lha_new_pdf_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import Jpsi as jpsi
from src import reweighting as rw

def str_pcen(pcen):
    if pcen:
        return 'pcen'
    else:
        return 'nopcen'

def str_FCEL(FCEL):
    if FCEL:
        return 'FCEL'
    else:
        return 'noFCEL'


# Creation du reweighting object

# NNPDF30 --------------------------------------

# proton = "nNNPDF30_nlo_as_0118_noLHCbD_p"
# Pb = "nNNPDF30_nlo_as_0118_noLHCbD_A208_Z82"

proton = "nNNPDF30_nlo_as_0118_noLHCbD_1000_p"
Pb = "nNNPDF30_nlo_as_0118_noLHCbD_1000_A208_Z82"

p_set = lha.getPDFSet(proton)
A_set = lha.getPDFSet(Pb)

p_list = p_set.mkPDFs()
A_list = A_set.mkPDFs()

Nmem = len(A_list)-1
Nrep = Nmem
local = False

# EPPS21 ---------------------------------------

# proton = "CT18ANLO" # The true proton baseline for EPPS21, see arXiv:2112.12462
# Pb = "EPPS21nlo_CT18Anlo_Pb208" # Be carefull to use only the first 48 eigenvalues, the rest is pPb correlations
# local = False
# Pb = "bayesianised_EPPS21nlo_CT18Anlo_Pb208_250"
# Pb = "bayesianised_EPPS21nlo_CT18Anlo_Pb208_1000"
# local = True

original = False
# original = True

# Pb = "original_bayesianised_EPPS21nlo_CT18Anlo_Pb208_1000"

# p_set = lha.getPDFSet(proton)
# A_set = lha.getPDFSet(Pb)

# p_list = p_set.mkPDFs()
# A_list = A_set.mkPDFs()

# Nmem = 48
# Nrep = 1000
# Nmem = len(A_list)-1
# Nrep = Nmem
# ----------------------------------------------

atom = 'Pb'
# atom = 'Au'

Atom = jpsi.Atom #dictionary where we stock important atomic aspect (now just Z and A) to plot and put into file names

Z = Atom[atom]['Z']
A = Atom[atom]['A']

# FCEL = False
FCEL = True
proton_cen = True # define if we use, for the proton, the central member or the same as the Pb one 
# proton_cen = False # ONLY FOR nNNPDF30
###### Valeurs fictives ######

name =  str_FCEL(FCEL)+"_"+str_pcen(proton_cen)+"_"+"ALICE_5_8_and_LHCb_8"+"_"+Pb

# creation du dictionaire d'info sur les data
# rs = 8000 # GeV
# x_type = "y"
# x_value  = [1,2,3,4,5]
# Data = [0.75,0.73,0.7,0.65,0.6]
# # x_unc = [0.2,0.2,0.2,0.2,0.2]
# x_unc = [0.05,0.05,0.05,0.05,0.05]
# # x_unc = [0.01,0.01,0.01,0.01,0.01]

### Vraies valeurs (Alice 8.16 TeV) ###
Data_dict = {}
Data_info_dict= {}
Obs_dict={}
LHC_Jpsi = {}

mu = 2
# mu = 1
p_T = 1

#  LHCb 8.16 TeV ----------------------------------------------
A_name = "LHCb_8160"
rs = 8160 # GeV
x_type = "y"
x_value  = [-4.25,-3.75,-3.25,-2.75,1.75,2.25,2.75,3.25,3.75] # central values, bin width of 0.5
bin_width = 0.5
A_Data = [0.86,0.84,0.87,0.90,0.68,0.71,0.62,0.59,0.57]
x_unc = [0.10,0.09,0.10,0.13,0.09,0.07,0.06,0.05,0.05]

LHC_Jpsi_A = jpsi.SigmaJpsi(p_set,p_list,A_set,A_list,rs**2,Z,A)
LHC_Jpsi[A_name] = LHC_Jpsi_A
A_info = rw.Data_info_dict(rs,x_type,x_value,x_unc)
Data_dict[A_name] = A_Data
Data_info_dict[A_name] = A_info

# ALICE 8.16 TeV ----------------------------------------------
B_name = "ALICE_8160"
rs = 8160 # GeV
x_value  = [-4.335,-4.085,-3.835,-3.585,-3.335,-3.085,2.155,2.405,2.655,2.905,3.155,3.405]
x_type = "y"
bin_width = 1.5
B_Data = [0.99,1.04,0.99,0.94,0.94,0.95,0.74,0.74,0.69,0.66,0.62,0.57]
x_stat = [0.02,0.01,0.01,0.01,0.01,0.02,0.01,0.01,0.01,0.01,0.01,0.01]
x_sys_unco = [0.09,0.07,0.06,0.06,0.06,0.06,0.05,0.04,0.04,0.04,0.04,0.04]
x_unc = [(x_stat[i]**2+x_sys_unco[i]**2)**0.5 for i in range(len(x_stat))]

LHC_Jpsi_B = jpsi.SigmaJpsi(p_set,p_list,A_set,A_list,rs**2,Z,A)
LHC_Jpsi[B_name] = LHC_Jpsi_B
B_info = rw.Data_info_dict(rs,x_type,x_value,x_unc)
Data_dict[B_name] = B_Data
Data_info_dict[B_name] = B_info

# LHCb 5 TeV ----------------------------------------------
# C_name = "LHCb_5000"
# rs = 5000 # GeV
# x_value  = [-3.25,3.25] 
# x_type = "y"
# bin_width = 1.5
# C_Data = [0.93,0.62]
# x_stat = [0.03,0.01]
# x_sys = [0.05,0.02]
# x_interpol= [0.05,0.03] # not taken in count bc i'm not sure
# x_unc = [(x_stat[i]**2+x_sys[i]**2)**0.5 for i in range(len(x_stat))]

# LHC_Jpsi_C = jpsi.SigmaJpsi(p_set,p_list,A_set,A_list,rs**2,Z,A)
# LHC_Jpsi[C_name] = LHC_Jpsi_C
# C_info = rw.Data_info_dict(rs,x_type,x_value,x_unc)
# Data_dict[C_name] = C_Data
# Data_info_dict[C_name] = C_info

# ALICE 5.02 TeV----------------------------------------------
D_name = "ALICE_5020"
rs = 5020  # GeV
x_value  = [-3.71,2.78] 
x_type = "y"
bin_width = 1.5
D_Data = [1.08,0.7]
x_stat = [0.01,0.01]
x_sys_unco = [0.09,0.05]
x_unc = [(x_stat[i]**2+x_sys_unco[i]**2)**0.5 for i in range(len(x_stat))]

LHC_Jpsi_D = jpsi.SigmaJpsi(p_set,p_list,A_set,A_list,rs**2,Z,A)
LHC_Jpsi[D_name] = LHC_Jpsi_D
D_info = rw.Data_info_dict(rs,x_type,x_value,x_unc)
Data_dict[D_name] = D_Data
Data_info_dict[D_name] = D_info

# ----------------------------------------------

for k in LHC_Jpsi.keys():
    if FCEL:
        Obs_dict[k] = lambda y, i, k=k: LHC_Jpsi[k].RpA(y, p_T, i, mu_factor=mu, mu_f_factor=mu, p_cen=proton_cen)
    else:
        Obs_dict[k] = lambda y, i, k=k: LHC_Jpsi[k].vanilla_RpA(y, p_T, i, mu_factor=mu, mu_f_factor=mu, p_cen=proton_cen)

### Reweight Object for the first creation ###

Reweigth_test = rw.Reweight(A_set,A_list,Data = Data_dict,Data_info=Data_info_dict,Obs = Obs_dict,Nrep= Nrep,FCEL=FCEL,Nmem_max=Nmem,local= local,original = original,name_adons=str_pcen(proton_cen))

# Test plot du Obs

Reweigth_test.plot_obs("RpA",member = True)

# Test plot 3D

Reweigth_test.plot_obs_3D('RpA')

# Test distribution en chi2

Reweigth_test.plot_chi2_distribution(Nbins=50,Chi_max=20)

# Enregistrement du ficher des poids omegas

Reweigth_test.save_data(Obs_name='RpA',name= name)

### Weight Object for the computation after creation ###

# d = rw.open_data(Obs_name='RpA',name=name)

# Weight_test = rw.Weights(A_set,A_list,data_d=d)

# omegas = Weight_test.Omega_list
# chi2 = Weight_test.Chi2_list

# print(50*"=")
# print('sum omegas = '+str(sum(omegas))) # normalement ça fait neff 

# plot the Chi2 distribution

# Weight_test.plot_chi2_distribution(Nbins=50,Chi_max=20,name_plot=name)

# plot 3D
# Weight_test.plot_obs_3D(Obs.__name__,Obs)
