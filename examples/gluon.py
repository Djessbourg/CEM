# -*- coding: utf-8 -*-

# =============================================================================
# File for testing the make_lhapdf_set module
# Last modified: 4/08/2026
# =============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
omega_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'omega'))
lha_pdf_path = '/opt/homebrew/Cellar/lhapdf/6.5.4/share/LHAPDF' # maybe find an 'os-friendly' way here

lha_new_pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'myPDFs'))
import lhapdf as lha
lha.pathsAppend(lha_new_pdf_path)
print(lha.paths())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# with D mesons 
# proton_D = "nNNPDF30_nlo_as_0118_p"
# Pb_D = "nNNPDF30_nlo_as_0118_A208_Z82"
# proton = "NNPDF30_nlo_as_0118"
# proton = "NNPDF40_nlo_as_01180"
# proton = "nNNPDF30_nlo_as_0118_p"

# without D mesons
# proton = "nNNPDF30_nlo_as_0118_noLHCbD_p"
# proton = "nNNPDF30_nlo_as_0118_noLHCbD_1000_p"
# proton = "nNNPDF20_nlo_as_0118_p_A2_Z1"
# Pb = "nNNPDF30_nlo_as_0118_noLHCbD_A208_Z82"
# Pb = "nNNPDF30_nlo_as_0118_noLHCbD_1000_A208_Z82"
# Pb_new_FCEL = "unweighted_38rep_FCELnNNPDF30_nlo_as_0118_noLHCbD_1000_A208_Z82" #proton[k]
# Pb_new_FCEL = "unweighted_46rep_FCELnNNPDF30_nlo_as_0118_noLHCbD_1000_A208_Z82" #proton[0]
# Pb_new_noFCEL = "unweighted_39rep_noFCELnNNPDF30_nlo_as_0118_noLHCbD_A208_Z82" 
# Pb_new_noFCEL = "unweighted_42rep_noFCELnNNPDF30_nlo_as_0118_noLHCbD_1000_A208_Z82" # proton[k]
# Pb_new_noFCEL = "unweighted_94rep_noFCELnNNPDF30_nlo_as_0118_noLHCbD_1000_A208_Z82" # proton[0]

proton = "CT18ANLO"
Pb = "EPPS21nlo_CT18Anlo_Pb208" 
# original = True
original = False

if original: # Have to compute them first bc there was a pb in the test_reweighting.py file
    Pb_new_FCEL = "unweighted_131rep_FCELoriginal_bayesianised_EPPS21nlo_CT18Anlo_Pb208_1000"
    Pb_new_noFCEL = "unweighted_248rep_noFCELoriginal_bayesianised_EPPS21nlo_CT18Anlo_Pb208_1000"
else:
    Pb_new_FCEL = "unweighted_103rep_FCELbayesianised_EPPS21nlo_CT18Anlo_Pb208_1000"
    Pb_new_noFCEL = "unweighted_286rep_noFCELbayesianised_EPPS21nlo_CT18Anlo_Pb208_1000"


# p_D_set = lha.getPDFSet(proton_D)
# Pb_D_set = lha.getPDFSet(Pb_D)
p_set = lha.getPDFSet(proton)
Pb_set = lha.getPDFSet(Pb)
Pb_new_FCEL_set = lha.getPDFSet(Pb_new_FCEL)
Pb_new_noFCEL_set = lha.getPDFSet(Pb_new_noFCEL)

# p_D_list = p_D_set.mkPDFs()
# Pb_D_list = Pb_D_set.mkPDFs()
p_list = p_set.mkPDFs()
Pb_list = Pb_set.mkPDFs()
Pb_new_FCEL_list = Pb_new_FCEL_set.mkPDFs()
Pb_new_noFCEL_list = Pb_new_noFCEL_set.mkPDFs()

p_cen = True

# x0 = 1e-6
# Q2 = 10
# R_Pb_x0 = np.array([Pb_list[k].xfxQ2(21, x0, Q2) / p_list[0].xfxQ2(21, x0, Q2) for k in range(1, len(Pb_list))])
# print("max :", R_Pb_x0.max())
# print("nombre de répliques > 1.1 :", (R_Pb_x0 > 1.1).sum(), "/", len(R_Pb_x0))
# plt.hist(R_Pb_x0, bins=50)
# plt.show()

### Rg vectors ###
Q2 = 10 # GeV2
X = np.logspace(-5,-0.15,100)
Old_cen, Old_plus, Old_minus = [],[],[]
Old_D_cen, Old_D_plus, Old_D_minus = [],[],[]
New_FCEL_cen, New_FCEL_plus, New_FCEL_minus = [],[],[]
New_noFCEL_cen, New_noFCEL_plus, New_noFCEL_minus = [],[],[]

### Test of the PDFs ###
# Y_p = np.array([p_list[0].xfxQ2(21,x,Q2)for x in X])
# Y_pb_cen,Y_pb_plus,Y_pb_minus=[],[],[]
# R_pb_cen,R_pb_plus,R_pb_minus=[],[],[]
# R_pb2_cen,R_pb2_plus,R_pb2_minus=[],[],[]

# for x in X:
#     Y_Pb = np.array([Pb_list[k].xfxQ2(21,x,Q2)/p_list[k].xfxQ2(21,x,Q2) for k in range(len(Pb_list))])
#     R_Pb = np.copy(Y_Pb)
#     Y_Pb_unc = Pb_set.uncertainty(Y_Pb,alternative=False)
#     Y_pb_cen.append(Y_Pb_unc.central);Y_pb_plus.append(Y_Pb_unc.errplus);Y_pb_minus.append(Y_Pb_unc.errminus)
#     # Y_pb_cen.append(np.median(Y_Pb));Y_pb_plus.append(np.nanpercentile(Y_Pb, 84));Y_pb_minus.append(np.nanpercentile(Y_Pb, 16))
#     R_pb_cen.append(np.median(R_Pb));R_pb_plus.append(np.nanpercentile(R_Pb, 84));R_pb_minus.append(np.nanpercentile(R_Pb, 16))
#     R_pb2_cen.append(np.mean(R_Pb));R_pb2_plus.append(np.std(R_Pb));R_pb2_minus.append(np.std(R_Pb))

# Y_pb_cen,Y_pb_plus,Y_pb_minus = np.array(Y_pb_cen),np.array(Y_pb_plus),np.array(Y_pb_minus)
# R_pb_cen,R_pb_plus,R_pb_minus = np.array(R_pb_cen),np.array(R_pb_plus),np.array(R_pb_minus)
# R_pb2_cen,R_pb2_plus,R_pb2_minus= np.array(R_pb2_cen),np.array(R_pb2_plus),np.array(R_pb2_minus)
# # #absolute comparison
# plt.plot(X,Y_p,color='red',label='proton')
# plt.plot(X,Y_pb_cen,color='blue',label = 'Pb')
# plt.fill_between(X,Y_pb_minus,Y_pb_plus,color='blue',alpha=0.3)
# plt.legend()
# # plt.yscale('log')
# plt.xscale('log')
# plt.xlabel("x")
# plt.ylabel(r"gluon nNNPDF3.0 noLHCBD (nlo)")
# plt.xlim(10**-5,0.7)
# plt.savefig(os.path.join(plots_dir, "Test_gluon_pPb_absolute.pdf"))
# plt.show()
# plt.close()

# #nuclear ratio
# plt.plot(X,Y_pb_cen,color='blue',label = 'Pb/p')
# plt.fill_between(X,(Y_pb_cen-Y_pb_minus),(Y_pb_cen+Y_pb_plus),color='blue',alpha=0.3)
# plt.plot(X,R_pb_cen,color='red',label = 'Rg median')
# plt.fill_between(X,(R_pb_minus),(R_pb_plus),color='red',alpha=0.3)
# plt.plot(X,R_pb2_cen,color='green',label = 'Rg std')
# plt.fill_between(X,(R_pb2_cen-R_pb2_minus),(R_pb2_cen+R_pb2_plus),color='green',alpha=0.3)
# plt.legend()
# # plt.yscale('log')
# plt.xscale('log')
# plt.xlabel("x")
# plt.ylabel(r"Rg nNNPDF3.0 noLHCBD (nlo)")
# plt.xlim(10**-6,0.7)
# plt.ylim(0,1.6)
# plt.savefig(os.path.join(plots_dir, "Test_gluon_pPb_ratio.pdf"))
# plt.show()
# plt.close()


### Rg test ###

Alt = True
for x in X:
    # old_D = [Pb_D_list[k].xfxQ2(21,x,Q2)/p_D_list[0].xfxQ2(21,x,Q2) for k in range(len(Pb_D_list))] if p_cen else [Pb_D_list[k].xfxQ2(21,x,Q2)/p_D_list[k].xfxQ2(21,x,Q2) for k in range(len(Pb_D_list))] # 3.0 with LHCb D messons
    old = [Pb_list[k].xfxQ2(21,x,Q2)/p_list[0].xfxQ2(21,x,Q2) for k in range(len(Pb_list))] if p_cen else [Pb_list[k].xfxQ2(21,x,Q2)/p_list[k].xfxQ2(21,x,Q2) for k in range(len(Pb_list))] # 3.0 without LHCb D mesons 
    new_FCEL = [Pb_new_FCEL_list[k].xfxQ2(21,x,Q2)/p_list[0].xfxQ2(21,x,Q2) for k in range(len(Pb_new_FCEL_list))] if p_cen else [Pb_new_FCEL_list[k].xfxQ2(21,x,Q2)/p_list[k].xfxQ2(21,x,Q2) for k in range(len(Pb_new_FCEL_list))] # 3.0 w/o + Jpsi + FCEL
    new_noFCEL = [Pb_new_noFCEL_list[k].xfxQ2(21,x,Q2)/p_list[0].xfxQ2(21,x,Q2) for k in range(len(Pb_new_noFCEL_list))] if p_cen else [Pb_new_noFCEL_list[k].xfxQ2(21,x,Q2)/p_list[k].xfxQ2(21,x,Q2) for k in range(len(Pb_new_noFCEL_list))] #3.0 w/o + Jpsi 
    # old_D_unc = Pb_D_set.uncertainty(old_D,alternative=Alt)
    old_unc = Pb_set.uncertainty(old,alternative=Alt)
    new_FCEL_unc = Pb_new_FCEL_set.uncertainty(new_FCEL,alternative=Alt)
    new_noFCEL_unc = Pb_new_noFCEL_set.uncertainty(new_noFCEL,alternative=Alt)
    # Old_D_cen.append(old_D_unc.central);Old_D_plus.append(old_D_unc.errplus);Old_D_minus.append(old_D_unc.errminus)
    Old_cen.append(old_unc.central);Old_plus.append(old_unc.errplus);Old_minus.append(old_unc.errminus)
    New_FCEL_cen.append(new_FCEL_unc.central);New_FCEL_plus.append(new_FCEL_unc.errplus);New_FCEL_minus.append(new_FCEL_unc.errminus)
    New_noFCEL_cen.append(new_noFCEL_unc.central);New_noFCEL_plus.append(new_noFCEL_unc.errplus);New_noFCEL_minus.append(new_noFCEL_unc.errminus)

Old_D_cen, Old_D_plus, Old_D_minus = np.array(Old_D_cen), np.array(Old_D_plus), np.array(Old_D_minus)
Old_cen, Old_plus, Old_minus = np.array(Old_cen), np.array(Old_plus), np.array(Old_minus)
New_FCEL_cen, New_FCEL_plus, New_FCEL_minus = np.array(New_FCEL_cen), np.array(New_FCEL_plus), np.array(New_FCEL_minus)
New_noFCEL_cen, New_noFCEL_plus, New_noFCEL_minus = np.array(New_noFCEL_cen), np.array(New_noFCEL_plus), np.array(New_noFCEL_minus)

# plt.plot(X,Old_cen,color= "green",label = "nNNPDF3.0")
plt.plot(X,Old_cen,color= "green",label = "EPPS21")
plt.fill_between(X,Old_cen-Old_minus,Old_cen+Old_plus,color="green",alpha=0.5)

# plt.plot(X,New_noFCEL_cen,color="magenta",label="nNNPDF3.0 + LHC")
plt.plot(X,New_noFCEL_cen,color="magenta",label="EPPS21 + LHC")
plt.fill_between(X,New_noFCEL_cen-New_noFCEL_minus,New_noFCEL_cen+New_noFCEL_plus,color="magenta",alpha=0.5)

# plt.plot(X,New_FCEL_cen,color="orange",label="nNNPDF3.0 + LHC + FCEL")
plt.plot(X,New_FCEL_cen,color="orange",label="EPPS21 + LHC + FCEL")
plt.fill_between(X,New_FCEL_cen-New_FCEL_minus,New_FCEL_cen+New_FCEL_plus,color="orange",alpha=0.5)

# plt.fill_between(X,Old_D_cen-Old_D_minus,Old_D_cen+Old_D_plus,linestyle = '--',facecolor='none',hatch='/',edgecolor="brown",label="LHCbD")

plt.axhline(y=1,color='grey')

plt.legend()
plt.xlabel("x")
plt.xscale("log")
plt.ylabel(r"$R^{Pb}_g$ at $Q^2=10$ GeV$^2$")
plt.xlim(10**-5,0.7)
plt.ylim(0,1.6)
if original:
    plt.savefig(os.path.join(plots_dir, "original_Rg_comp_"+Pb_new_FCEL+".pdf"))
else:
    plt.savefig(os.path.join(plots_dir, "Rg_comp_"+Pb_new_FCEL+".pdf"))
# plt.savefig(os.path.join(plots_dir, "Rg_comp_.pdf"))
plt.show()
plt.close()