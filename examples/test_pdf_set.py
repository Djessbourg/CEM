# -*- coding: utf-8 -*-

# =============================================================================
# File for testing the make_lhapdf_set module
# Last modified: 3/07/2026
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
# from src import Jpsi as jpsi
# from src import reweighting as rw
from src import parse_lhapdf_member,create_lhapdf_set,read_info_file, get_info_value
from scipy.special import erfinv


####################################################################
### test of parse function to retrieve xs, q and flavour vectors ###
####################################################################

# set_name = "EPPS21nlo_CT18Anlo_Pb208"
set_name = "CT14nnlo"
central_loc = os.path.join(lha_pdf_path,set_name,set_name+'_0000.dat')
# get the x,Q,flavour grid 
blocks = parse_lhapdf_member(central_loc)

block = blocks[0]

xs = block["xs"] 
qs = block["qs"]          
flavors = block["flavors"]  
central_values = block["values"]

info_loc = os.path.join(lha_pdf_path,set_name,set_name+'.info')
info = read_info_file(info_loc)
print(info["ErrorType"],info["ErrorConfLevel"])

def ConfLevel_to_sigma(b):
    "b en % (genre 90% => b = 90)"
    return erfinv(b/100.)*(2**0.5)

a = ConfLevel_to_sigma(float(info["ErrorConfLevel"]))
print(f"conversion factor of {a}")

########################################################
### test of create function for the new bayesian set ###
########################################################

asym = False

def asym_str():
    if asym:
        return "asym"
    else:
        return ""
    
if asym :
    new_set_name = "asym2_bayesian_"+set_name
else:
    new_set_name = "bayesian_"+set_name
# Nmem_max = 48               # Number of useful members (48 for EPPS21_Pb bc the 49-106 members are due to proton baseline uncertainties)
Nmem_max = 56 #CT14NNLO
Neig = Nmem_max//2
h_set = lha.getPDFSet(set_name)
h_list = h_set.mkPDFs()
Nrep = 250                  # Number of wanted replicas in the final set

if os.path.exists(os.path.join(lha_new_pdf_path,new_set_name)):
    print(f"The set '{new_set_name}' already exists. It is loaded.")
    h_new_set = lha.getPDFSet(new_set_name) # We don't want to compute it again and again if it already exists
    h_new_list = h_new_set.mkPDFs()
else:
    info_dict = {
        "SetDesc": asym_str()+" Bayesian version of " +set_name,
        "SetIndex": 90000,
        "Authors": "Djessy",
        "Reference": "arXiv:xxxx.xxxxx",
        "Format": "lhagrid1",
        "DataVersion": 1,
        "NumMembers": Nrep + 1 ,
        "Particle": info["Particle"],
        "Flavors": flavors,
        "OrderQCD": info["OrderQCD"],
        "FlavorScheme": info["FlavorScheme"],
        "NumFlavors": info["NumFlavors"],
        "ErrorType": "replicas",
        "XMin": float(xs[0]),
        "XMax": float(xs[-1]),
        "QMin": float(qs[0]),
        "QMax": float(qs[-1]),
        "MZ": info["MZ"],
        "MUp": info["MUp"],
        "MDown": info["MDown"],
        "MStrange": info["MStrange"],
        "MCharm": info["MCharm"],
        "MBottom": info["MBottom"],
        "MTop": info["MTop"],
        "AlphaS_MZ": info["AlphaS_MZ"],
        "AlphaS_OrderQCD": info["AlphaS_OrderQCD"],
        "AlphaS_Type": info["AlphaS_Type"],
        "AlphaS_Qs":  info["AlphaS_Qs"],
        "AlphaS_Vals": info["AlphaS_Vals"]
    }
    R_ik = np.random.normal(0, 1, size=(Neig,Nrep)) # The matrix of the paper whit Nmem_max lines and Nrep col
    members = [{"pdftype": "central", "values": central_values}]
    if asym:
        def d_k(k,fl,x,q):
            f0 = h_list[0].xfxQ(fl,x,q)
            return sum([(h_list[2*i-1].xfxQ(fl,x,q)-h_list[2*i].xfxQ(fl,x,q))*R_ik[i-1][k]/(2*a)+(h_list[2*i-1].xfxQ(fl,x,q)+h_list[2*i].xfxQ(fl,x,q)-2*f0)*(R_ik[i-1][k]**2)/(2*a**2) for i in range(1,Neig+1)])
        
        # def Delta(fl,x,q):
        #     return sum([d_k(k,fl,x,q) for k in range(Nrep)])
        
    Delta = np.zeros(shape=(len(xs),len(qs),len(flavors)),dtype=float) 
    values_k = np.zeros(shape=(Nrep,len(xs),len(qs),len(flavors)),dtype=float)   
    for k in range(Nrep):
        print(f"replica {k+1}/{Nrep}")
        for i,x in enumerate(xs):
            for j,q in enumerate(qs):
                for l,fl in enumerate(flavors):
                    f0 = h_list[0].xfxQ(fl,x,q)
                    if not(asym):
                        values_k[k][i][j][l] = f0 + sum([(h_list[2*idx-1].xfxQ(fl,x,q)-h_list[2*idx].xfxQ(fl,x,q))*R_ik[idx-1][k]/(2*a) for idx in range(1,Neig+1)])
                    elif asym:
                        dk =  d_k(k,fl,x,q)
                        values_k[k][i][j][l] = f0 + dk
                        Delta[i][j][l] +=  dk
    Delta = Delta/Nrep
    for k in range(Nrep):
        values_k[k] = values_k[k]-Delta
        members.append({"pdftype": "replica", "values": values_k[k]})
    
    
    create_lhapdf_set(lha_new_pdf_path,new_set_name,xs,qs,flavors,members,info_dict)
    h_new_set = lha.getPDFSet(new_set_name) # We don't want to compute it again and again if it already exists
    h_new_list = h_new_set.mkPDFs()


#######################################################
### Comparision btwn the hessain and bayesian sets  ###
#######################################################
h_asym_new_set = lha.getPDFSet("asym2_"+new_set_name) # We don't want to compute it again and again if it already exists
h_asym_new_list = h_asym_new_set.mkPDFs()

# Trouver les fonction d'erreur adequates et montrer que le CL 1sigma ou 2 reste identique entre les deux types de stat

# Test avec les pdf simples

q = 1.3 #GeV
X = np.logspace(-5,-0.05,100)
# f_90_to_68 = 1.64485
f_90_to_68 = 1
CL = 68

#test avec gluon
fl = -3
V_old = [[h_list[i].xfxQ(fl,x,q) for i in range(len(h_list))]for x in X]
V_old2 = [[h_list[i].xfxQ(fl,x,q) for i in range(Nmem_max+1)]for x in X]
V_new = [[h_new_list[i].xfxQ(fl,x,q)for i in range(len(h_new_list))]for x in X]
V_new_asym = [[h_asym_new_list[i].xfxQ(fl,x,q)for i in range(len(h_asym_new_list))]for x in X]

def hess_error(H,sym = True, cl=90): # H matrice [x_id][mem]
    factor = ConfLevel_to_sigma(cl)
    f_90_to_68 = 1.64485
    factor = factor/f_90_to_68
    Nx = len(H)
    Neig = (len(H[0])-1)//2
    cen = [H[i][0] for i in range(len(H))]
    if sym:
        delta = [np.sqrt(sum([((H[i][2*j-1]-H[i][2*j])**2)/2 for j in range(1,Neig+1)])) for i in range(Nx)]
        return [np.array(cen),0,0,np.array(delta)]
    Oplus = [[max(cen[i],H[i][2*k-1],H[i][2*k])for i in range(Nx)]for k in range(1,Neig+1)]
    Ominus= [[min(cen[i],H[i][2*k-1],H[i][2*k])for i in range(Nx)]for k in range(1,Neig+1)]
    delta_plus = [np.sqrt(sum([(cen[i]-Oplus[k][i])**2 for k in range(Neig)]))for i in range(Nx)]
    delta_minus = [np.sqrt(sum([(cen[i]-Ominus[k][i])**2 for k in range(Neig)]))for i in range(Nx)]
    return [np.array(cen),factor*np.array(delta_plus),factor*np.array(delta_minus),factor*np.array(max(delta_minus,delta_plus))]

def bayesian_error(H):
    Nx = len(H)
    Nrep = len(H[0])-1
    cen = [sum([H[i][j+1]for j in range(Nrep)])/Nrep for i in range(Nx)]
    delta = [f_90_to_68*np.sqrt(sum([(H[i][j+1]-cen[i])**2 for j in range(Nrep)])/Nrep) for i in range(Nx)]
    return [np.array(cen),np.array(delta)]

def mean_asym():
    return np.array([sum([V_new_asym[x_id][k] for k in range(Nrep)])/Nrep for x_id in range(len(X))])

def asym_Hou_unc(cl= 90):
    factor = ConfLevel_to_sigma(cl)
    mean = mean_asym()
    F_geq, F_leq = [],[]
    for x_id in range(len(X)):
        f_geq, f_leq = [],[]
        for k in range(Nrep):
            f = V_new_asym[x_id][k]
            if f >= mean[x_id]:
                f_geq.append(f)
            else:
                f_leq.append(f)
        f_geq, f_leq = np.array(f_geq),np.array(f_leq)
        F_geq.append(f_geq);F_leq.append(f_leq)
    # F_geq, F_leq = np.array(F_geq), np.array(F_leq)
    Ngeq, Nleq = [len(F_geq[x_id]) for x_id in range(len(X))],[len(F_leq[x_id]) for x_id in range(len(X))] # la somme fait bien Nrep
    df_geq = [np.sqrt(sum([(F_geq[x_id][k]-mean[x_id])**2 for k in range(Ngeq[x_id])])/Ngeq[x_id]) for x_id in range(len(X))]
    print(df_geq)
    df_leq = [np.sqrt(sum([(F_leq[x_id][k]-mean[x_id])**2 for k in range(Nleq[x_id])])/Nleq[x_id]) for x_id in range(len(X))]
    return[mean, factor*np.array(df_geq,dtype=float),factor*np.array(df_leq,dtype=float)]             


Old_cen, Old_plus, Old_minus = [],[],[]
New_cen, New_plus, New_minus = [],[],[]
# New_asym_cen, New_asym_plus, New_asym_minus = [],[],[]
New_asym_cen, New_asym_plus, New_asym_minus = asym_Hou_unc(cl =CL)
print(New_asym_cen)
print(New_asym_minus)

for j in range(len(X)):
    old_unc = h_set.uncertainty(V_old[j]) # put cl=90 to be in the same scale as the EPPS21 raw file
    new_unc = h_new_set.uncertainty(V_new[j],alternative=False)
    # new_asym_unc = h_asym_new_set.uncertainty(V_new_asym[j],cl=90,alternative=False)
    Old_cen.append(old_unc.central);Old_plus.append(old_unc.errplus);Old_minus.append(old_unc.errminus)
    New_cen.append(new_unc.central);New_plus.append(new_unc.errplus);New_minus.append(new_unc.errminus)
    # New_asym_cen.append(new_asym_unc.central);New_asym_plus.append(new_asym_unc.errplus);New_asym_minus.append(new_asym_unc.errminus)
Old_cen, Old_plus, Old_minus = np.array(Old_cen), np.array(Old_plus), np.array(Old_minus)
New_cen, New_plus, New_minus = np.array(New_cen), np.array(New_plus), np.array(New_minus)
# New_asym_cen, New_asym_plus, New_asym_minus = np.array(New_asym_cen), np.array(New_asym_plus), np.array(New_asym_minus)

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

plt.sca(ax1)
# Old_hess = hess_error(V_old2,sym=False,cl=CL)
Old_hess = [Old_cen, Old_plus, Old_minus]

plt.plot(X,Old_hess[0],color='green',label=f"old (hesssian DIY, {Neig} eig)")
plt.fill_between(X,Old_hess[0]-Old_hess[2],Old_hess[1]+Old_hess[0],color='green',alpha=0.3)
plt.plot(X,New_cen,color='blue',label = 'Bayesian sym')
plt.fill_between(X,New_cen-New_minus,New_cen+New_plus,color='blue',alpha=0.3)
plt.plot(X,New_asym_cen,color='red',label = 'Bayesian asym')
plt.fill_between(X,New_asym_cen-New_asym_minus,New_asym_cen+New_asym_plus,color='red',alpha=0.3)


plt.xlabel('x')
plt.xlim(5*10**-5,1)
# plt.ylim(0.5,1.5)
plt.xscale('log')
# plt.yscale('log')
# plt.ylabel('gluon PDF EPPS21 Pb')
# plt.ylabel(r'$\bar{u}$ for CT14NNLO')
# plt.ylabel(r'$d$ for CT14NNLO')
plt.ylabel(r'$\bar{s}$ for CT14NNLO')
# plt.ylabel(r'$g$ for CT14NNLO')
plt.title(r'absolute comparison at 68%')
plt.legend()
plt.grid()

plt.sca(ax2)  # Définir ax2 comme subplot actif
plt.axhline(y=1,color='green',label=f"old (hesssian DIY, {Neig} eig)")
plt.fill_between(X,(Old_hess[0]-Old_hess[2])/Old_hess[0],(Old_hess[0]+Old_hess[1])/Old_hess[0],color='green',alpha=0.3)

# plt.plot(X,(New_cen-Old_hess[0])/Old_hess[0],color='blue',label = 'Bayesian sym')
# plt.fill_between(X,(New_cen-New_minus-Old_hess[0])/Old_hess[0],(New_cen+New_plus-Old_hess[0])/Old_hess[0],color='blue',alpha=0.3)
# plt.plot(X,(New_asym_cen-Old_hess[0])/Old_hess[0],color='red',label = 'Bayesian asym')
# plt.fill_between(X,(New_asym_cen-New_asym_minus-Old_hess[0])/Old_hess[0],(New_asym_cen+New_asym_plus-Old_hess[0])/Old_hess[0],color='red',alpha=0.3)

plt.plot(X,(New_cen)/Old_hess[0],color='blue',label = 'Bayesian sym')
plt.fill_between(X,(New_cen-New_minus)/Old_hess[0],(New_cen+New_plus)/Old_hess[0],color='blue',alpha=0.3)
plt.plot(X,(New_asym_cen)/Old_hess[0],color='red',label = 'Bayesian asym')
plt.fill_between(X,(New_asym_cen-New_asym_minus)/Old_hess[0],(New_asym_cen+New_asym_plus)/Old_hess[0],color='red',alpha=0.3)
plt.xlim(5*10**-5,1)
# plt.ylim(0.5,1.5)
plt.ylim(0,2)
plt.xscale('log')
plt.title(r'relative comparison with hessian at 68%')
plt.grid()
plt.legend()

# Ajustement automatique de la mise en page
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'Relative_hessian_bayesian.pdf'))
plt.show()
plt.close()

