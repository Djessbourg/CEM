# -*- coding: utf-8 -*-

# =============================================================================
# File for computing R_pA of J/psi at LO
# Last modified: 20/04/2026
# =============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import lhapdf as lha
from src import Jpsi as jpsi
from src import Collision, Var_choice
from scipy.special import erfinv
from src import parse_lhapdf_member,create_lhapdf_set,read_info_file, get_info_value,write_member,reduce_set
lha_pdf_path = '/opt/homebrew/Cellar/lhapdf/6.5.4/share/LHAPDF' # maybe find an 'os-friendly' way here



### Initialisation of a sigma object ###
# rs =13000
# rs = 8000
rs = 8160
# rs = 200
# rs = 5000
s = (rs)**2 # CM energy in Gev2
proton_names = ["nNNPDF30_nlo_as_0118_noLHCbD_p", #"nNNPDF30_nlo_as_0118_noLHCbD_p"
			   "CT18ANLO",
			   "nCTEQ15WZSIH_1_1"
			   ]
proton_sets = [lha.getPDFSet(p) for p in proton_names]
proton_lists= [p_set.mkPDFs() for p_set in proton_sets]

Pb_names = ["nNNPDF30_nlo_as_0118_noLHCbD_A208_Z82",
		   "EPPS21nlo_CT18Anlo_Pb208",
		   "nCTEQ15WZSIH_208_82"
		   ]
Pb_sets = [lha.getPDFSet(Pb) for Pb in Pb_names]
Pb_lists= [Pb_set.mkPDFs() for Pb_set in Pb_sets]

std_proton = proton_sets[0]
std = False
# p_cen = False


atom = 'Pb'
# atom = 'Au'

Atom = jpsi.Atom #dictionary where we stock important atomic aspect (now just Z and A) to plot and put into file names

Z = Atom[atom]['Z']
A = Atom[atom]['A']

# LHC_Jpsi = [jpsi.SigmaJpsi(proton,Pb,s,Z,A) for proton in p_list]
LHC_Jpsi = [jpsi.SigmaJpsi(proton_sets[i],proton_lists[i],Pb_sets[i],Pb_lists[i],s,Z,A,p_stand_set=std_proton) for i in range(len(proton_sets))]

### first atempts to reproduce last results ###
num = 0
p_T = 1 # GeV																			# The central set 
# Y = jpsi.Y_list(rs,p_T)														

# A plot for sigma tot and its components
f_size = 15
fig_size=(5,5)
alph = 0.3
q0 =0.07
col_type = 'pA'
err ='mu,q0,pdf'
mu = 2
mu_f = mu

is_pp=Collision(col_type)

RpA_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'RpA_dir')) # the direcory to save data from this file
a=5
b=0	

# n=8.3
# n=19.2
# n=32.3
#8 GeV
# n = 20.81 # mon fit avec +/- 3.67 
# detla_n = 3.67
#13 GeV
# n = 31.05 # mon fit avec +/- 6.03 
# detla_n = 6.03

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
        delta = [np.sqrt(sum([((H[i][2*j-1]-H[i][2*j])**2)/2 for j in range(1,Neig+1)])) for i in range(Nx)]
        return [np.array(cen),0,0,np.array(delta)]
    Oplus = [[max(cen[i],H[i][2*k-1],H[i][2*k])for i in range(Nx)]for k in range(1,Neig+1)]
    Ominus= [[min(cen[i],H[i][2*k-1],H[i][2*k])for i in range(Nx)]for k in range(1,Neig+1)]
    delta_plus = [np.sqrt(sum([(cen[i]-Oplus[k][i])**2 for k in range(Neig)]))for i in range(Nx)]
    delta_minus = [np.sqrt(sum([(cen[i]-Ominus[k][i])**2 for k in range(Neig)]))for i in range(Nx)]
    return [np.array(cen),factor*np.array(delta_plus),factor*np.array(delta_minus)]

def plot_usuals(n=1,s1=f_size,s2=f_size,loca = 'best'):
	plt.legend(frameon= False, fontsize = s1,ncols=n,loc=loca )
	plt.tick_params(labelsize=s2)

fig, ax = plt.subplots(constrained_layout=True)
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
plt.axhline(y=1, color='grey',alpha=alph)

# f_name='no.txt'
# f_name = proton+'_RpA_Jpsi_'+str(rs)+'GeV_Z'+str(Z)+'_A'+str(A)+'_pT'+str(p_T)+'GeV.txt'
# for i,p in enumerate(Pb_names):
def cen_str(p_cen):
	if p_cen:
		return '_cen'
	else:
		return '_corr'

for i,p in enumerate(proton_names if is_pp else Pb_names):
	if is_pp:
		f_name = p+'_Rpp_Jpsi_v3_'+str(rs)+'GeV_Z'+str(Z)+'_A'+str(A)+'_pT'+str(p_T)+'GeV_mu'+str(mu)+cen_str(p_cen)+'.txt'
	else:
		f_name = p+'_RpA_Jpsi_v3_'+str(rs)+'GeV_Z'+str(Z)+'_A'+str(A)+'_pT'+str(p_T)+'GeV_mu'+str(mu)+'.txt'
	# f_name = Pb+'_stand_RpA_Jpsi_v3_'+str(rs)+'GeV_Z'+str(Z)+'_A'+str(A)+'_pT'+str(p_T)+'GeV_mu'+str(mu)+'.txt'
	# f_name = 'test.txt'
	# Y = np.linspace(-5,5,31)
	# Rpa = [LHC_Jpsi[i].RpA(y,p_T,num,mu_factor=mu,mu_f_factor=mu_f)for y in Y]

	if os.path.exists(os.path.join(RpA_dir,f_name)):
		print(f"The file '{f_name}' already exists. It is loaded.")
		# Y,Rpa ,Rpa_plus,Rpa_minus, Rpa_plus_q,Rpa_minus_q,Rpa_plus_mu,Rpa_minus_mu,Rpa_plus_pdf,Rpa_minus_pdf = np.loadtxt(os.path.join(RpA_dir,f_name))
		Y,Rpa ,D_plus,D_minus = np.loadtxt(os.path.join(RpA_dir,f_name))
	else:
		Y = np.linspace(-5,5,31)
		if is_pp:
			info_loc = os.path.join(lha_pdf_path,proton_names[i],proton_names[i]+'.info')
		else:
			info_loc = os.path.join(lha_pdf_path,Pb_names[i],Pb_names[i]+'.info')
		info = read_info_file(info_loc)
		print("The file does not exists")
		if info["ErrorType"] == "hessian" or info["ErrorType"] == "symmHessian" :
			print(info["ErrorType"])
			Val = []
			if p == "EPPS21nlo_CT18Anlo_Pb208":
				n_max = 48
			else:
				if is_pp:
					n_max = len(proton_lists[i])-1
					p_cen = False
				else:
					n_max = len(Pb_lists[i])-1
					p_cen = True
			print("n_max = "+str(n_max))
			for y in Y:
				if is_pp:
					Val.append([LHC_Jpsi[i].Rpp(y,p_T,mem,mu_factor=mu,mu_f_factor=mu_f,std=std,p_cen=p_cen) for mem in range(n_max+1)])
				else:
					Val.append([LHC_Jpsi[i].RpA(y,p_T,mem,mu_factor=mu,mu_f_factor=mu_f,std=std,p_cen=p_cen) for mem in range(n_max+1)])
			Rpa ,D_plus, D_minus = hess_error(Val,cl=68)
		else: # so bayesian case
			print("bayesian")
			Rpa, Rpa_plus,Rpa_minus = [],[],[]
			p_cen = False
			if is_pp:
				n_max = len(proton_lists[i])-1
			else:
				n_max = len(Pb_lists[i])-1
			for y in Y:
				if is_pp:
					R = np.array([LHC_Jpsi[i].Rpp(y,p_T,mem,mu_factor=mu,mu_f_factor=mu_f,std=std,p_cen=p_cen) for mem in range(n_max+1)])
				else:
					R = np.array([LHC_Jpsi[i].RpA(y,p_T,mem,mu_factor=mu,mu_f_factor=mu_f,std=std,p_cen=p_cen) for mem in range(n_max+1)])
				Rpa.append(np.median(R));Rpa_plus.append(np.nanpercentile(R, 84));Rpa_minus.append(np.nanpercentile(R, 16))
			Rpa, Rpa_plus,Rpa_minus = np.array(Rpa),np.array(Rpa_plus),np.array(Rpa_minus)
			D_plus,D_minus = Rpa_plus-Rpa, Rpa-Rpa_minus
				
		# Y,Rpa ,err_Rpa, err_var_Rpa = LHC_Jpsi[i].Uncertainties_RpA_dy(p_T,mu_factor=mu,mu_f_factor=mu_f,var_err= err,version=2)
		# Y,Rpa ,err_Rpa, err_var_Rpa = LHC_Jpsi.Uncertainties_RpA_dy_test(var_err= err,q0_list=[0.075,0.055,0.095])
		# Rpa_plus,Rpa_minus = err_Rpa[0],err_Rpa[1]
		# Rpa_plus_q,Rpa_minus_q,Rpa_plus_mu,Rpa_minus_mu,Rpa_plus_pdf,Rpa_minus_pdf= err_var_Rpa[0],err_var_Rpa[1],err_var_Rpa[2],err_var_Rpa[3],err_var_Rpa[4],err_var_Rpa[5]
		# r = [Y,Rpa,Rpa_plus,Rpa_minus,Rpa_plus_q,Rpa_minus_q,Rpa_plus_mu,Rpa_minus_mu,Rpa_plus_pdf,Rpa_minus_pdf]
		r = [Y,Rpa,D_plus,D_minus] # v3
		np.savetxt(os.path.join(RpA_dir,f_name), r)
		print(f" '{f_name}' has been created")

	plt.plot(Y,Rpa,label=p)
	plt.fill_between(Y,Rpa-D_minus,Rpa+D_plus,alpha=alph)

# R_param = []
# for N in [n,n-detla_n,n+detla_n]:
# 	f_name = str(N)+'_test.txt'
# 	if os.path.exists(os.path.join(RpA_dir,f_name)):
# 		print(f"The file '{f_name}' already exists. It is loaded.")
# 		Y,Rpa ,Rpa_plus,Rpa_minus, Rpa_plus_q,Rpa_minus_q,Rpa_plus_mu,Rpa_minus_mu,Rpa_plus_pdf,Rpa_minus_pdf = np.loadtxt(os.path.join(RpA_dir,f_name))
# 	else:
# 		print("The file does not exists")
# 		Y,Rpa ,err_Rpa, err_var_Rpa = LHC_Jpsi[0].Uncertainties_RpA_dy_test(p_T,n=N,var_err='')
# 		Rpa_plus,Rpa_minus = err_Rpa[0],err_Rpa[1]
# 		Rpa_plus_q,Rpa_minus_q,Rpa_plus_mu,Rpa_minus_mu,Rpa_plus_pdf,Rpa_minus_pdf= err_var_Rpa[0],err_var_Rpa[1],err_var_Rpa[2],err_var_Rpa[3],err_var_Rpa[4],err_var_Rpa[5]
# 		r = [Y,Rpa,Rpa_plus,Rpa_minus,Rpa_plus_q,Rpa_minus_q,Rpa_plus_mu,Rpa_minus_mu,Rpa_plus_pdf,Rpa_minus_pdf]
# 		np.savetxt(os.path.join(RpA_dir,f_name), r)
# 		print(f" '{f_name}' has been created")
# 	R_param.append(Rpa)

# plt.fill_between(Y,R_param[1],R_param[2],color='blue',alpha=alph)
# plt.plot(Y,R_param[0],color='blue',label='param')
if is_pp :
	plt.ylabel(r'$R_{\text{pp}}^{J/\psi}$',fontsize= f_size-a)
else:
	plt.ylabel(r'$R_{\text{pA}}^{J/\psi}$',fontsize= f_size-a)		
# plt.ylabel(r'$R_{\text{stand,pA}}^{J/\psi}$',fontsize= f_size-a)
plt.ylim(0,1.2)
plt.xlim(-5,5)
plt.xlabel('y',fontsize=f_size)
plot_usuals(s1=f_size-a,s2=f_size,loca = 'lower left',n=1)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.text(0.8, 0.9," p"+atom, horizontalalignment='right', verticalalignment='center',transform=ax.transAxes,fontsize=f_size)
plt.text(0.2, 0.9, r'$\sqrt{s} =$'+str(rs/1000) +r' TeV', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes,fontsize=f_size)
# plt.text(0.1, 0.8, r'$\sqrt{s} =$'+str(rs) +r' GeV', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes,fontsize=f_size)
# plt.tight_layout()
# plt.savefig(os.path.join(plots_dir, proton+'_RpA_Jpsi_dy_'+str(rs)+'GeV.pdf'),bbox_inches="tight")
# plt.savefig(os.path.join(plots_dir, proton+'_RpA_Jpsi_v2_dy_'+str(rs)+'GeV.pdf'),bbox_inches="tight")
# plt.savefig(os.path.join(plots_dir, proton+'_RpA_Jpsi_v2_dy_'+str(rs)+'GeV_mu'+str(mu)+'.pdf'),bbox_inches="tight")
# plt.savefig(os.path.join(plots_dir, 'All_RpA_Jpsi_v2_dy_'+str(rs)+'GeV.pdf'),bbox_inches="tight")
if is_pp:
	plt.savefig(os.path.join(plots_dir, 'All_Rpp_Jpsi_v3_dy_'+str(rs)+'GeV.pdf'),bbox_inches="tight")
else: 
	plt.savefig(os.path.join(plots_dir, 'All_RpA_Jpsi_v3_dy_'+str(rs)+'GeV.pdf'),bbox_inches="tight")
# plt.savefig(os.path.join(plots_dir, 'standard_All_RpA_Jpsi_v3_dy_'+str(rs)+'GeV.pdf'),bbox_inches="tight")
plt.show()