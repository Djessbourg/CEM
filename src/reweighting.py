# -*- coding: utf-8 -*-

# =============================================================================
# Rewighting code for PDF sets (Bayesyan at that moment)
# Credits : BOURGEAIS Djessy; ARLEO Francois.
# Subatech (IN2P3,CNRS)
# Nantes, France
# Last modified: 02/07/2026
# =============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
omega_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'omega'))
lha_pdf_path = '/opt/homebrew/Cellar/lhapdf/6.5.4/share/LHAPDF' # maybe find an 'os-friendly' way here

lha_new_pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'myPDFs'))
import lhapdf as lha
lha.pathsAppend(lha_new_pdf_path)

import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections.abc import Callable
from . import parse_lhapdf_member,create_lhapdf_set,read_info_file, get_info_value,write_member,reduce_set
from scipy.special import erfinv 

def ConfLevel_to_sigma(b):
                "b en % (genre 90% => b = 90)"
                return erfinv(b/100.)*(2**0.5)

def Chi2(Th:list[float] ,D: list[float],dD:list[float])->float:
    if (len(Th)!=len(D))|(len(D)!=len(dD)):
        raise ValueError("Argument lists don't share the same size")
    R = 0
    for i in range(len(Th)):
        R += ((Th[i]-D[i])/dD[i])**2
    return R

def Data_info_dict(rs:float,x_type:str,x_value:list[float],x_unc:list[float])->dict:
    d = {"rs" :rs,
         "x_type": x_type,
         "x_value": x_value,
         "x_unc": x_unc #Here, think about what uncertainties to use 
    }
    return d 

def open_data(Obs_name,name):
    f_name = f"Omega_file_{Obs_name}_{name}.pickle"
    with open(os.path.join(omega_dir,f_name), 'rb') as f:
        d = pk.load(f)
    return d

def ConfLevel_to_sigma(b):
    "b en % (genre 90% => b = 90)"
    if b == 68:
        return 1
    else:
        return erfinv(b/100.)*(2**0.5)

def d_k(A_list,R_ik,Neig,k,fl,x,q):
    f0 = A_list[0].xfxQ(fl,x,q)
    return sum([(A_list[2*i-1].xfxQ(fl,x,q)-A_list[2*i].xfxQ(fl,x,q))*R_ik[i-1][k]/(2*1)+(A_list[2*i-1].xfxQ(fl,x,q)+A_list[2*i].xfxQ(fl,x,q)-2*f0)*(R_ik[i-1][k]**2)/(2*1**2) for i in range(1,Neig+1)])

def original_d_k(A_list,R_ik,Neig,k,fl,x,q):
    return sum([(A_list[2*i-1].xfxQ(fl,x,q)-A_list[2*i].xfxQ(fl,x,q))*R_ik[i-1][k]/(2*1) for i in range(1,Neig+1)])
     
def to_bayesianize(set_name:str,Neig:int,Nrep:int,R_ik:np.ndarray=None,original:bool=False):
    try:
        info_loc = os.path.join(lha_pdf_path,set_name,set_name+'.info')
        info = read_info_file(info_loc)
    except:
        info_loc = os.path.join(lha_new_pdf_path,set_name,set_name+'.info')
        info = read_info_file(info_loc)
    if R_ik is None:
        R_ik = np.random.normal(0, 1, size=(Neig, Nrep)) # The matrix of the paper whith Neig lines and Nrep col
    else:
        if R_ik.shape != (Neig, Nrep):
            raise ValueError(f"R_ik should have shape ({Neig}, {Nrep}), but has shape {R_ik.shape}.")
    central_loc = os.path.join(lha_pdf_path,set_name,set_name+'_0000.dat')
    # get the x,Q,flavour grid 
    blocks = parse_lhapdf_member(central_loc)
    block = blocks[0]
    xs = block["xs"]
    qs = block["qs"]          
    flavors = block["flavors"]  
    central_values = block["values"]
    try:
        a = ConfLevel_to_sigma(float(info["ErrorConfLevel"]))
    except:
        a = 1.
    
    conv_fact = ConfLevel_to_sigma(float(info["ErrorConfLevel"]))
    print(f"conversion factor of {conv_fact}")
    new_set_name = "original_bayesianised_"+set_name+'_'+str(Nrep) if original else "bayesianised_"+set_name+'_'+str(Nrep) 
    
    if os.path.exists(os.path.join(lha_new_pdf_path,new_set_name)):
        print(f"The set '{new_set_name}' already exists. It is loaded.")
        h_new_set = lha.getPDFSet(new_set_name) # We don't want to compute it again and again if it already exists
        h_new_list = h_new_set.mkPDFs()
        A_set = h_new_set
        A_list = h_new_list
        return [A_set,A_list,R_ik]
    else:
        members = [{"pdftype": "central", "values": central_values}]
        values_k = np.zeros(shape=(Nrep,len(xs),len(qs),len(flavors)),dtype=float)   
        if not(original):
            Delta = np.zeros(shape=(len(xs),len(qs),len(flavors)),dtype=float) 
            for k in range(Nrep):
                print(f"replica {k+1}/{Nrep}")
                for i,x in enumerate(xs):
                    for j,q in enumerate(qs):
                        for l,fl in enumerate(flavors):
                            f0 = A_list[0].xfxQ(fl,x,q)
                            dk =  d_k(A_list,R_ik,Neig,k,fl,x,q)
                            values_k[k][i][j][l] = f0 + dk
                            Delta[i][j][l] +=  dk
            Delta = Delta/Nrep
            for k in range(Nrep):
                values_k[k] = values_k[k]-Delta
                members.append({"pdftype": "replica", "values": values_k[k]})
        else:
            for k in range(Nrep):
                print(f"replica {k+1}/{Nrep}")
                for i,x in enumerate(xs):
                    for j,q in enumerate(qs):
                        for l,fl in enumerate(flavors):
                            f0 = A_list[0].xfxQ(fl,x,q)
                            dk =  original_d_k(A_list,R_ik,Neig,k,fl,x,q)
                            values_k[k][i][j][l] = f0 + dk
                members.append({"pdftype": "replica", "values": values_k[k]})
    
        info_dict = {
        "SetDesc": ("Bayesian (original method) version of " +set_name) if original else ("Bayesian version of " +set_name),
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
        create_lhapdf_set(lha_new_pdf_path,new_set_name,xs,qs,flavors,members,info_dict)
        # Renewing the set with the bayesian version
        h_new_set = lha.getPDFSet(new_set_name) # We don't want to compute it again and again if it already exists
        h_new_list = h_new_set.mkPDFs()
        return [h_new_set,h_new_list,R_ik]

class Reweight:
    def __init__(self, A_set,A_list, Nrep, Data:dict,Data_info: dict, Obs:dict,Nmem_max:int=-1,FCEL = True,local = False,original = False,name_adons =''):		# Obs as to be Callable[[float,int],float]				
        self.A_set = A_set										# create the list of all members for the pdf set
        self.A_list = A_list									# Initialise the pdf member to 0
        self.Data = Data
        self.Data_info = Data_info
        self.k_names= list(self.Data.keys())
        self.k0 = self.k_names[0] # first key
        self.rs = {kn:self.Data_info[kn]["rs"] for kn in self.k_names}
        self.x_type = {kn:self.Data_info[kn]["x_type"] for kn in self.k_names}
        self.x_value = {kn:self.Data_info[kn]["x_value"] for kn in self.k_names}
        self.x_unc = {kn:self.Data_info[kn]["x_unc"] for kn in self.k_names}
        self.Ndata = sum([len(self.Data[kn]) for kn in self.k_names])
        self.Nmem_max = Nmem_max # has to be an even nummber bc it's 2*Neig
        self.Neig = int(self.Nmem_max/2)
        set_name = self.A_set.name
        self.set_name = set_name
        self.original = original # say if you want the original version of bayesianisation
        if name_adons =='':
            self.name_adons = name_adons
        else:
            self.name_adons = name_adons+'_'
        try:
            info_loc = os.path.join(lha_pdf_path,set_name,set_name+'.info')
            self.info = read_info_file(info_loc)
        except:
            info_loc = os.path.join(lha_new_pdf_path,set_name,set_name+'.info')
            self.info = read_info_file(info_loc)
        self.FCEL = FCEL
        self.Nrep = Nrep
        if self.info["ErrorType"] == "hessian" or self.info["ErrorType"] == "symmHessian" :
            self.is_hessian = True
            self.local = True
        else:
            self.is_hessian = False
            if self.Nrep == len(self.A_list)-1 :
                self.local = local
            else: # To Do: générer un set bayesien plus grand a partir du set de base pour remplir ce cas de figure
                self.local = local
        if self.is_hessian:
            self.A_set,self.A_list,self.R_ik = to_bayesianize(set_name,self.Neig,self.Nrep,R_ik=None,original=self.original)
        self.Obs = Obs # Try to think about how to modify this with hessian set getting bayesianised
        self.obs_dict = {} # keys are the k_names, then x_values, each elts are a list from 0 to Nrep-1 of each Obs(i) for one x
        self.obs_mean_dict = {}
        self.obs_new_mean_dict= {}
        self.Chi2_list = []
        self.set_Chi_squared()
        self.obs_dict_mem = {kn:{str(i):[self.obs_dict[kn][str(x)][i-1] for x in self.x_value[kn]] for i in range(1,self.Nrep+1)} for kn in self.k_names}# how to modify that with multiple data values ?
        self.D = self.D_norm()
        self.Omega_list = []
        self.set_omega()
        self.N_eff = int(np.round(self.Neff()))
        self.reweighting_info()
        self.unweight(FCEL=self.FCEL)

    def FCEL_str(self):
        if self.FCEL:
            return "FCEL"
        else:
            return "noFCEL"
        
    def read_info(self):
        for kn in self.k_names:
            print(50*'=')
            print(f"{kn:^50}")
            print(50*'=')
            for k in self.Data_info.keys():
                print("k = {}, d[k]= {}".format(k,self.Data_info[kn][k]))

    def set_Chi_squared(self): # issue here if the entry set is hessian bc the Obs will vary differently with i in the bayesianised version
        """Genrate all the Chi_i^2 and put them into self.Chi
        """
        print(20*"-")
        print("Computing Xi")
        for i in range(1,self.Nrep+1):
            print("i = {}/{}".format(i,self.Nrep))
            Th = {}
            for kn in self.k_names:
                Th[kn] = []
                try:
                    D = self.obs_dict[kn]
                except:
                    self.obs_dict[kn] = {}
                    D = self.obs_dict[kn]
                for x in self.x_value[kn]:
                    try:
                        d = D[str(x)]
                    except:
                        D[str(x)] = [None for j in range(self.Nrep)]
                        d = D[str(x)]
                    O = self.Obs[kn](x,i)
                    d[i-1] = O
                    Th[kn].append(O)
            # We merge the data_list to compute Chi2
            T,Data,deltaD = [],[],[]
            for kn in self.k_names:
                T += Th[kn]
                Data += self.Data[kn]
                deltaD += self.x_unc[kn]
            self.Chi2_list.append(Chi2(T,Data,deltaD))

    def d_k(self,k,fl,x,q):
        f0 = self.A_list[0].xfxQ(fl,x,q)
        return sum([(self.A_list[2*i-1].xfxQ(fl,x,q)-self.A_list[2*i].xfxQ(fl,x,q))*self.R_ik[i-1][k]/(2*self.a)+(self.A_list[2*i-1].xfxQ(fl,x,q)+self.A_list[2*i].xfxQ(fl,x,q)-2*f0)*(self.R_ik[i-1][k]**2)/(2*self.a**2) for i in range(1,self.Neig+1)])

    def original_d_k(self,k,fl,x,q):
        return sum([(self.A_list[2*i-1].xfxQ(fl,x,q)-self.A_list[2*i].xfxQ(fl,x,q))*self.R_ik[i-1][k]/(2*self.a) for i in range(1,self.Neig+1)])
        
    def get_h_to_b_list(self):
        return self.h_to_b_list

    def D_norm(self):
        # print(20*"-")
        # print("computing D normalisation factor")
        R = 0
        for chi2 in self.Chi2_list:
            R += chi2**((self.Ndata-1)/2)*np.exp(-chi2/2)
        return R/self.Nrep
    
    def set_omega(self):
        # print(20*"-")
        # print("Computing the weights")
        for i,chi2 in enumerate(self.Chi2_list):
            # print("i = {}/{}".format(i+1,self.Nrep))
            omega = chi2**((self.Ndata-1)/2)*np.exp(-chi2/2)/self.D
            self.Omega_list.append(omega)
    
    def Obs_mean(self,kn,x):
        if kn not in self.k_names:
            raise KeyError(f"{kn} is not a legal key, try something in {self.k_names}.")
        try:
            return self.obs_mean_dict[kn][str(x)]
        except:
            try:
                M = self.obs_mean_dict[kn]
            except:
                self.obs_mean_dict[kn] = {}
                M = self.obs_mean_dict[kn]  
        try:
            d = self.obs_dict[kn][str(x)]
        except:
            try:
                D = self.obs_dict[kn]
            except:
                self.obs_dict[kn] = {}
                D = self.obs_dict[kn]
            # print("creaitng a new key")
            D[str(x)] = []
            d = D[str(x)]
        if d == []:
            O = 0
            for i in range(1,self.Nrep+1):
                # print(" i = {}/{}".format(i,self.Nrep))
                d.append(self.Obs[kn](x,i))
                O += d[i-1] # i-1 bc we start at i = 1
            O = O/self.Nrep
            M[str(x)] = O
            return O
        else:
            O = 0
            for i in range(1,self.Nrep+1):
                # print(" i = {}/{}".format(i,self.Nrep))
                O += d[i-1] # i-1 bc we start at i = 1 
            O = O/self.Nrep
            M[str(x)] = O
            return O

    def delta_Obs(self,kn,x):
        # print(50*"-")
        # print("delta_obs")
        if kn not in self.k_names:
            raise KeyError(f"{kn} is not a legal key, try something in {self.k_names}.")
        try:
            M = self.obs_dict[kn]
        except:
            self.obs_dict[kn] = {}
            M = self.obs_dict[kn]
        try:
            d = M[str(x)]
        except:
            # print("creaitng a new key")
            M[str(x)] = []
            d = M[str(x)]
        if d == []:
            O = 0
            for i in range(1,self.Nrep+1):
                # print(" i = {}/{}".format(i,self.Nrep))
                d.append(self.Obs[kn](x,i))
                O += (d[i-1]-self.Obs_mean(kn,x))**2
            return np.sqrt(O/self.Nrep)
        else:
            O = 0
            for i in range(1,self.Nrep+1):
                # print(" i = {}/{}".format(i,self.Nrep))
                O += (d[i-1]-self.Obs_mean(kn,x))**2
            return np.sqrt(O/self.Nrep)
    
    def Obs_mean_new(self,kn,x):
        if kn not in self.k_names:
            raise KeyError(f"{kn} is not a legal key, try something in {self.k_names}.")
        try:
            return self.obs_new_mean_dict[kn][str(x)]
        except:
            try:
                N =self.obs_new_mean_dict[kn]
            except:
                self.obs_new_mean_dict[kn] = {}
                N = self.obs_new_mean_dict[kn]
        try:
            M = self.obs_dict[kn]
        except:
            self.obs_dict[kn] = {}
            M = self.obs_dict[kn]
        try:
            d = M[str(x)]
        except:
            # print("creaitng a new key")
            M[str(x)] = []
            d = M[str(x)]
        if d == []:
            O = 0
            for (i,omega) in enumerate(self.Omega_list): # Careful here, bc i starts at 0 bc of the enumerate function
                d.append(self.Obs[kn](x,i))
                O += omega*d[i] # So we modify i-1 here by just i 
            O = O/self.Nrep
            N[str(x)] = O
            return O
        else:
            O = 0
            for (i,omega) in enumerate(self.Omega_list):
                O += omega*d[i]
            O = O/self.Nrep
            N[str(x)] = O
            return O

    def delta_Obs_new(self,kn,x):
        # print(50*"-")
        # print("delta_obs_new")
        if kn not in self.k_names:
            raise KeyError(f"{kn} is not a legal key, try something in {self.k_names}.")
        try:
            M = self.obs_dict[kn]
        except: 
            self.obs_dict[kn] = {}
            M = self.obs_dict[kn]
        try:
            d = M[str(x)]
        except:
            # print("creaitng a new key")
            M[str(x)] = []
            d =  M[str(x)]
        if d == []:
            O = 0
            for (i,omega) in enumerate(self.Omega_list):
                d.append(self.Obs[kn](x,i))
                O += omega*(d[i]-self.Obs_mean_new(kn,x))**2
            return np.sqrt(O/self.Nrep)
        else:
            O = 0
            for (i,omega) in enumerate(self.Omega_list):
                O += omega*(d[i]-self.Obs_mean_new(kn,x))**2
            return np.sqrt(O/self.Nrep)
    
    def Neff(self):
        r = 0
        for omega in self.Omega_list:
            if omega == 0:  # permet de skip la division par 0 
                continue
            r+= omega*np.log(self.Nrep/omega)
        return np.exp(r/self.Nrep)

    def reweighting_info(self):
        print(50*"=")
        print(f"{'Reweighting Info':^50}")
        print(50*"-")
        print(f"Nrep={self.Nrep:<15} Neff={self.N_eff:<15}")
        print(50*'=')

    def plot_obs(self,obs_name,Ylim = [0.5,1.6],Xlim = [-5,5],member = False):
        for kn in self.k_names:
            X = self.x_value[kn]
            Vold = np.array([self.Obs_mean(kn,x) for x in X ])
            dVold = np.array([self.delta_Obs(kn,x) for x in X ])
            Vnew = np.array([self.Obs_mean_new(kn,x) for x in X ])
            dVnew = np.array([self.delta_Obs_new(kn,x) for x in X ])
            if member:
                for i in range(1,self.Nrep+1):
                    plt.plot(X,self.obs_dict_mem[kn][str(i)],'--',color='purple',linewidth=0.5,alpha= 0.1)
            plt.plot(X,Vold,color='red',label = 'old')
            plt.plot(X,Vnew,color = 'blue',label= 'new')
            plt.fill_between(X,Vold-dVold,Vold+dVold,color = 'red',alpha= 0.5)
            plt.fill_between(X,Vnew-dVnew,Vnew+dVnew,color = 'blue',alpha= 0.5)
            plt.errorbar(X,self.Data[kn],yerr=self.x_unc[kn],fmt='+',color = 'black',label ='Data '+kn)
            plt.xlabel(self.x_type[kn])
            plt.ylabel(obs_name)
            plt.legend()
            plt.ylim(Ylim[0],Ylim[1])
            plt.savefig(os.path.join(plots_dir,self.FCEL_str()+'_'+self.name_adons+'test_reweighting_'+self.set_name+'_'+kn+'.pdf'),bbox_inches="tight")
            plt.close()
            # plt.show()

    def plot_obs_3D(self,obs_name):
        for kn in self.k_names:
            ax = plt.figure().add_subplot(projection='3d')

            X = self.x_value[kn]
            Z = np.zeros(len(X))
            Vold = np.array([self.Obs_mean(kn,x) for x in X])
            dVold = np.array([self.delta_Obs(kn,x) for x in X])
            Vnew = np.array([self.Obs_mean_new(kn,x) for x in X])
            dVnew = np.array([self.delta_Obs_new(kn,x) for x in X])

            # ax.plot(X,Vold, zs=0,color='red', zdir='z', label='Old')
            ax.fill_between(X, Vold-dVold,0,X,Vold+dVold,0,color='red',alpha=0.5, label='Old')
            # ax.plot(X,Vnew, zs=0,color='blue', zdir='z', label='new')
            ax.fill_between(X, Vnew-dVnew,0,X,Vnew+dVnew,0,color='blue',alpha=0.5, label='new')
            ax.errorbar(X,self.Data[kn],Z,yerr=self.x_unc[kn],fmt='+',color ='black',label = 'Data '+kn)
            # Plot scatterplot data (20 2D points per colour) on the x and z axes.

            cmap = cm.plasma
            m,M = min(self.Omega_list),max(self.Omega_list)
            # print(f"omega min = {m}, omega max = {M}")
            norm = mcolors.Normalize(vmin=m, vmax=M)

            for i,omega in enumerate(self.Omega_list):
                Obsi = self.obs_dict_mem[kn][str(i+1)]
                # print(omega)
                col = cmap(norm(omega))
                ax.plot(X, Obsi, zs=omega,color=col,zdir='z',alpha = 0.3)

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # cbar = plt.colorbar(sm, ax=ax)

            # Make legend, set axes limits and labels
            ax.legend()
            ax.set_xlim(min(self.x_value[kn]), max(self.x_value[kn]))
            ax.set_ylim(0.4, 1.1)
            ax.set_zlim(0, M+1)
            ax.set_xlabel(self.x_type[kn])
            ax.set_ylabel(obs_name)
            ax.set_zlabel(r'$\omega$')

            # Customize the view angle so it's easier to see that the scatter points lie
            # on the plane y=0
            ax.view_init(elev=20., azim=-15, roll=0)
            plt.savefig(os.path.join(plots_dir,self.FCEL_str()+'_'+self.name_adons+'test_reweighting_3D_'+self.set_name+'_'+kn+'.pdf'),bbox_inches="tight")
            plt.close()
            # plt.show()

    def plot_chi2_distribution(self,Nbins = None,Chi_max = 20): # Do a comparison btw old and new chi2 ?
        Chis = np.array(self.Chi2_list)
        Omegas = np.array(self.Omega_list)
        Omegas_tilde = Omegas*self.N_eff/self.Nrep 
        Bins = np.linspace(0,Chi_max,Nbins+1)
        fig, ax = plt.subplots(constrained_layout=True)
        plt.hist(Chis/self.Ndata,bins=Bins,color='blue',alpha=0.5,histtype="bar", linewidth=0.3,
        edgecolor='blue',label = r'$\chi^2$')
        plt.hist(Chis/self.Ndata,bins=Bins,weights=Omegas_tilde,color='green',alpha=0.5,histtype="bar", linewidth=0.3,edgecolor='green',label = r'$\tilde{\omega}\chi^2$')
        plt.xlabel(r'$\chi^2/\text{ndf}$')
        plt.ylabel(r'$\frac{\text{d}N}{\text{d}\chi^2}$')
        plt.text(0.5, 0.9,r"$N_\text{eff}/N_\text{rep} = $ "+str(self.N_eff*100/self.Nrep) +'%', horizontalalignment='center', verticalalignment='center',transform=ax.transAxes,fontsize=12)
        plt.xlim(0,Chi_max)
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(plots_dir,self.FCEL_str()+'_'+self.name_adons+'Chi2_comp'+self.set_name+'.pdf'),bbox_inches="tight")
        plt.show()

    def save_data(self,Obs_name,name): # The idea here is to save the omega_i in a file with the data info
        r = {
            'name': name,
            'rs':self.rs,
            'Obs_name':Obs_name,
            'FCEL': self.FCEL_str(),
            'x_type':self.x_type,
            'x_value':self.x_value,
            'x_unc':self.x_unc,
            'Data':self.Data,
            'Chi2_list':self.Chi2_list,
            'Omega_list':self.Omega_list,
            'Neff':self.N_eff
        }
        f_name = f"Omega_file_{Obs_name}_{name}.pickle"
        with open(os.path.join(omega_dir,f_name), 'wb') as f:
            # Pickle the 'r' dictionary using the highest protocol available.
            pk.dump(r, f, pk.HIGHEST_PROTOCOL)

    def central_member(self,new_set_name):
        set_name = self.A_set.name
        central_loc = os.path.join(lha_pdf_path,set_name,set_name+'_0000.dat') if not(self.local) else os.path.join(lha_new_pdf_path,set_name,set_name+'_0000.dat')
        new_set_dir = os.path.join(lha_new_pdf_path,new_set_name,new_set_name+'_0000.dat') 
        # get the x,Q,flavour grid 
        blocks = parse_lhapdf_member(central_loc)
        block = blocks[0]
        xs = block["xs"] 
        qs = block["qs"]          
        flavors = block["flavors"]  
        central_values = block["values"]
        members = [{"pdftype": "central", "values": central_values}]
        Delta = np.zeros(shape=(len(xs),len(qs),len(flavors)),dtype=float) 
        values = np.zeros(shape=(len(xs),len(qs),len(flavors)),dtype=float)   
        Omega = self.Omega_list
        for k in range(self.Nrep):
            for i,x in enumerate(xs):
                for j,q in enumerate(qs):
                    for l,fl in enumerate(flavors):
                        values[i][j][l] += self.A_list[k+1].xfxQ(fl,x,q)*Omega[k]
        values = values/self.Nrep
        write_member(new_set_dir,"central",xs,qs,flavors,values)
        
    def unweight(self,FCEL=True):
        set_name = self.A_set.name
        N_eff = self.N_eff
        if FCEL:
            new_set_name = "unweighted_"+str(N_eff)+"rep_FCEL"+'_'+self.name_adons+set_name
        else:
            new_set_name = "unweighted_"+str(N_eff)+"rep_noFCEL"+'_'+self.name_adons+set_name
        set_dir = os.path.join(lha_new_pdf_path,new_set_name)
        os.makedirs(set_dir, exist_ok=True)
        if self.local:
            old_path = lha_new_pdf_path
        else:
            old_path = lha_pdf_path
        # First, determine wich replica of the set will be keept by the following statistical draw
        X = np.random.uniform(0,self.Nrep,size=N_eff)
        replica_list=np.zeros(shape=self.Nrep,dtype=int)
        for x in X:
            omega_min = 0
            omega_max = np.copy(self.Omega_list[0])
            for i,omega in enumerate(self.Omega_list):
                if (omega_min<=x) and (x<omega_max):
                    replica_list[i] += 1
                    break
                omega_min += np.copy(omega)
                if i != len(self.Omega_list)-1:
                    omega_max += np.copy(self.Omega_list[i+1])
        # central set
        self.central_member(new_set_name)
        # then the rest of it
        reduce_set(replica_list,old_path,lha_new_pdf_path,set_name,new_set_name,FCEL=FCEL)
        
class Weights:
    def __init__(selfw,A_set,A_list,data_d:dict):
        selfw.A_set = A_set									
        selfw.A_list = A_list
        selfw.d = data_d
        selfw.rs = selfw.d['rs']
        selfw.Obs_name = selfw.d['Obs_name']
        selfw.x_type = selfw.d['x_type']
        selfw.x_value = selfw.d['x_value']
        selfw.x_unc = selfw.d['x_unc']
        selfw.Data = selfw.d['Data']
        selfw.Chi2_list = selfw.d['Chi2_list']
        selfw.Omega_list = selfw.d['Omega_list']
        selfw.N_eff = selfw.d['Neff']
        selfw.k_names= list(selfw.Data.keys())
        selfw.Ndata = sum([len(selfw.Data[kn]) for kn in selfw.k_names])
        selfw.Nrep = len(A_list)-1
        selfw.Obs_id = 0                  # To link computation of Obs to an acutal observable
        selfw.Obs = {"0":lambda x,i:1}                    # dictionary of all observables used
        selfw.actual_obs = lambda x,i:1
        selfw.id_max = 0
        selfw.Obs_dict = {}
        selfw.Obs_new_mean_dict ={}
        
    def set_Obs(selfw,O:Callable[[float,int],float],id:int = -1):
        if id in range(0,selfw.id_max+1):
            selfw.actual_obs = selfw.Obs[str(id)]
            selfw.Obs_id = id
            print("observalble alredy existing")
        else:
            if (id != selfw.id_max+1):
                print(f"Caution ! id = {id} instead of id_max + 1 = {selfw.id_max+1} (for a new obs) or just in [0,{selfw.id_max}] to select an existing one")
                print(f"We will replace id = {id} by {selfw.id_max+1}")
            print("Saving the new observable")
            id = selfw.id_max+1
            selfw.Obs[str(id)] = O
            selfw.Obs_dict[str(id)]= {}
            selfw.Obs_new_mean_dict[str(id)]={}
            selfw.actual_obs = O
            selfw.Obs_id = id
            selfw.id_max += 1

    def plot_chi2_distribution(selfw,Nbins = 20,Chi_max = 30,name_plot=''): # Do a comparison btw old and new chi2 ?
        Chis = np.array(selfw.Chi2_list)
        Omegas = np.array(selfw.Omega_list)
        Omegas_tilde = Omegas*selfw.N_eff/selfw.Nrep
        Bins = np.linspace(0,Chi_max,Nbins+1)
        fig, ax = plt.subplots(constrained_layout=True)
        plt.hist(Chis/selfw.Ndata,bins=Bins,color='blue',alpha=0.5,histtype="bar", linewidth=0.3,
        edgecolor='blue',label = r'$\chi^2$')
        # plt.hist(Chis/selfw.Ndata,bins=Bins,weights=Omegas,color='yellow',alpha=0.5,histtype="bar", linewidth=0.3,
        # edgecolor='yellow',label = r'$\omega\chi^2$')
        plt.hist(Chis/selfw.Ndata,bins=Bins,weights=Omegas_tilde,color='green',alpha=0.5,histtype="bar", linewidth=0.3,
        edgecolor='green',label = r'$\tilde{\omega}\chi^2$')
        plt.xlabel(r'$\chi^2/\text{ndf}$')
        plt.ylabel(r'$\text{d}N/\text{d}\chi^2$')
        plt.xlim(0,Chi_max)
        y_min,y_max = ax.get_ylim()
        plt.ylim = (y_min,3*y_max)
        plt.text(0.5, 0.95,r"$N_\text{eff}/N_\text{rep} = $ "+str(selfw.N_eff*100/selfw.Nrep) +'%', horizontalalignment='center', verticalalignment='center',transform=ax.transAxes,fontsize=12)
        plt.legend()
        plt.yscale('log')
        plt.savefig(os.path.join(plots_dir,'Chi2_comp'+name_plot+'.pdf'),bbox_inches="tight")
        plt.show()

    def Obs_mean(selfw,x:float,Obs:Callable[[float,int],float],id:int = -1 )->float:
        if id == -1: #means put a new obs
                id = selfw.id_max+1
        try:
            return selfw.Obs_dict[str(id)]["mean"][str(x)]
        except:
            try:
                mean = selfw.Obs_dict[str(id)]["mean"]
            except:
                selfw.set_Obs(Obs,id)
                selfw.Obs_dict[str(id)]["mean"]={}
                mean = selfw.Obs_dict[str(id)]["mean"]
            try:
                d = selfw.Obs_dict[str(id)]["mem"][str(x)]
            except:
                try:
                    f = selfw.Obs_dict[str(id)]["mem"]
                except:
                    selfw.Obs_dict[str(id)]["mem"] = {}
                    f = selfw.Obs_dict[str(id)]["mem"]
                
                selfw.Obs_dict[str(id)]["mem"][str(x)]=[]
                d = selfw.Obs_dict[str(id)]["mem"][str(x)]
            if d == []:
                O = 0
                for i in range(1,selfw.Nrep+1):
                    # print(" i = {}/{}".format(i,self.Nrep))
                    d.append(selfw.Obs[str(id)](x,i))
                    O += d[i-1] # i-1 bc we start at i = 1
                O = O/selfw.Nrep
                mean = O
                return O
            else:
                O = 0
                for i in range(1,selfw.Nrep+1):
                    # print(" i = {}/{}".format(i,self.Nrep))
                    O += d[i-1] # i-1 bc we start at i = 1 
                O = O/selfw.Nrep
                mean = O
                return O

    def delta_Obs(selfw,x:float,Obs:Callable[[float,int],float],id:int=-1)->float:
        if id == -1: #means put a new obs
                id = selfw.id_max+1
        try:
            d = selfw.Obs_dict[str(id)]["mem"][str(x)]
        except:
            try:
                f = selfw.Obs_dict[str(id)]["mem"]
            except:
                selfw.Obs_dict[str(id)]["mem"] = {}
                f = selfw.Obs_dict[str(id)]["mem"]
            selfw.Obs_dict[str(id)]["mem"][str(x)]=[]
            d = selfw.Obs_dict[str(id)]["mem"][str(x)]
        if d == []:
            O = 0
            for i in range(1,selfw.Nrep+1):
                d.append(selfw.Obs[str(id)](x,i))
                O += (d[i-1]-selfw.Obs_mean(x,Obs,id))**2
            return np.sqrt(O/selfw.Nrep)
        else:
            O = 0
            for i in range(1,selfw.Nrep+1):
                O += (d[i-1]-selfw.Obs_mean(x,Obs,id))**2
            return np.sqrt(O/selfw.Nrep)
    
    def Obs_mean_new(selfw,x:float,Obs:Callable[[float,int],float],id:int=-1)->float:
        if id == -1: #means put a new obs
                id = selfw.id_max+1
        try:
            return selfw.Obs_new_mean_dict[str(id)][str(x)]
        except:
            try:
                f = selfw.Obs_new_mean_dict[str(id)]
            except:
                selfw.Obs_new_mean_dict[str(id)] = {}
                f = selfw.Obs_new_mean_dict[str(id)]
            try:
                d = selfw.Obs_dict[str(id)]["mem"][str(x)]
            except:
                try:
                    g = selfw.Obs_dict[str(id)]["mem"]
                except:
                    selfw.Obs_dict[str(id)]["mem"] = {}
                    g = selfw.Obs_dict[str(id)]["mem"]
                g[str(x)]=[]
                d = g[str(x)]
            O = 0
            if d == []:
                for (i,omega) in enumerate(selfw.Omega_list):
                    d.append(selfw.Obs[str(id)](x,i))
                    O += omega*d[i]
            else:
                for (i,omega) in enumerate(selfw.Omega_list):
                    O += omega*d[i]
            O = O/selfw.Nrep
            f[str(x)] = O
            return O

    def delta_Obs_new(selfw,x:float,Obs:Callable[[float,int],float],id:int=-1)->float:
        if id == -1: #means put a new obs
                id = selfw.id_max+1
        try:
            d = selfw.Obs_dict[str(id)]["mem"][str(x)]
        except:
            try:
                f = selfw.Obs_dict[str(id)]["mem"]
            except:
                selfw.Obs_dict[str(id)]["mem"] = {}
                f = selfw.Obs_dict[str(id)]["mem"]
            selfw.Obs_dict[str(id)]["mem"][str(x)]=[]
            d = selfw.Obs_dict[str(id)]["mem"][str(x)]
        if d == []:
            O = 0
            for (i,omega) in enumerate(selfw.Omega_list):
                d.append(selfw.Obs[str(id)](x,i))
                O += omega*(d[i]-selfw.Obs_mean_new(x,Obs,id))**2
            return np.sqrt(O/selfw.Nrep)
        else:
            O = 0
            for (i,omega) in enumerate(selfw.Omega_list):
                O += omega*(d[i]-selfw.Obs_mean_new(x,Obs,id))**2
            return np.sqrt(O/selfw.Nrep)
    
    def Neff(selfw):
        r = 0
        for omega in selfw.Omega_list:
            if omega == 0:  # permet de skip la division par 0 
                continue
            r+= omega*np.log(selfw.Nrep/omega)
        return np.exp(r/selfw.Nrep)
    
    def reweighting_info(selfw):
        print(50*"=")
        print(f"{'Reweighting Info':^50}")
        print(50*"=")
        N_eff = selfw.Neff()
        print(f"Nrep={selfw.Nrep:<15} Neff={N_eff:<15}")
        print(50*'-')

    def plot_obs(selfw,obs_name,Obs:Callable[[float,int],float],id:int=-1,member:bool = False):
        if id == -1: #means put a new obs
                id = np.copy(selfw.id_max)+1 #copy mandatory otherwise it will increase dynamically for the vectors bellow
        X = selfw.x_value
        Vold = np.array([selfw.Obs_mean(x,Obs,id) for x in X])
        dVold = np.array([selfw.delta_Obs(x,Obs,id) for x in X])
        Vnew = np.array([selfw.Obs_mean_new(x,Obs,id) for x in X])
        dVnew = np.array([selfw.delta_Obs_new(x,Obs,id) for x in X])
        if member:
            for i in range(1,selfw.Nrep+1):
                Y = [selfw.Obs_dict[str(id)]['mem'][str(x)][i-1] for x in X]
                plt.plot(X,Y,'--',color='purple',linewidth=0.5,alpha= 0.8)
        plt.plot(X,Vold,color='red',label = 'old')
        plt.plot(X,Vnew,color = 'blue',label= 'new')
        plt.fill_between(X,Vold-dVold,Vold+dVold,color = 'red',alpha= 0.3)
        plt.fill_between(X,Vnew-dVnew,Vnew+dVnew,color = 'blue',alpha= 0.3)
        plt.errorbar(X,selfw.Data,yerr=selfw.x_unc,fmt='+',color = 'black',label ='Data')
        plt.xlabel(selfw.x_type)
        plt.ylabel(obs_name)
        plt.legend()
        plt.savefig(os.path.join(plots_dir,'test_reweighting.pdf'),bbox_inches="tight")
        plt.show()

    def plot_obs_3D(selfw,obs_name,Obs:Callable[[float,int],float],id:int=-1):
        ax = plt.figure().add_subplot(projection='3d')

        # Plot a sin curve using the x and y axes.
        if id == -1: #means put a new obs
                id = np.copy(selfw.id_max)+1
        X = selfw.x_value
        Z = np.zeros(selfw.Ndata)
        Vold = np.array([selfw.Obs_mean(x,Obs,id) for x in X])
        dVold = np.array([selfw.delta_Obs(x,Obs,id) for x in X])
        Vnew = np.array([selfw.Obs_mean_new(x,Obs,id) for x in X])
        dVnew = np.array([selfw.delta_Obs_new(x,Obs,id) for x in X])

        # ax.plot(X,Vold, zs=0,color='red', zdir='z', label='Old')
        ax.fill_between(X, Vold-dVold,0,X,Vold+dVold,0,color='red',alpha=0.3, label='Old')
        # ax.plot(X,Vnew, zs=0,color='blue', zdir='z', label='new')
        ax.fill_between(X, Vnew-dVnew,0,X,Vnew+dVnew,0,color='blue',alpha=0.3, label='new')
        # ax.errorbar(X,selfw.Data,Z,fmt='+',color ='black',label = 'Data')
        # Plot scatterplot data (20 2D points per colour) on the x and z axes.

        cmap = cm.plasma
        m,M = min(selfw.Omega_list),max(selfw.Omega_list)
        norm = mcolors.Normalize(vmin=m, vmax=M)

        for i,omega in enumerate(selfw.Omega_list):
            Obsi = [selfw.Obs_dict[str(id)]['mem'][str(x)][i] for x in X]
            col = cmap(norm(omega))
            ax.plot(X, Obsi, zs=omega,color=col,zdir='z',alpha = 0.3)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # cbar = plt.colorbar(sm, ax=ax)

        # Make legend, set axes limits and labels
        ax.legend()
        ax.set_xlim(min(selfw.x_value), max(selfw.x_value))
        ax.set_ylim(0.4, 1.1)
        ax.set_zlim(0, M+1)
        ax.set_xlabel(selfw.x_type)
        ax.set_ylabel(obs_name)
        ax.set_zlabel(r'$\omega$')

        # Customize the view angle so it's easier to see that the scatter points lie
        # on the plane y=0
        ax.view_init(elev=20., azim=-15, roll=0)
        plt.savefig(os.path.join(plots_dir,'test_reweighting_3D.pdf'),bbox_inches="tight")
        plt.show()