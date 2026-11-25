# -*- coding: utf-8 -*-

# =============================================================================
# Code to compute J/Psi cross sections 
# for testing the Color Evaporation Model (CEM)
# Credits : BOURGEAIS Djessy; ARLEO Francois.
# Subatech (IN2P3,CNRS)
# Nantes, France
# Last modified: 4/11/2026 (x_F FCEL for E866 and E789 fits)
# =============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))

import matplotlib.pyplot as plt
import lhapdf as lha
import numpy as np
import numpy.typing as npt 
from scipy import (integrate,differentiate)
from scipy.interpolate import PchipInterpolator
try :
	from . import Probability as prob
except ImportError:
	print('Caution! : running from src directory, using local import')
	import Probability as prob


# Ininitalisation of a patricle dictionary
particle = {
	"d": {"id": 1, "charge": -1./3}, 
	"u": {"id": 2, "charge": 2./3}, 
	"s": {"id": 3, "charge": -1./3}, 
	"c": {"id": 4, "charge": 2./3}, 
	"b": {"id": 5, "charge": -1./3}, 
	"t": {"id": 6, "charge": 2./3},												# Quarks

	"dbar": {"id": -1, "charge": 1./3}, 
	"ubar": {"id": -2, "charge": -2./3}, 
	"sbar": {"id": -3, "charge": 1./3}, 
	"cbar": {"id": -4, "charge": -2./3}, 
	"bbar": {"id": -5, "charge": 1./3}, 
	"tbar": {"id": -6, "charge": -2./3},										# Antiquarks

	"g": {"id": 21, "charge": 0},												# Gluon
	"gamma": {"id": 22, "charge": 0},											# Photon
	"Z0": {"id": 23, "charge": 0},												# Z boson
	"W+": {"id": 24, "charge": 1},
	"W-": {"id": -24, "charge": -1},											# W bosons 
	"h0": {"id": 25, "charge": 0},												# Higgs boson
}

Atom ={ # Add little by little the nuclei needed
	'Pb': {'Z':82 ,'A':208,'ylim':6} ,
	'Au': {'Z':79,'A':197,'ylim':3},
	'W': {'Z':74,'A':184,'ylim':3},											# E866 heavy target
	'Be': {'Z':4,'A':9,'ylim':3},												# E866 light target
	'Pt':{'Z':78,'A':195,'ylim':3}
}

Z_to_ylim = {
	82:6,
	79:4
}

Z_to_p_Tlim ={
	82:(10,80),
	79:(2,6)
}

id_to_charge = {v["id"]: v["charge"] for v in particle.values()}

def charge(particle_id):
	'''return the charge of a particle giving its id'''
	return(id_to_charge.get(particle_id,None))

# Variables initiallisation
N_c = 3. 							  											# number of colors 
C_F = (N_c**2 -1)/(2*N_c) 														# color factor  
alpha = 1./137.																	# alpha EM
conv_fact = pow(0.197,2)*pow(10,10)												# to convert Gev^-2 into barns
pi = np.pi

# Integration parameters
epsilon = pow(10,-10)															# to avoid phase space borders 
N_y= 41
N_x_F = 41
N_limit = 1000																	# number of subdivisions for quad integration

# Extern functions
def x1_y(y:float,m:float,rs:float)->float:
	"""x1 rapidity dependent

	Args:
		y (float): rapidity
		m (float): charmonium mass in GeV
		rs (float): s (Mandelstamm) root in GeV

	Returns:
		float
	"""
	return np.exp(y)*m/rs

def x2_y(y:float,m:float,rs:float)->float:
	"""x2 rapidity dependent

	Args:
		y (float): rapidity
		m (float): charmonium mass in GeV
		rs (float): s (Mandelstamm) root in GeV

	Returns:
		float
	"""
	return np.exp(-y)*m/rs

def x1_ym2(y:float,m2:float,rs:float)->float:
	"""x1 rapidity dependent

	Args:
		y (float): rapidity
		m2 (float): charmonium mass squared in GeV^2
		rs (float): s (Mandelstamm) root in GeV

	Returns:
		float
	"""
	return np.exp(y)*np.sqrt(m2)/rs

def x2_ym2(y:float,m2:float,rs:float)->float:
	"""x2 rapidity dependent

	Args:
		y (float): rapidity
		m2 (float): charmonium mass squared in GeV^2
		rs (float): s (Mandelstamm) root in GeV

	Returns:
		float
	"""
	return np.exp(-y)*np.sqrt(m2)/rs

def x1_xF(xF:float,m:float,rs:float)->float:
	"""x1 xF dependent

	Args:
		xF (float): Feynman's x variable
		m (float): charmonium mass in GeV
		rs (float): s (Mandelstamm) root in GeV

	Returns:
		float:
	"""
	return (np.sqrt(xF**2+(2*m/rs)**2)+xF)/2.

def x2_xF(xF:float,m:float,rs:float)->float:
	"""x2 xF dependent

	Args:
		xF (float): Feynman's x variable
		m (float): charmonium mass in GeV
		rs (float): s (Mandelstamm) root in GeV

	Returns:
		float:
	"""
	return (np.sqrt(xF**2+(2*m/rs)**2)-xF)/2.

def xprime_xF(xF:float,m:float,rs:float)->float:
	"""x' xF dependent

	Args:
		xF (float): Feynman's x variable
		m (float): charmonium mass in GeV
		rs (float): s (Mandelstamm) root in GeV

	Returns:
		float:
	"""
	return x1_xF(xF,m,rs)+x2_xF(xF,m,rs)

def x1_xFm2(xF:float,m2:float,rs:float)->float:
	"""x1 xF dependent

	Args:
		xF (float): Feynman's x variable
		m2 (float): charmonium mass squared in GeV^2
		rs (float): s (Mandelstamm) root in GeV

	Returns:
		float:
	"""
	return (np.sqrt(xF**2+4*m2/(rs**2))+xF)/2.

def x2_xFm2(xF:float,m2:float,rs:float)->float:
	"""x2 xF dependent

	Args:
		xF (float): Feynman's x variable
		m2 (float): charmonium mass squared in GeV^2
		rs (float): s (Mandelstamm) root in GeV

	Returns:
		float:
	"""
	return (np.sqrt(xF**2+4*m2/(rs**2))-xF)/2.

m_c , m_D = 1.2, 1.864 # GeV, from PDG 2024
m_min, m_max = 2*m_c, 2*m_D
# m_jpsi = 3.0969 #GeV, from PDG 2024
m_jpsi = 3.0 #GeV

def mu_R(k,pt): #replace k by the mu or mu_f factor (see with F its implementation)
	return lambda m: k*2*m_c 

def mu_F(k,pt): #replace k by the mu or mu_f factor (see with F its implementation)
	return lambda m: k*m 

def Y_list(rs:float,p_T:float=0)-> npt.NDArray[np.float64]:
	'''Return a numpy linespace array of N_y rapidities given x_T'''
	y_max = min(6,np.log(rs)-np.log(m_max**2+p_T**2)/2)
	y_min = -1*y_max
	Y = np.linspace(y_min,y_max,N_y)
	return Y

def X_F_list(rs:float,p_T:float=0)->npt.NDArray[np.float64]:
	'''Return a numpy linespace array of N_x_F points '''
	x_F_max = 1-(np.sqrt(m_max**2+p_T**2)/rs)**2
	x_F_min = -1*x_F_max
	X_F = np.linspace(x_F_min,x_F_max,N_x_F)
	# X_F = np.linspace(-0.1,0.1,N_x_F)
	return X_F

def min_max(ys:list)->tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
	'''return the min list and max list of a list of lists'''
	ys_array = np.vstack(ys)
	y_min = np.min(ys_array, axis=0)
	y_max = np.max(ys_array, axis=0)
	return (y_min,y_max)

def alpha_from_R(R_heavy,R_light,A_heavy,A_light):
	'''Nuclear dependence exponent alpha, sigma_pA ~ A^alpha, from the per-nucleon ratios (heavy vs light target):
	alpha = 1 + ln(R_heavy/R_light)/ln(A_heavy/A_light)   (E866 convention, W/Be)'''
	return 1.+np.log(R_heavy/R_light)/np.log(A_heavy/A_light)

# Conveniance objects

def Collision(string:str)->bool:
	'''Return True(bool) if the argument is "pp", False if it is "pA"'''
	if string == 'pp':
		return True
	elif (string == 'pA') or (string == 'Ap') :
		return False
	else :
		raise ValueError("The argument must be 'pp' or 'pA'")
	
def Var_choice(string:str)->bool:
	'''Return True(bool) if the argument is "y", False if it is "xf/F" or "x_f/F"'''
	if string == 'y':
		return True
	elif (string == 'xf') or (string == 'xF')or (string == 'x_f')or (string == 'x_F') :
		return False
	else :
		raise ValueError("The argument must be 'y' or 'xf'")

class SigmaJpsi:
	def __init__(self, p_set,p_list, A_set,A_list,s:float, Z:int, A:int,p_stand_set = 0):
		"""
		To initialize Sigma object
		
		:param p_set: Proton PDF set
		:type p_set: lhapdf.getPDFSet
		:param A_set: Nuclei nPDF set 
		:type A_set: lhapdf.getPDFSet
		:param s: Mandelstam center of mass energy squared
		:type s: float
		:param Z: Atomic number of the nuclei
		:type Z: int
		:param A: Mass number of the nuclei
		:type A: int
		:p_stand_set: Standard proton PDF set
		:type p_stand_set: deflaut 0 else lhapdf.getPDFSet

		"""
		if p_stand_set == 0:
			self.is_p_stand = False
		else:
			self.is_p_stand = True # tell if there is a standard proton pdf set (in the case of comparing btw different nuclei sets)
			self.p_stand_set = p_stand_set
			self.p_stand_list = self.p_stand_set.mkPDFs()
			self.current_p_stand = self.p_stand_list[0]
			self.current_p_stand_num = 0 
		self.p_set = p_set								# set of pdf for the projectile (with the pdf name as a string) 
		self.A_set = A_set		 						# same for target
		self.s = float(s) 														# center of mass energy
		self.rs =np.sqrt(self.s)
		self.Z = float(Z) 														# the number of proton of the target element
		self.A = float(A)														# the number of nucleus of the target element
		self.B = 1 																# the number of nucelus of the proton 
		self.p_list = p_list										# create the list of all members for the pdf set
		self.A_list = A_list
		self.current_p = self.p_list[0]											# Initialise the pdf member to 0
		self.current_A = self.A_list[0] 
		self.current_p_num = 0
		self.current_A_num = 0
		self.M_jpsi_T = lambda p_T: np.sqrt(m_jpsi**2+p_T**2)
		self.m_T_min = lambda p_T: np.sqrt(m_min**2+p_T**2)
		self.m_T_max = lambda y,p_T: min([np.sqrt(m_max**2+p_T**2),self.rs*np.exp(y),self.rs*np.exp(-y)])
		self.m_T_max_xF = lambda xF,p_T: min([np.sqrt(m_max**2+p_T**2),self.rs*np.sqrt(max(1.-abs(xF),0.))])	# x1 (or x2) <= 1 at fixed x_F
		
	### init functions ###
	
	def pdf_p(self ,num: int)->lha.PDF: # for some reasons i can't specify the output type, need more research on it
		"""Redefine the current pdf of the projectile (proton) for a certain 
		member number of the set (if it is different from the current one) 
		and return the new current proton pdf set

		Args:
			num (int): member number of the PDF set

		Raises:
			ValueError: num should be between 0 and the maximum number of members in the set

		Returns:
			lha.PDF:
		"""
		Num = self.current_p_num
		size = self.p_set.size
		if num < 0 or num > size:
			raise ValueError("num should be between 0 and "+str(size))
		if num != Num:
			p = self.p_list														# verify that the member belongs to the list
			self.current_p_num = num
			self.current_p = p[num]
		return self.current_p

	def pdf_p_stand(self ,num: int)->lha.PDF: # for some reasons i can't specify the output type, need more research on it
			"""Redefine the current pdf standard of the projectile (proton) for a certain 
			member number of the set (if it is different from the current one) 
			and return the new current proton pdf set
	
			Args:
				num (int): member number of the PDF set
	
			Raises:
				ValueError: num should be between 0 and the maximum number of members in the set
				TypeError : p_stand_set is not type lhapdf.getPDFset
	
			Returns:
				lha.PDF:
			"""
			if self.is_p_stand:
				Num = self.current_p_stand_num
				size = self.p_stand_set.size
				if num < 0 or num > size:
					raise ValueError("num should be between 0 and "+str(size))
				if num != Num:
					p = self.p_stand_list														# verify that the member belongs to the list
					self.current_p_stand_num = num
					self.current_p_stand = p[num]
				return self.current_p_stand
			else:
				raise TypeError(f"p_stand_set is not type lhapdf.getPDFset but type {type(self.p_stand_set)}")

	def pdf_A(self,num : int)->lha.PDF:
		"""
		Redefine the current pdf of the target (nuclei) for a certain member
		number of the set (if it is different from the current one) 
		and return the new current nuclei pdf set
		
		:param num: Member number of the nPDF set
		:type num: int
		:return: the nPDF member associated with 'num' 
		:rtype: lha.PDF
		"""
		Num = self.current_A_num
		# size = self.A_set.size
		size = len(self.A_list)
		if num < 0 or num > size:
			raise ValueError("num should be between 0 and "+str(size))
		if num != Num:
			A = self.A_list 													# verify that the member belongs to the list 
			self.current_A_num = num
			self.current_A = A[num]
		return self.current_A

		
	def alpha_s_A(self,num:int,mu2:float)->float:
		"""
		Return the strong coulping constant for the nuclei nPDF
		
		:param num: Member number of the nPDF set
		:type num: int
		:param mu2: Energy scale squared
		:type mu2: float
		:return: 
		:rtype: float
		"""
		# solve here the pb with hessain to bayesian list
		A_num = self.init_A_list[num]
		return A_num.alphasQ2(mu2)
	
	def alpha_s_p(self,num:int,mu2:float)->float:
		"""
		Return the strong coulping constant for the proton PDF
		
		:param num: Member number of the PDF set
		:type num: int
		:param mu2: Energy scale squared
		:type mu2: float
		:return: 
		:rtype: float
		"""
		'''return alpha_s for the proton at mu2'''
		p_num = self.pdf_p(num)
		return p_num.alphasQ2(mu2)

	def alpha_s_p_stand(self,num:int,mu2:float)->float:
		"""
		Return the strong coulping constant for the proton PDF
		
		:param num: Member number of the PDF set
		:type num: int
		:param mu2: Energy scale squared
		:type mu2: float
		:return: 
		:rtype: float
		"""
		'''return alpha_s for the proton at mu2'''
		p_num = self.pdf_p_stand(num)
		return p_num.alphasQ2(mu2)
		
	def n_xfxQ2(self,particle_id:int,x:float,mu_f2:float,num:int,std:bool = False)->float:
		"""
		Return the value of the neutron PDF (x*f(x)) for a given particle id, x and energy scale
		
		:param particle_id: Id of the particle of interest (e.g. d = 1, d_bar = -1)
		:type particle_id: int
		:param x: Momentum fraction
		:type x: float
		:param mu_f2: Energy scale scared (pdf)
		:type mu_f2: float
		:param num: Member number of the PDF set
		:type num: int
		:param std: boolean to say if the neutron pdf has to be made from the standard proton one
		:type std: bool default False
		:return:
		:rtype: float
		"""
		if std:
			p = self.pdf_p_stand(num)
			if np.abs(particle_id) >= 3:
				return p.xfxQ2(particle_id,x,mu_f2)
			else :
				return p.xfxQ2(2//particle_id,x,mu_f2)								# Math trick using euclide division to make (-)1 ->(-)2, vice versa
		else:
			p = self.pdf_p(num)
			if np.abs(particle_id) >= 3:
				return p.xfxQ2(particle_id,x,mu_f2)
			else :
				return p.xfxQ2(2//particle_id,x,mu_f2)
	
	def F2_p(self,x:float,mu_f2:float,num:int,iso:str = 'p',n_f:int=3,std:bool=False)->float:
		"""

		Gives back the F2 function of the proton for an x and a mu_f2 given
		
		:param x: Momentum fraction
		:type x: float
		:param mu_f2: Energy scale squared (pdf)
		:type mu_f2: float
		:param num: Member number of the PDF set 
		:type num: int
		:param iso: 'p' by default. Isospin equivalent of F2 for the neutron with 'n'
		:type iso: str
		:param n_f: 3 by default. Number of quark flavour
		:type n_f: int
		:param std: boolean to say if the results has to be made from the standard proton one
		:type std: bool default False
		:return:
		:rtype: float
		"""
		F = 0.
		if iso == 'p':
			if std:
				p = self.pdf_p_stand(num)
			else:
				p = self.pdf_p(num)
			for i in range(1,n_f+1):
				F += (charge(i)**2)*(p.xfxQ2(i,x,mu_f2)+p.xfxQ2(-i,x,mu_f2))
		elif iso == 'n':
			for i in range(1,n_f+1):
				F += (charge(i)**2)*(self.n_xfxQ2(i,x,mu_f2,num,std)+self.n_xfxQ2(-i,x,mu_f2,num,std))
		return F/x

	def A_std(self,num_std:int,num_A:int,i:int,x:float,mu_f2:float):
		A = self.pdf_A(num_A)
		A_pdf = A.xfxQ2(i,x,mu_f2)
		p = self.pdf_p(0) 
		p_pdf = p.xfxQ2(i,x,mu_f2)
		p_std = self.pdf_p_stand(num_std)
		p_std_pdf = p_std.xfxQ2(i,x,mu_f2)
		return p_std_pdf*A_pdf/p_pdf

	def F2_A(self,x:float,mu_f2:float,num:int,n_f:int=3,std:bool=False)->float:
		"""
		Gives back the F2 function of the nuclei for an x and a mu_f2 given
		
		:param x: Momentum fraction
		:type x: float
		:param mu_f2: Energy scale squared (pdf)
		:type mu_f2: float
		:param num: Member number of the nPDF set 
		:type num: int
		:param n_f: 3 by default. Number of quark flavour
		:type n_f: int
		:return:
		:rtype: float
		"""
		F = 0.
		if not(std):
			A = self.pdf_A(num)
			for i in range(1,n_f+1):
				F += (charge(i)**2)*(A.xfxQ2(i,x,mu_f2)+A.xfxQ2(-i,x,mu_f2))
		else:
			for i in range(1,n_f+1):
				F += (charge(i)**2)*(self.A_std(0,num,i,x,mu_f2)+self.A_std(0,num,-i,x,mu_f2))
		return F/x
	
	def F_ij(self,xa:float,xb:float,mu_f2:float,num:int, direction:str = 'qqbar',iso:str='p',n_f:int=3,is_pp:bool = False)->float:
		# Don't need here to comput with standard proton bc it's not part of the cross section
		"""
		Return the F_ij function that appears in the annihilation process
		
		:param xa: Momentum fraction of the projectile (proton) parton
		:type xa: float
		:param xb: Momentum fraction of the target (nuclei/neutron/proton) parton
		:type xb: float
		:param mu_f2: Energy scale squared (pdf)
		:type mu_f2: float
		:param num: Member number of the pdf set
		:type num: int
		:param direction: 'qqbar' by default. Can be 'qbarq' to describe the other initial state
		:type direction: str
		:param iso: 'p' by default. Isospin equivalent of F2 for the neutron with 'n'
		:type iso: str
		:param n_f: 3 by default. Number of quark flavors
		:type n_f: int
		:param is_pp: False by default. Be True computed proton/proton(neutron if iso = 'n' )
		:type is_pp: bool
		:return:
		:rtype: float
		"""
		F = 0.
		p = self.pdf_p(num)
		A = self.pdf_A(num)
		if iso == 'n':
			if direction == 'qqbar':
				for i in range(1,n_f+1):
					val1 = p.xfxQ2(i, xa, mu_f2)
					val2 = self.n_xfxQ2(-i, xb, mu_f2,num)
					if val1 is None or val2 is None:
						raise ValueError("xfxQ2 returned None for i={}, xa={}, xb={}, mu_f2={}".format(i, xa, xb, mu_f2))
					F += (charge(i)**2) * val1 * val2
				if xa == 0 or xb == 0:
					raise ValueError("xa or xb is zero, leading to division by zero.")
				return F/(xa * xb)
			elif direction == 'qbarq':
				for i in range(1,n_f+1):
					val1 = p.xfxQ2(-i, xa, mu_f2)
					val2 = self.n_xfxQ2(i, xb, mu_f2,num)
					if val1 is None or val2 is None:
						raise ValueError("xfxQ2 returned None for i={}, xa={}, xb={}, mu_f2={}".format(i, xa, xb, mu_f2))
					F += (charge(i)**2) * val1 * val2
				if xa == 0 or xb == 0:
					raise ValueError("xa or xb is zero, leading to division by zero.")
				return F/(xa * xb)
		elif iso == 'p':
			if not is_pp:
				if direction == 'qqbar':
					for i in range(1,n_f+1):
						val1 = p.xfxQ2(i, xa, mu_f2)
						val2 = A.xfxQ2(-i, xb, mu_f2)
						if val1 is None or val2 is None:
							raise ValueError("xfxQ2 returned None for i={}, xa={}, xb={}, mu_f2={}".format(i, xa, xb, mu_f2))
						F += (charge(i)**2) * val1 * val2
				elif direction == 'qbarq':
					for i in range(1,n_f+1):
						val1 = p.xfxQ2(-i, xa, mu_f2)
						val2 = A.xfxQ2(i, xb, mu_f2)
						if val1 is None or val2 is None:
							raise ValueError("xfxQ2 returned None for i={}, xa={}, xb={}, mu_f2={}".format(i, xa, xb, mu_f2))
						F += (charge(i)**2) * val1 * val2
				if xa == 0 or xb == 0:
					raise ValueError("xa or xb is zero, leading to division by zero.")
				return F/(xa * xb) 
			elif is_pp:
				A = self.pdf_p(num)
				if direction == 'qqbar':
					for i in range(1,n_f+1):
						val1 = p.xfxQ2(i, xa, mu_f2)
						val2 = A.xfxQ2(-i, xb, mu_f2)
						if val1 is None or val2 is None:
							raise ValueError("xfxQ2 returned None for i={}, xa={}, xb={}, mu_f2={}".format(i, xa, xb, mu_f2))
						F += (charge(i)**2) * val1 * val2
				elif direction == 'qbarq':
					for i in range(1,n_f+1):
						val1 = p.xfxQ2(-i, xa, mu_f2)
						val2 = A.xfxQ2(i, xb, mu_f2)
						if val1 is None or val2 is None:
							raise ValueError("xfxQ2 returned None for i={}, xa={}, xb={}, mu_f2={}".format(i, xa, xb, mu_f2))
						F += (charge(i)**2) * val1 * val2
				if xa == 0 or xb == 0:
					raise ValueError("xa or xb is zero, leading to division by zero.")
				return F/(xa * xb)
		else:
			raise ValueError("Iso type '{}' is not recognized. Expected 'p' or 'n'.".format(iso))
		
	def F_qqbar(self,xa:float,xb:float,mu_f2:float,num:int,iso:str='p',n_f:int=3,is_pp:bool=False,std:bool=False)->float:
		"""Return the sum of pdf functions that appears in front of the annihilation process

		Args:
			xa (float): _projectile fraction momentum_
			xb (float): _target fraction momentum_
			mu_f2 (float): _energy scale squared (pdf)_
			num (int): _member of the pdf set_
			iso (str, optional): _isospoin variable_. Defaults to 'p'.
			n_f (int, optional): _light flavour number_. Defaults to 3.
			is_pp (bool, optional): _argument for choosing pp or pA collision_. Defaults to False.

		Raises:
			ValueError: _xfxQ2 returned None_
			ValueError: _xa or xb is zero, leading to division by zero_
			ValueError: _Iso type is not recognized. Expected 'p' or 'n'_

		Returns:
			float: 
		"""
		F = 0.
		if is_pp:
			p = self.pdf_p_stand(num) if std else self.pdf_p(num)
			q_A=lambda i: p.xfxQ2(i, xb, mu_f2) # is pp collision, we keep the same PDF
			qbar_A = lambda i: p.xfxQ2(-i, xb, mu_f2)
		elif std:
			p = self.pdf_p_stand(0) # if standard, we change in consequences
			q_A= lambda i:self.A_std(0,num,i,xb,mu_f2)
			qbar_A =lambda i: self.A_std(0,num,-i,xb,mu_f2)
		else:
			p = self.pdf_p(0) # Else it's the regualar proton and nuclei PDFs 
			A = self.pdf_A(num)
			q_A= lambda i: A.xfxQ2(i, xb, mu_f2)
			qbar_A =lambda i: A.xfxQ2(-i, xb, mu_f2)
		if iso == 'n':
			for i in range(1,n_f+1):
				val1 = p.xfxQ2(i, xa, mu_f2)*self.n_xfxQ2(-i, xb, mu_f2,num,std)
				val2 = p.xfxQ2(-i, xa, mu_f2)*self.n_xfxQ2(i, xb, mu_f2,num,std)
				if val1 is None or val2 is None:
					raise ValueError("xfxQ2 returned None for i={}, xa={}, xb={}, mu_f2={}".format(i, xa, xb, mu_f2))
				F +=  val1 + val2
			if xa == 0 or xb == 0:
				raise ValueError("xa or xb is zero, leading to division by zero.")
			return F/(xa * xb)
		elif iso == 'p':
			for i in range(1,n_f+1):
				val1 = p.xfxQ2(i, xa, mu_f2)*qbar_A(i)
				val2 = p.xfxQ2(-i, xa, mu_f2)*q_A(i)
				if val1 is None or val2 is None:
					raise ValueError("xfxQ2 returned None for i={}, xa={}, xb={}, mu_f2={}".format(i, xa, xb, mu_f2))
				F += val1 + val2
			if xa == 0 or xb == 0:
				raise ValueError("xa or xb is zero, leading to division by zero.")
			return F/(xa * xb) 
		else:
			raise ValueError("Iso type '{}' is not recognized. Expected 'p' or 'n'.".format(iso))
		
	def Gluon_p(self,x,mu_f2,num,std:bool=False):
		'''Return the gluon pdf of the proton'''
		if std:
			p = self.pdf_p_stand(num)
		else: 
			p = self.pdf_p(num)
		return p.xfxQ2(particle["g"]["id"],x,mu_f2)/x
	
	def Gluon_A(self,x,mu_f2,num,std:bool=False):
		'''Return the gluon pdf of the nuclei'''
		if std:
			return self.A_std(0,num,particle["g"]["id"],x,mu_f2)
		else:
			A = self.pdf_A(num)
			return A.xfxQ2(particle["g"]["id"],x,mu_f2)/x
	
	### Cross sections ### (checked, they're good)

	def Lambda(self,m:float)->float:
		"""Lambda variable from R.Vogt papers on CEM LO

		Args:
			m (float): _ccbar mass_

		Returns:
			float:
		"""
		M= m_c/m
		return np.sqrt(1-4*(M**2))
	
	def gg(self,m:float,num:int,mu_factor:float=1,p_T:float= 1,std:bool=False)->float:
		"""Returns $\sigma_gg(m^2)$

		Args:
			m (float): _ccbar mass_ in GeV
			num (int): _member of the pdf set_
			mu_factor (float, optional): _energy scale factor used as: mu^2 = (mu_factor*m)^2_. Defaults to 1.
			p_T (float, optional): _transverse momentum of the produced ccbar pair in GeV_. Defaults to 1.

		Returns:
			float:
		"""
		mu = mu_R(mu_factor,p_T)
		if std:
			alpha_s = self.alpha_s_p_stand(num,mu(m)**2)
		else:
			alpha_s = self.alpha_s_p(num,mu(m)**2)
		# print("alpha_s ={}".format(alpha_s))
		L = self.Lambda(m)
		M = m_c/m
		return pi*(alpha_s**2)*((1+4*(M**2)+M**4)*np.log((1+L)/(1-L))-(7+31*(M**2))*L/4)/(3*(m**2))
	
	def qqbar(self,m:float,num:int,mu_factor:float=1,p_T:float= 1,std:bool=False)->float:
		"""Returns $\sigma_qqbar(m^2)$

		Args:
			m (float): _ccbar mass_
			num (int): _member of the pdf set_
			mu_factor (float, optional): _energy scale factor used as: mu^2 = (mu_factor*m)^2_. Defaults to 1.
			p_T (float, optional): _transverse momentum of the produced ccbar pair in GeV_. Defaults to 1.

		Returns:
			float:
		"""
		mu = mu_R(mu_factor,p_T)
		if std:
			alpha_s = self.alpha_s_p_stand(num,mu(m)**2)
		else:
			alpha_s = self.alpha_s_p(num,mu(m)**2)
		L = self.Lambda(m)
		M = m_c/m
		return 8*pi*(alpha_s**2)*L*(1+2*(M**2))/(27*(m**2))
	
	### Integrands ### (checked, good too)

	def x1_x2_Jac(self,var:float,m:float,Var_y:bool=True)->float:
		"""Returns [x_1,x_2,Jacobian] dependant on the choice of the variables (y or x_F)

		Args:
			var (float): _rapidity or Feynman's x_
			m (float): _ccbar mass_
			Var_y (bool, optional): _if True, work in rapidity. Else, work in x_F_. Defaults to True.

		Returns:
			float:
		"""
		rs = self.rs
		s = self.s
		if Var_y:
			x1 = x1_y(var,m,rs)
			x2 = x2_y(var,m,rs)
			Jac = 1/s
		else:
			x1 = x1_xF(var,m,rs)
			x2 = x2_xF(var,m,rs)
			Jac = 1/(np.sqrt((var*s)**2+4*(m**2)*s))
		return [x1,x2,Jac]

	def integrand_gg(self,var:float,m:float,num:int,mu_factor:float=1,mu_f_factor:float=1,p_T:float= 1,n_f:int=3,is_pp:bool=False,Var_y:bool = True,std:bool=False)->float:
		"""Returns the integrand of the gg part, in the form 2m*dsigma/dvardm2

		Args:
			var (float): _description_
			m (float): _description_
			num (int): _description_
			mu_factor (float, optional): _description_. Defaults to 1.
			mu_f_factor (float, optional): _description_. Defaults to 1.
			p_T (float, optional): _transverse momentum of the produced ccbar pair in GeV_. Defaults to 1.
			n_f (int, optional): _description_. Defaults to 3.
			is_pp (bool, optional): _description_. Defaults to False.
			Var_y (bool, optional): _description_. Defaults to True.

		Returns:
			float: _description_
		"""
		s = self.s
		rs = self.rs
		mu = mu_F(mu_factor,p_T)(m)
		x1,x2,Jac = self.x1_x2_Jac(var,m,Var_y=Var_y)
		if is_pp:
			pdf = self.Gluon_p(x1,mu**2,num,std)*self.Gluon_p(x2,mu**2,num,std)
			sig = self.gg(m,num,mu_factor,p_T,std)
		else:
			pdf = self.Gluon_p(x1,mu**2,0,std)*self.Gluon_A(x2,mu**2,num,std)
			sig = self.gg(m,0,mu_factor,p_T,std)
		dsig_dvardm2 = pdf*Jac*sig
		return 2*m*dsig_dvardm2
	
	def integrand_qqbar(self,var:float,m:float,num:int,mu_factor:float=1,mu_f_factor:float=1,p_T:float= 1,n_f:int=3,iso:str='p',is_pp:bool=False,Var_y:bool=True,std:bool=False):
		"""Returns the integrand of the qqbar part, in the form 2m*dsigma/dvardm2

		Args:
			var (float): _description_
			m (float): _description_
			num (int): _description_
			mu_factor (float, optional): _description_. Defaults to 1.
			mu_f_factor (float, optional): _description_. Defaults to 1.
			p_T (float, optional): _transverse momentum of the produced ccbar pair in GeV_. Defaults to 1.
			n_f (int, optional): _description_. Defaults to 3.
			iso (str, optional): _description_. Defaults to 'p'.
			is_pp (bool, optional): _description_. Defaults to False.
			Var_y (bool, optional): _description_. Defaults to True.

		Returns:
			_type_: _description_
		"""
		s = self.s
		rs = self.rs
		mu = mu_F(mu_factor,p_T)(m)
		x1,x2,Jac = self.x1_x2_Jac(var,m,Var_y=Var_y)
		if is_pp:
			pdf = self.F_qqbar(x1,x2,mu**2,num,iso,n_f,is_pp,std)
			sig = self.qqbar(m,num,mu_factor,p_T,std)
		else:
			pdf = self.F_qqbar(x1,x2,mu**2,num,iso,n_f,is_pp,std)
			sig = self.qqbar(m,0,mu_factor,p_T,std)
		dsig_dvardm2 = pdf*Jac*sig
		return 2*m*dsig_dvardm2
	
	def integrand_tot(self,var:float,m:float,num:int,mu_factor:float=1,mu_f_factor:float=1,p_T:float= 1,n_f:int=3,iso:str='p',is_pp:bool=False,Var_y:bool=True,std:bool=False):
		"""Returns the total integrand, in the form 2m*dsigma/dvardm2

		Args:
			var (float): _description_
			m (float): _description_
			num (int): _description_
			mu_factor (float, optional): _description_. Defaults to 1.
			mu_f_factor (float, optional): _description_. Defaults to 1.
			p_T (float, optional): _transverse momentum of the produced ccbar pair in GeV_. Defaults to 1.
			n_f (int, optional): _description_. Defaults to 3.
			iso (str, optional): _description_. Defaults to 'p'.
			is_pp (bool, optional): _description_. Defaults to False.
			Var_y (bool, optional): _description_. Defaults to True.

		Returns:
			_type_: _description_
		"""
		s = self.s
		rs = self.rs
		mu = mu_F(mu_factor,p_T)(m)
		x1,x2,Jac = self.x1_x2_Jac(var,m,Var_y=Var_y)
		if is_pp:
			pdf_gg = self.Gluon_p(x1,mu**2,num,std)*self.Gluon_p(x2,mu**2,num,std)
			pdf_qqbar = self.F_qqbar(x1,x2,mu**2,num,iso,n_f,is_pp,std)
			dsig_dvardm2 = Jac*(pdf_gg*self.gg(m,num,mu_factor,p_T,std)+pdf_qqbar*self.qqbar(m,num,mu_factor,p_T,std))
		else:
			pdf_gg = self.Gluon_p(x1,mu**2,0,std)*self.Gluon_A(x2,mu**2,num,std)
			pdf_qqbar = self.F_qqbar(x1,x2,mu**2,num,iso,n_f,is_pp,std)
			dsig_dvardm2 = Jac*(pdf_gg*self.gg(m,0,mu_factor,p_T,std)+pdf_qqbar*self.qqbar(m,0,mu_factor,p_T,std))
		return 2*m*dsig_dvardm2
	
	### Integration ###

	def _quad_mT(self,I,var,p_T,Var_y:bool=True):
		"""m_T integration of the integrand I, with the right upper bound for y or x_F (returns (0,0) if the phase space is empty)"""
		lo = self.m_T_min(p_T)
		hi = self.m_T_max(var,p_T) if Var_y else self.m_T_max_xF(var,p_T)
		if hi <= lo:
			return (0.,0.)
		return integrate.quad(I,lo,hi)

	def dsigma_gg_dvar(self,var,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,is_pp=False,Var_y=True,std:bool=False):
		I = lambda m_T :self.integrand_gg(var,m_T,num,mu_factor,mu_f_factor,p_T,n_f,is_pp,Var_y,std)
		sigma, err_sigma = self._quad_mT(I,var,p_T,Var_y)
		return (conv_fact*sigma,conv_fact*err_sigma)
	
	def dsigma_qqbar_dvar(self,var,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',is_pp=False,Var_y = True,std:bool=False):
		I = lambda m_T : self.integrand_qqbar(var,m_T,num,mu_factor,mu_f_factor,p_T,n_f,iso,is_pp,Var_y,std)
		sigma, err_sigma = self._quad_mT(I,var,p_T,Var_y)
		return (conv_fact*sigma,conv_fact*err_sigma)
	
	def dsigma_tot_dvar(self,var,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',is_pp=False,Var_y=True,std:bool=False):
		I = lambda m_T : self.integrand_tot(var,m_T,num,mu_factor,mu_f_factor,p_T,n_f,iso,is_pp,Var_y,std)
		sigma, err_sigma = self._quad_mT(I,var,p_T,Var_y)
		return (conv_fact*sigma,conv_fact*err_sigma)
	
	def dsigma_gg_Y_or_x_F(self,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,is_pp=False, Var_y= True,std:bool=False):
		rs = self.rs
		if Var_y:
			L = Y_list(rs,p_T)
		else:
			L = X_F_list(rs,p_T)
		Sigma , Err_sigma = [], []
		for var in L:
			sigma, err_sigma = self.dsigma_gg_dvar(var,p_T,num,mu_factor,mu_f_factor,n_f,is_pp,Var_y,std)
			Sigma.append(sigma), Err_sigma.append(err_sigma)
		return [np.array(Sigma),np.array(Err_sigma)]
	
	def dsigma_qqbar_Y_or_x_f(self,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',is_pp=False,Var_y=True,std:bool=False):
		rs = self.rs
		if Var_y:
			L = Y_list(rs,p_T)
		else:
			L = X_F_list(rs,p_T)
		Sigma , Err_sigma = [], []
		for var in L:
			sigma, err_sigma = self.dsigma_qqbar_dvar(var,p_T,num,mu_factor,mu_f_factor,n_f,iso,is_pp,Var_y,std)
			Sigma.append(sigma), Err_sigma.append(err_sigma)
		return [np.array(Sigma),np.array(Err_sigma)]
	
	def dsigma_tot_Y_or_x_f(self,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',is_pp=False,Var_y=True,std:bool=False):
		rs = self.rs
		if Var_y:
			L = Y_list(rs,p_T)
		else:
			L = X_F_list(rs,p_T)
		Sigma , Err_sigma = [], []
		for var in L:
			sigma, err_sigma = self.dsigma_tot_dvar(var,p_T,num,mu_factor,mu_f_factor,n_f,iso,is_pp,Var_y,std)
			Sigma.append(sigma), Err_sigma.append(err_sigma)
		return [np.array(Sigma),np.array(Err_sigma)]
	
	# section efficace test (pour voir si ma partie proba fonctionne bien)

	def sigma_Jpsi(self,y,p_T=1,n=19.2):
		M_T = np.sqrt(m_jpsi**2+p_T**2)
		rs = self.rs
		return (1-2*M_T*np.cosh(y)/rs)**n

	def sigma_Jpsi_xf(self,xf,p_T,n,N):
		rs = self.rs
		M_T = np.sqrt(m_jpsi**2+p_T**2)
		x = xprime_xF(xf,M_T,rs)
		return N*((1-x)**n)/x
	
	# RpA simple sans effet FCEL

	def vanilla_RpA(self,y,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,p_cen = True,std:bool=False):
		if p_cen:
			sig_pp = self.dsigma_tot_dvar(y,p_T,0,mu_factor=mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,is_pp=True,std=std)[0]
		else:
			sig_pp = self.dsigma_tot_dvar(y,p_T,num,mu_factor=mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,is_pp=True,std=std)[0]
		sig_pA = self.dsigma_tot_dvar(y,p_T,num,mu_factor=mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,is_pp=False,std=std)[0]
		return sig_pA/sig_pp

	# Implementation de FCEL pour le cas 100% octet de couleur

	def Integrand_FCEL(self,y,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,iso ='p',var_int='nu',q0=0.07,std:bool=False):
		'''Gives back the integrand of each component as function of nu,nu_min,
		with:
		- y the rapidity
		- p_T the tranverse momentum
		- num the member of the pdf set
		- mu_factor that describes mu = m*mu_factor (same for mu_f_factor)
		- n_f the number of flavours (=3 by default)
		- iso the isospin variable (='p' by default)
		- is_pp a booleen (=False by default) to tell the collision type
		- q0 the transport coefficient
		as:
			[I_gg,I_qbar]'''
		alpha_s = 0.5 # toujours la meme valeur ici?
		rs = self.rs
		A = self.A
		m = lambda m_T: np.sqrt(m_T**2-p_T**2)
		proba_FCEL = lambda m_T :prob.proba(A,1.,rs,p_T,y,alpha_s,N_c,m(m_T),q0,photon=False)
		Xi = 0.5
		sigma_hat = lambda m_T: proba_FCEL(m_T).sigma_hat(Xi)
		chi = lambda m_T: proba_FCEL(m_T).Chi(Xi)
		if var_int == 'nu':	
			P_FCEL = lambda nu,m_T: prob.p_tilde_u(np.exp(nu),chi(m_T),N_c,alpha=alpha_s)
			jacobian_FCEL = lambda nu,m_T: np.exp(nu)*pow(sigma_hat(m_T)*np.exp(nu)+1,-1)
			delta = lambda nu,m_T: np.log(1+sigma_hat(m_T)*np.exp(nu))
			dsigma_gg =lambda nu,m_T: self.integrand_gg(y+delta(nu,m_T),m_T,num,mu_factor=mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,is_pp = True,Var_y=True,std=std)
			dsigma_qqbar =lambda nu,m_T: self.integrand_qqbar(y+delta(nu,m_T),m_T,num,mu_factor=mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso =iso,is_pp = True,Var_y=True,std=std)
			I_gg = lambda nu,m_T: jacobian_FCEL(nu,m_T)*P_FCEL(nu,m_T)*dsigma_gg(nu,m_T)*conv_fact
			I_qqbar = lambda nu,m_T:jacobian_FCEL(nu,m_T)*P_FCEL(nu,m_T)*dsigma_qqbar(nu,m_T)*conv_fact
			return [I_gg,I_qqbar]
		elif var_int == 'delta':												# delta = ln(1+x)
			raise ValueError('Not implemented yet')
		elif var_int == 'nu2':													# u = exp(-nu)
			raise ValueError('Not implemented yet')
		
	def Integrand_FCEL_v2(self,y,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,iso ='p',is_pp= True,var_int='nu',q0=0.07,std:bool=False):
		'''Gives back the integrand of each component as function of nu,nu_min,
		with:
		- y the rapidity
		- p_T the tranverse momentum
		- num the member of the pdf set
		- mu_factor that describes mu = m*mu_factor (same for mu_f_factor)
		- n_f the number of flavours (=3 by default)
		- iso the isospin variable (='p' by default)
		- is_pp a booleen (=False by default) to tell the collision type
		- q0 the transport coefficient
		as:
			[I_gg,I_qbar]'''
		alpha_s = 0.5 # toujours la meme valeur ici?
		rs = self.rs
		A = self.A
		# on remplace les masses par celle du jpsi
		proba_FCEL = prob.proba(A,1.,rs,p_T,y,alpha_s,N_c,m_jpsi,q0,photon=False)
		Xi = 0.5
		sigma_hat = proba_FCEL.sigma_hat(Xi)
		chi = proba_FCEL.Chi(Xi)
		if var_int == 'nu':	
			P_FCEL = lambda nu: prob.p_tilde_u(np.exp(nu),chi,N_c,alpha=alpha_s)
			jacobian_FCEL = lambda nu: np.exp(nu)*pow(sigma_hat*np.exp(nu)+1,-1)
			delta = lambda nu: np.log(1+sigma_hat*np.exp(nu))
			dsigma_gg =lambda nu: self.dsigma_gg_dvar(y+delta(nu),p_T,num,mu_factor=mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,is_pp = is_pp,Var_y=True,std=std)[0]
			dsigma_qqbar =lambda nu: self.dsigma_qqbar_dvar(y+delta(nu),p_T,num,mu_factor=mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso =iso,is_pp = is_pp,Var_y=True,std=std)[0]
			I_gg = lambda nu: jacobian_FCEL(nu)*P_FCEL(nu)*dsigma_gg(nu)
			I_qqbar = lambda nu:jacobian_FCEL(nu)*P_FCEL(nu)*dsigma_qqbar(nu)
			return [I_gg,I_qqbar]
		elif var_int == 'delta':												# delta = ln(1+x)
			raise ValueError('Not implemented yet')
		elif var_int == 'nu2':													# u = exp(-nu)
			raise ValueError('Not implemented yet')
		
	def Integrand_FCEL_test(self,y,p_T,num,n=19.2,mu_factor=1,mu_f_factor=1,n_f=3,iso ='p',var_int='nu',q0=0.07):
		'''Gives back the integrand of each component as function of nu,nu_min,
		with:
		- y the rapidity
		- num the member of the pdf set
		- mu_factor that describes mu = m*mu_factor (same for mu_f_factor)
		- n_f the number of flavours (=3 by default)
		- iso the isospin variable (='p' by default)
		- is_pp a booleen (=False by default) to tell the collision type
		- q0 the transport coefficient
		as:
			I(nu)'''
		alpha_s = 0.5 # toujours la meme valeur ici?
		rs = self.rs
		A = self.A
		m = m_jpsi 		
		proba_FCEL = prob.proba(A,1.,rs,p_T,y,alpha_s,N_c,m,q0,photon=False)
		Xi = 0.5
		sigma_hat = proba_FCEL.sigma_hat(Xi)
		chi = proba_FCEL.Chi(Xi)
		if var_int == 'nu':	
			P_FCEL = lambda nu: prob.p_tilde_u(np.exp(nu),chi,N_c,alpha=alpha_s)
			jacobian_FCEL = lambda nu: np.exp(nu)*pow(sigma_hat*np.exp(nu)+1,-1)
			delta = lambda nu: np.log(1+sigma_hat*np.exp(nu))
			dsigma = lambda nu: self.sigma_Jpsi(y+delta(nu),p_T,n=n)
			I = lambda nu: jacobian_FCEL(nu)*P_FCEL(nu)*dsigma(nu)
			return I
		elif var_int == 'delta':												# delta = ln(1+x)
			raise ValueError('Not implemented yet')
		elif var_int == 'nu2':													# u = exp(-nu)
			raise ValueError('Not implemented yet')
		
	def FCEL_integration(self,y,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',eps = 1e-15,var_int='nu',q0 = 0.07,std:bool=False):
		'''Return the integrated FCELcomponnent,
		with:
		- y, the rapidity
		- num the member of the pdf set
		- mu_factor that describes mu = m*mu_factor (same for mu_f_factor)
		- iso the isospin variable (='p' by default)
		- n_f the number of flavours (=3 by default)
		- is_pp a booleen (=False by default) to tell the collision type
		- eps the minimun delta_y can be
		- var_int (str) the integration variable (="nu" by default)
		- q0 the transport coefficient
		as:
			(sigma_FCEL,err_sigma_FCEL)'''
		if eps<=0:
			raise ValueError("eps <= 0, leading to integration error.")
		else:
			A = self.A
			rs = self.rs
			alpha_s = 0.5														# don't use the pdf one, bc the energy scale here is sqrt(q_hat*L)
			Fc_FCEL = N_c
			Xi = 0.5
			sigma_FCEL = []
			err_sigma_FCEL = []
			m = lambda m_T: np.sqrt(m_T**2-p_T**2)
			prob_FCEL = lambda m_T: prob.proba(A,1.,rs,p_T,y,alpha_s,Fc_FCEL,m(m_T),q0,photon=False)
			sigma_hat = lambda m_T: prob_FCEL(m_T).sigma_hat(Xi)
			delta_y_FCEL_max = lambda m_T: np.log(rs/m_T)-y
			delta_y_min = eps
			if var_int == 'nu':
				max_FCEL = lambda m_T: np.log((np.exp(delta_y_FCEL_max(m_T))-1.)/sigma_hat(m_T))
				if delta_y_min > 1e-15:
					min_FCEL =  lambda m_T: np.log((np.exp(delta_y_min)-1)/sigma_hat(m_T))
				elif delta_y_min > 0:
					min_FCEL = lambda m_T:  np.log(delta_y_min/sigma_hat(m_T))
			I_gg, I_qqbar = self.Integrand_FCEL(y,p_T,num,mu_factor,mu_f_factor,n_f,iso,var_int=var_int,q0=q0,std=std)
			I_FCEL = lambda nu,m_T: I_gg(nu,m_T)+I_qqbar(nu,m_T)
			Int_FCEL = lambda m_T: integrate.quad(lambda nu:I_FCEL(nu,m_T),min_FCEL(m_T),max_FCEL(m_T))[0]
			Int = integrate.quad(Int_FCEL,self.m_T_min(p_T),self.m_T_max(y,p_T))
			# print('FCEL =' +str(Int))
			sigma_FCEL=Int[0]		# Conversion factor is already taken in count here
			err_sigma_FCEL=Int[1]
			return (sigma_FCEL,err_sigma_FCEL)
		
	def FCEL_integration_v2(self,y,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',is_pp=True,eps = 1e-15,var_int='nu',q0 = 0.07,std:bool=False):
		'''Return the integrated FCELcomponnent,
		with:
		- y, the rapidity
		- num the member of the pdf set
		- mu_factor that describes mu = m*mu_factor (same for mu_f_factor)
		- iso the isospin variable (='p' by default)
		- n_f the number of flavours (=3 by default)
		- is_pp a booleen (=False by default) to tell the collision type
		- eps the minimun delta_y can be
		- var_int (str) the integration variable (="nu" by default)
		- q0 the transport coefficient
		as:
			(sigma_FCEL,err_sigma_FCEL)'''
		if eps<=0:
			raise ValueError("eps <= 0, leading to integration error.")
		else:
			A = self.A
			rs = self.rs
			alpha_s = 0.5														# don't use the pdf one, bc the energy scale here is sqrt(q_hat*L)
			Fc_FCEL = N_c
			Xi = 0.5
			sigma_FCEL = []
			err_sigma_FCEL = []
			prob_FCEL = prob.proba(A,1.,rs,p_T,y,alpha_s,Fc_FCEL,m_jpsi,q0,photon=False)
			sigma_hat = prob_FCEL.sigma_hat(Xi)
			delta_y_FCEL_max = np.log(rs/self.M_jpsi_T(p_T))-y
			delta_y_min = eps
			if var_int == 'nu':
				max_FCEL = np.log((np.exp(delta_y_FCEL_max)-1.)/sigma_hat)
				if delta_y_min > 1e-15:
					min_FCEL =  np.log((np.exp(delta_y_min)-1)/sigma_hat)
				elif delta_y_min > 0:
					min_FCEL =  np.log(delta_y_min/sigma_hat)
			I_gg, I_qqbar = self.Integrand_FCEL_v2(y,p_T,num,mu_factor=mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso=iso,is_pp=is_pp,var_int=var_int,q0=q0,std=std)
			I_FCEL = lambda nu: I_gg(nu)+I_qqbar(nu)
			Int = integrate.quad(I_FCEL,min_FCEL,max_FCEL)
			# print('FCEL =' +str(Int))
			sigma_FCEL=Int[0]		# Conversion factor is already taken in count here
			err_sigma_FCEL=Int[1]
			return (sigma_FCEL,err_sigma_FCEL)
		
	def FCEL_integration_test(self,y,p_T,num,n=19.2,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',eps = 1e-15,var_int='nu',q0 = 0.07):
		'''Return the integrated FCELcomponnent,
		with:
		- y, the rapidity
		- num the member of the pdf set
		- mu_factor that describes mu = m*mu_factor (same for mu_f_factor)
		- iso the isospin variable (='p' by default)
		- n_f the number of flavours (=3 by default)
		- is_pp a booleen (=False by default) to tell the collision type
		- eps the minimun delta_y can be
		- var_int (str) the integration variable (="nu" by default)
		- q0 the transport coefficient
		as:
			(sigma_FCEL,err_sigma_FCEL)'''
		if eps<=0:
			raise ValueError("eps <= 0, leading to integration error.")
		else:
			A = self.A
			rs = self.rs
			alpha_s = 0.5														# don't use the pdf one, bc the energy scale here is sqrt(q_hat*L)
			Fc_FCEL = N_c
			Xi = 0.5
			sigma_FCEL = []
			err_sigma_FCEL = []
			prob_FCEL = prob.proba(A,1.,rs,p_T,y,alpha_s,Fc_FCEL,m_jpsi,q0,photon=False)
			sigma_hat =  prob_FCEL.sigma_hat(Xi)
			delta_y_FCEL_max = np.log(rs/m_max)-y
			delta_y_min = eps
			if var_int == 'nu':
				max_FCEL = np.log((np.exp(delta_y_FCEL_max)-1.)/sigma_hat)
				if delta_y_min > 1e-15:
					min_FCEL =  np.log((np.exp(delta_y_min)-1)/sigma_hat)
				elif delta_y_min > 0:
					min_FCEL =  np.log(delta_y_min/sigma_hat)
			I_FCEL = self.Integrand_FCEL_test(y,p_T,num,n,mu_factor,mu_f_factor,n_f,iso,var_int=var_int,q0=q0)
			Int_FCEL = integrate.quad(I_FCEL,min_FCEL,max_FCEL)
			sigma_FCEL=Int_FCEL[0]		# Conversion factor is already taken in count here
			err_sigma_FCEL=Int_FCEL[1]
			return (sigma_FCEL,err_sigma_FCEL)
		
	def FCEL_integration_dy(self,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',eps = 1e-15,var_int='nu',q0 = 0.07,version=1,std:bool=False):
		'''Return the integrated FCEL and FCEG componnents rapidity dependent,
		with:
		- num the member of the pdf set
		- mu_factor that describes mu = m*mu_factor (same for mu_f_factor)
		- iso the isospin variable (='p' by default)
		- n_f the number of flavours (=3 by default)
		- is_pp a booleen (=False by default) to tell the collision type
		- eps the minimun delta_y can be
		- var_int (str) the integration variable (="nu" by default)
		- q0 the transport coefficient
		- version is the number of integration order for FCEL (1 by default for )
		as:
			(sigma_FCEL,err_sigma_FCEL)'''
		if eps<=0:
			raise ValueError("eps <= 0, leading to integration error.")
		else:
			rs = self.rs
			Y = Y_list(rs,p_T)
			sigma_FCEL = []
			err_sigma_FCEL = []
			for i,y in enumerate(Y):
				print('y = '+str(y)+'| '+str(i+1)+'/'+str(N_y))
				if version==1:
					Int = self.FCEL_integration(y,p_T,num,mu_factor,mu_f_factor,n_f,iso,eps,var_int,q0,std=std)
				elif version==2:
					Int = self.FCEL_integration_v2(y,p_T,num,mu_factor,mu_f_factor,n_f,iso,eps,var_int,q0,std=std)
				sigma_FCEL.append(Int[0]) 			# Conversion factor is already taken in count here
				err_sigma_FCEL.append(Int[1])
			return (np.array(sigma_FCEL),np.array(err_sigma_FCEL))
		
	def FCEL_integration_dy_test(self,p_T,num,n=19.2,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',eps = 1e-15,var_int='nu',q0 = 0.07):
		'''Return the integrated FCEL and FCEG componnents rapidity dependent,
		with:
		- p_T the transverse momentum
		- num the member of the pdf set
		- mu_factor that describes mu = m*mu_factor (same for mu_f_factor)
		- iso the isospin variable (='p' by default)
		- n_f the number of flavours (=3 by default)
		- is_pp a booleen (=False by default) to tell the collision type
		- eps the minimun delta_y can be
		- var_int (str) the integration variable (="nu" by default)
		- q0 the transport coefficient
		as:
			(sigma_FCEL,err_sigma_FCEL)'''
		if eps<=0:
			raise ValueError("eps <= 0, leading to integration error.")
		else:
			rs = self.rs
			Y = Y_list(rs)
			sigma_FCEL = []
			err_sigma_FCEL = []
			for i,y in enumerate(Y):
				print('y = '+str(y)+'| '+str(i+1)+'/'+str(N_y))
				Int_FCEL = self.FCEL_integration_test(y,p_T,num,n,mu_factor,mu_f_factor,n_f,iso,eps,var_int,q0)
				sigma_FCEL.append(Int_FCEL[0]) 			# Conversion factor is already taken in count here
				err_sigma_FCEL.append(Int_FCEL[1])
			return (np.array(sigma_FCEL),np.array(err_sigma_FCEL))
		
	def RpA(self,y,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',eps = 1e-15,var_int='nu',q0=0.07,p_cen = True,std:bool=False):
		rs = self.rs
		FCEL_cen = self.FCEL_integration_v2(y,p_T,num,mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso=iso,is_pp=False,eps = eps,var_int=var_int,q0=q0,std=std)
		if p_cen:
			sigma_pp,err_pp = self.dsigma_tot_dvar(y,p_T,0, mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso =iso,is_pp=True,Var_y=True,std=std)
		else:
			sigma_pp,err_pp= self.dsigma_tot_dvar(y,p_T,num,mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso =iso,is_pp=True,Var_y=True,std=std)
		Ucen = FCEL_cen[0]/sigma_pp
		return Ucen

	def Rpp(self,y,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',eps = 1e-15,var_int='nu',q0=0.07,p_cen = True,std:bool=False):
			rs = self.rs
			FCEL_cen = self.FCEL_integration_v2(y,p_T,num,mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso=iso,is_pp=True,eps = eps,var_int=var_int,q0=q0,std=std)
			if p_cen:
				sigma_pp,err_pp = self.dsigma_tot_dvar(y,p_T,0, mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso =iso,is_pp=True,Var_y=True,std=std)
			else:
				sigma_pp,err_pp= self.dsigma_tot_dvar(y,p_T,num,mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso =iso,is_pp=True,Var_y=True,std=std)
			Ucen = FCEL_cen[0]/sigma_pp
			return Ucen

	### FCEL in x_F (fixed-target data, e.g. E866) ###

	def y_of_xF(self,xF:float,p_T:float)->float:
		"""Rapidity of a J/psi (transverse mass M_T) at a given x_F, from x_F = x1 - x2 = 2 M_T sinh(y)/rs"""
		return np.arcsinh(xF*self.rs/(2.*self.M_jpsi_T(p_T)))

	def xF_of_y(self,y:float,p_T:float)->float:
		"""Inverse of y_of_xF"""
		return 2.*self.M_jpsi_T(p_T)*np.sinh(y)/self.rs

	def sigma_xF_interp(self,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',is_pp=False,std:bool=False,xF_min=-0.15,xF_max=0.99,n_pts=100):
		"""Tabulate the CEM cross section dsigma/dx_F (no FCEL) on a x_F grid for the member `num`
		and return a fast callable xF -> dsigma/dxF (log-PCHIP interpolation, 0 outside the grid).
		It avoids recomputing the m_T integral at each q0 / each point of the FCEL integral.
		"""
		xs = np.linspace(xF_min,xF_max,n_pts)
		vals = np.array([self.dsigma_tot_dvar(x,p_T,num,mu_factor=mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso=iso,is_pp=is_pp,Var_y=False,std=std)[0] for x in xs])
		ok = vals > 0
		xs, vals = xs[ok], vals[ok]
		spl = PchipInterpolator(xs,np.log(vals))
		lo, hi = xs[0], xs[-1]
		def sigma(x):
			x = float(x)
			if x < lo or x > hi:
				return 0.
			return float(np.exp(spl(x)))
		return sigma

	def sigma_xF_param(self,p_T,n,norm=1.):
		"""Data-driven pp cross section of Arleo-Peigne (JHEP03(2013)122, Eq. 3.10): dsigma/dxF = norm (1-x')^n/x', x' = sqrt(xF^2+4M_T^2/s).
		Returns a callable xF -> dsigma/dxF (0 for x' >= 1). The normalisation cancels in R_pA."""
		M_T = self.M_jpsi_T(p_T)
		def sigma(x):
			xp = np.sqrt(x**2+4.*M_T**2/self.rs**2)
			return norm*(1.-xp)**n/xp if xp < 1. else 0.
		return sigma

	def FCEL_integration_xF(self,xF,p_T,sigma0,q0=0.07,eps=1e-15,B=1.,jacobian=True):
		"""Return the FCEL-convoluted cross section at a given x_F, following Arleo-Peigne (JHEP03(2013)122), Eq. (3.1)-(3.5):
			dsigma_pA/dxF(xF) = int_0^{eps_max} d(eps) P(eps,E) dsigma0/dxF(xF + delta_xF(eps))
		with eps_max = min(E_p - E, E), E = x1 E_p, and xF + delta_xF = xF(E+eps) = (E+eps)/E_p - M_T^2 E_p/(s (E+eps)).
		In terms of the variables of FCEL_integration_v2: u = eps/omega_hat = exp(nu), eps/E = sigma_hat*u, and P d(eps) = p_tilde(u) du = u p_tilde(u) dnu.
		The Jacobian factor of Eq. (3.6), E x'(E+eps)/((E+eps) x'(E)) = x'(xF_new)/(x'(xF) (1+eps/E)), is included when jacobian=True (it tends to 1 at large x_F).
		B is the reference nucleus of the spectrum (Eq. 2.17): B=1 -> A relative to the proton; B=A_Be -> A relative to Be, to be convoluted with the p-Be cross section for R_W/Be.

		Args:
			xF (float): Feynman x
			p_T (float): transverse momentum (GeV)
			sigma0 (callable): xF' -> dsigma/dxF' without FCEL, e.g. from sigma_xF_interp or sigma_xF_param
			q0 (float): transport coefficient (GeV^2/fm)
			eps (float): minimum delta_y
			B (float): mass number of the reference nucleus (Lambda_B^2 = max(Lambda_QCD^2, l_B^2))
			jacobian (bool): include the Jacobian of Eq. (3.6)

		Returns:
			(sigma_FCEL, err): same units as sigma0
		"""
		if eps <= 0:
			raise ValueError("eps <= 0, leading to integration error.")
		rs = self.rs
		M_T = self.M_jpsi_T(p_T)
		alpha_s = 0.5
		Fc_FCEL = N_c
		Xi = 0.5
		y0 = self.y_of_xF(xF,p_T)
		prob_FCEL = prob.proba(self.A,B,rs,p_T,y0,alpha_s,Fc_FCEL,m_jpsi,q0,photon=False)
		sigma_hat = prob_FCEL.sigma_hat(Xi)
		chi = prob_FCEL.Chi(Xi)
		delta_max = min(np.log(2.),np.log(rs/M_T) - y0)                     # eps_max = min(E_p-E, E): eps <= E and x1' <= 1
		if chi >= 1. or delta_max <= 0:                                   # Lambda_A <= Lambda_B : no energy loss (Heaviside of the model)
			return (sigma0(xF),0.)
		nu_min = np.log(eps/sigma_hat)
		nu_max = np.log((np.exp(delta_max)-1.)/sigma_hat)
		def integrand(nu):
			u = np.exp(nu)
			delta = np.log1p(sigma_hat*u)                                   # x1 -> x1 (1+eps/E)
			xF_new = self.xF_of_y(y0+delta,p_T)
			jac = 1.
			if jacobian:
				jac = np.sqrt(xF_new**2+4.*M_T**2/rs**2)/(np.sqrt(xF**2+4.*M_T**2/rs**2)*(1.+sigma_hat*u))
			return u*prob.p_tilde_u(u,chi,Fc_FCEL,alpha=alpha_s)*sigma0(xF_new)*jac
		pts = [0.] if nu_min < 0. < nu_max else None
		res = integrate.quad(integrand,nu_min,nu_max,points=pts,limit=N_limit)
		return (res[0],res[1])

	def RpA_xF(self,xF,p_T,num,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',eps=1e-15,q0=0.07,p_cen=True,std:bool=False,sigma0=None,sigma_pp=None):
		"""R_pA(x_F) (per nucleon, with FCEL). `sigma0` / `sigma_pp` can be fast callables xF -> dsigma/dxF
		(see sigma_xF_interp); if None they are computed with the m_T integral."""
		if sigma0 is None:
			sigma0 = lambda x: self.dsigma_tot_dvar(x,p_T,num,mu_factor=mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso=iso,is_pp=False,Var_y=False,std=std)[0]
		if sigma_pp is None:
			sigma_pp = lambda x: self.dsigma_tot_dvar(x,p_T,0 if p_cen else num,mu_factor=mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso=iso,is_pp=True,Var_y=False,std=std)[0]
		return self.FCEL_integration_xF(xF,p_T,sigma0,q0=q0,eps=eps)[0]/sigma_pp(xF)


	def Uncertainties_RpA_dy(self,p_T,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',eps = 1e-15,var_int='nu',q0_list=[0.07,0.05,0.09],var_err='q0,mu,pdf',version=1,std:bool=False):
		'''Return the uncertinites on the final R_pA (i.e taking in count FCEL effects), with:
		- p_T the transverse momentum
		- mu_factor that describes mu = m*mu_factor (same for mu_f_factor) (maybe m_T ???)
		- n_f the number of flavours (=3 by default)
		- iso the isospin variable (='p' by default)
		- eps the minimun delta_y can value
		- var_int (str) the integration variable (="nu" by default)
		- q0_list the transport coefficient with its uncertaintites as [q0,q0-,q0+]
		- var_err (str) containing the error on variable wanted like: "q0","mu","pdf" (can contains multiple ones like:"q0,pdf")
		- version number
		as:
			[Y,Ucen,[Uplus,Uminus],[Uplus_q,Uminus_q,Uplus_mu,Uminus_mu,Uplus_pdf,Uminus_pdf]]
		so :
		- for the 0th index the rapidity vector
		- for the 1st index the central value
		- for the 2nd index the + and - total uncertainties
		- for the 3rd index the + and - uncertainties for (in order) q0, mu, and pdfs'''
		p_set = self.p_set
		rs = self.rs
		Y = Y_list(rs,p_T)
		q_list = [q0_list[1],q0_list[2]]
		# factor = [1./2.,2]
		factor = [1./np.sqrt(2),np.sqrt(2)]
		mu_factor_list = [(i,j) for i in factor for j in factor]
		q0_cen =q0_list[0]
		num_cen = 0
		print('Central value')
		FCEL_cen = self.FCEL_integration_dy(p_T,num_cen,mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso=iso,eps = eps,var_int=var_int,q0=q0_cen,version=version,std=std)
		sigma_pp_cen,err_pp_cen = self.dsigma_tot_Y_or_x_f(p_T,num_cen, mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso =iso,is_pp=True,Var_y=True,std=std)
		Ucen = np.array(FCEL_cen[0]/sigma_pp_cen)
		if 'q0' in var_err:
			q_val= []
			for q in q_list:
				print('q0 = '+str(q))
				FCEL_q  = self.FCEL_integration_dy(p_T,num_cen,mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso=iso,eps = eps,var_int=var_int,q0=q,version=version,std=std)
				q_val.append(FCEL_q[0]/sigma_pp_cen)
			Uminus_q , Uplus_q = min_max(q_val)
			Uminus_q, Uplus_q = Ucen - Uminus_q, Uplus_q - Ucen
		if 'mu' in var_err:
			mu_val =[]
			for (a,b) in mu_factor_list:
				print('(mu_factor,mu_f_factor) = '+str((a,b)))
				FCEL_mu = self.FCEL_integration_dy(p_T,num_cen,mu_factor= a*mu_factor,mu_f_factor=b*mu_f_factor,n_f=n_f,iso=iso,eps = eps,var_int=var_int,q0=q0_cen,version=version,std=std)
				sigma_pp_mu , err_pp_mu = self.dsigma_tot_Y_or_x_f(p_T,num_cen,mu_factor= a*mu_factor,mu_f_factor=b*mu_f_factor,n_f=n_f,iso = iso,is_pp=True,Var_y=True,std=std)
				mu_val.append(FCEL_mu[0]/sigma_pp_mu)
			Uminus_mu, Uplus_mu = min_max(mu_val)
			Uminus_mu , Uplus_mu = Ucen-Uminus_mu, Uplus_mu-Ucen
		if 'pdf' in var_err:
			Uplus_pdf, Uminus_pdf, Ucen_pdf = [],[],[]
			print('Computation on pdf set')
			for j,y in enumerate(Y):
				print('y = '+str(y)+'| '+str(j+1)+'/'+str(N_y))
				y_pdf_val =[Ucen[j]]
				for i in range(1,p_set.size):
					if version==1:
						FCEL_pdf = self.FCEL_integration(y,p_T,i,mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso=iso,eps = eps,var_int=var_int,q0=q0_cen,std=std)
					elif version==2:
						FCEL_pdf = self.FCEL_integration_v2(y,p_T,i,mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso=iso,eps = eps,var_int=var_int,q0=q0_cen,std=std)
					sigma_pp_pdf , err_pp_pdf = self.dsigma_tot_dvar(y,p_T,i,mu_factor=mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso = iso,is_pp=True,Var_y=True,std=std)
					y_pdf_val.append(FCEL_pdf[0]/sigma_pp_pdf)
				unc_pdf = p_set.uncertainty(y_pdf_val)
				Uplus_pdf.append(unc_pdf.errplus); Uminus_pdf.append(unc_pdf.errminus);Ucen_pdf.append(unc_pdf.central)
			Uplus_pdf , Uminus_pdf, Ucen = np.array(Uplus_pdf), np.array(Uminus_pdf), np.array(Ucen_pdf) # if pdf are ploted, the central value will be the one on PDF
		if 'pdf' not in var_err:
			Uplus_pdf, Uminus_pdf = np.zeros_like(Y),np.zeros_like(Y)
		if 'mu' not in var_err:
			Uplus_mu, Uminus_mu = np.zeros_like(Y),np.zeros_like(Y)
		if 'q0' not in var_err:
			Uplus_q, Uminus_q = np.zeros_like(Y),np.zeros_like(Y)
		Uplus = np.sqrt(Uplus_q**2+Uplus_mu**2+Uplus_pdf**2)
		Uminus = np.sqrt(Uminus_q**2+Uminus_mu**2+Uminus_pdf**2)
		return [Y,Ucen,[Uplus,Uminus],[Uplus_q,Uminus_q,Uplus_mu,Uminus_mu,Uplus_pdf,Uminus_pdf]]
	
	def Uncertainties_RpA_dy_test(self,p_T,n = 19.2,mu_factor=1,mu_f_factor=1,n_f=3,iso='p',eps = 1e-15,var_int='nu',q0_list=[0.07,0.05,0.09],var_err='q0,mu,pdf'):
		'''Return the uncertinites on the final R_pA (i.e taking in count FCEL effects), with:
		- mu_factor that describes mu = m*mu_factor (same for mu_f_factor)
		- n_f the number of flavours (=3 by default)
		- iso the isospin variable (='p' by default)
		- eps the minimun delta_y can value
		- var_int (str) the integration variable (="nu" by default)
		- q0_list the transport coefficient with its uncertaintites as [q0,q0-,q0+]
		- var_err (str) containing the error on variable wanted like: "q0","mu","pdf" (can contains multiple ones like:"q0,pdf")
		as:
			[Y,Ucen,[Uplus,Uminus],[Uplus_q,Uminus_q,Uplus_mu,Uminus_mu,Uplus_pdf,Uminus_pdf]]
		so :
		- for the 0th index the rapidity vector
		- for the 1st index the central value
		- for the 2nd index the + and - total uncertainties
		- for the 3rd index the + and - uncertainties for (in order) q0, mu, and pdfs'''
		p_set = self.p_set
		rs = self.rs
		Y = Y_list(rs)
		q_list = [q0_list[1],q0_list[2]]
		factor = [1./2.,2]
		mu_factor_list = [(i,j) for i in factor for j in factor]
		q0_cen =q0_list[0]
		num_cen = 0
		print('Central value')
		FCEL_cen = self.FCEL_integration_dy_test(p_T,num_cen,n=n,mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso=iso,eps = eps,var_int=var_int,q0=q0_cen)
		# print(FCEL_cen[0])
		sigma_pp_cen = np.array([self.sigma_Jpsi(y,p_T,n=n) for y in Y])
		Ucen = np.array(FCEL_cen[0]/sigma_pp_cen)
		if 'q0' in var_err:
			q_val= []
			for q in q_list:
				print('q0 = '+str(q))
				FCEL_q  = self.FCEL_integration_dy_test(p_T,num_cen,n=n,mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso=iso,eps = eps,var_int=var_int,q0=q)
				q_val.append(FCEL_q[0]/sigma_pp_cen)
			Uminus_q , Uplus_q = min_max(q_val)
			Uminus_q, Uplus_q = Ucen - Uminus_q, Uplus_q - Ucen
		if 'mu' in var_err:
			mu_val =[]
			for (a,b) in mu_factor_list:
				print('(mu_factor,mu_f_factor) = '+str((a,b)))
				FCEL_mu = self.FCEL_integration_dy_test(p_T,num_cen,n=n,mu_factor= a*mu_factor,mu_f_factor=b*mu_f_factor,n_f=n_f,iso=iso,eps = eps,var_int=var_int,q0=q0_cen)
				mu_val.append(FCEL_mu[0]/sigma_pp_cen)
			Uminus_mu, Uplus_mu = min_max(mu_val)
			Uminus_mu , Uplus_mu = Ucen-Uminus_mu, Uplus_mu-Ucen
		if 'pdf' in var_err:
			Uplus_pdf, Uminus_pdf , Ucen_pdf = [],[],[]
			print('Computation on pdf set')
			for j,y in enumerate(Y):
				print('y = '+str(y)+'| '+str(j+1)+'/'+str(N_y))
				y_pdf_val =[Ucen[j]]
				for i in range(1,p_set.size):
					FCEL_pdf = self.FCEL_integration_test(y,p_T,i,n=n,mu_factor= mu_factor,mu_f_factor=mu_f_factor,n_f=n_f,iso=iso,eps = eps,var_int=var_int,q0=q0_cen)[0]
					y_pdf_val.append(FCEL_pdf[0]/sigma_pp_cen)
				unc_pdf = p_set.uncertainty(y_pdf_val)
				Uplus_pdf.append(unc_pdf.errplus); Uminus_pdf.append(unc_pdf.errminus); Ucen_pdf.append(unc_pdf.central)
			Uplus_pdf , Uminus_pdf, Ucen = np.array(Uplus_pdf), np.array(Uminus_pdf)
		if 'pdf' not in var_err:
			Uplus_pdf, Uminus_pdf = np.zeros_like(Y),np.zeros_like(Y)
		if 'mu' not in var_err:
			Uplus_mu, Uminus_mu = np.zeros_like(Y),np.zeros_like(Y)
		if 'q0' not in var_err:
			Uplus_q, Uminus_q = np.zeros_like(Y),np.zeros_like(Y)
		Uplus = np.sqrt(Uplus_q**2+Uplus_mu**2+Uplus_pdf**2)
		Uminus = np.sqrt(Uminus_q**2+Uminus_mu**2+Uminus_pdf**2)
		return [Y,Ucen,[Uplus,Uminus],[Uplus_q,Uminus_q,Uplus_mu,Uminus_mu,Uplus_pdf,Uminus_pdf]]