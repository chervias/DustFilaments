import numpy as np

class FilPop:
	def __init__(self,Nfil,theta_LH_RMS,size_ratio,size_scale,slope,magfield,seed,fixed_distance=True):
		self.Nfil			= Nfil
		self.realNfil    = Nfil
		self.magfield		= magfield
		self.fixed_distance	= fixed_distance
		self.max_length		= 1.0
		np.random.seed(seed)
		self.slope			= slope
		self.size_scale		= size_scale
		self.size_ratio		= size_ratio
		if theta_LH_RMS == None:
			self.theta_LH_RMS	= None
		elif theta_LH_RMS == -1.0:
			self.theta_LH_RMS	= -1.0
		else:
			self.theta_LH_RMS	= np.radians(theta_LH_RMS)
		self.centers				= self.get_centers()	
		self.angles,self.long_vec	= self.get_angles()
		self.sizes					= self.get_sizes()
	def get_centers(self):
		centers	= np.zeros((self.Nfil,3))
		# recipe to generate random centers
		# Usually this will be done in spherical coordinates
		if self.fixed_distance:
			radii_random	= np.ones(self.Nfil) * 0.4*self.magfield.size
			phi_random		= 2*np.pi*np.random.uniform(0.0,1.0,self.Nfil)
			theta_random	= np.arccos(1.0 - 2*np.random.uniform(0.0,1.0,self.Nfil))
			centers[:,0]	= radii_random*np.sin(theta_random)*np.cos(phi_random)
			centers[:,1]	= radii_random*np.sin(theta_random)*np.sin(phi_random)
			centers[:,2]	= radii_random*np.cos(theta_random)
		else:
			l_rand	 = np.random.uniform(0.2,1.0,self.Nfil)
			u_rand   = np.random.uniform(-1.0,1.0,self.Nfil)
			phi_rand = np.random.uniform(0.0,2.*np.pi,self.Nfil)
			centers[:,0] = (0.5*self.magfield.size)*l_rand**(1.0/3.0)*np.sqrt(1. - u_rand**2)*np.cos(phi_rand)
			centers[:,1] = (0.5*self.magfield.size)*l_rand**(1.0/3.0)*np.sqrt(1. - u_rand**2)*np.sin(phi_rand)
			centers[:,2] = (0.5*self.magfield.size)*l_rand**(1.0/3.0)*u_rand
		return centers
	def get_angles(self):
		angles			= np.zeros((self.Nfil,2))
		# get the euler angles according to the local magnetic field in the center pixel. The hatZ vector of the filament (long axis) follows local B
		local_magfield	= np.array([self.magfield.interp_fn((self.centers[n,0],self.centers[n,1],self.centers[n,2])) for n in range(self.Nfil)])
		# unit vector along the local mag field
		hatZ			= np.array([local_magfield[n,:]/np.linalg.norm(local_magfield[n,:]) for n in range(self.Nfil)])
		# we need the r unit vector, which corresponds to z in the local triad
		hatr = np.array([self.centers[n]/np.linalg.norm(self.centers[n]) for n in range(self.Nfil)])
		# rotate hatZ around hatr
		# This is in radians
		#theta_LH		= self.theta_LH_RMS*np.ones(self.Nfil) # Fixed angle between filament and mag field
		theta_LH		= np.fabs(np.random.normal(0,self.theta_LH_RMS,self.Nfil))
		#theta_LH		= np.random.normal(loc=0.0,scale=self.theta_LH_RMS,size=self.Nfil)
		self.theta_LH = theta_LH
		hatZprime		= np.array([hatZ[n]*np.cos(-theta_LH[n]) + np.cross(hatr[n],hatZ[n])*np.sin(-theta_LH[n]) + hatr[n]*np.dot(hatr[n],hatZ[n])*(1 - np.cos(-theta_LH[n])) for n in range(self.Nfil)])
		norm_hatZprime = np.linalg.norm(hatZprime,axis=1)
		# beta angle
		angles[:,1]		= np.arccos(hatZprime[:,2]/norm_hatZprime)
		# alpha angle
		angles[:,0]		= np.arctan2(hatZprime[:,1],hatZprime[:,0])
		return angles,hatZprime
	def get_sizes(self):
		# The sizes will be the ellipsoid semi axes a,b,c with a=b<c
		sizes			= np.zeros((self.Nfil,3))
		c_semiaxis		= self.size_scale*(1.0+np.random.pareto(self.slope-1,size=self.Nfil))
		c_semiaxis = 1.0*np.ones(self.Nfil)
		#a_semiaxis		= self.size_scale*(1.0+np.random.pareto(self.slope-1,size=self.Nfil))
		#sizes[:,2]		= np.clip(c_semiaxis,0,self.max_length)
		sizes[:,2]		= c_semiaxis
		sizes[:,0]		= self.size_ratio*sizes[:,2]
		sizes[:,1]		= self.size_ratio*sizes[:,2]
		#sizes[:,0] = a_semiaxis
		#sizes[:,1] = a_semiaxis
		#sizes[:,2] = (1./self.size_ratio)*sizes[:,0]
		return sizes

#---------------------------------------------------------------------