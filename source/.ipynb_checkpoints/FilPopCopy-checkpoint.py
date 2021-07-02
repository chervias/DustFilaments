import numpy as np
import healpy as hp

class FilPop:
	def __init__(self,nside,Nfil,theta_LH_RMS,size_ratio,size_scale,slope,magfield,seed,dust_template,fixed_distance=False):
		self.nside = nside
		self.Nfil			= Nfil
		self.magfield		= magfield
		self.fixed_distance	= fixed_distance
		self.max_length		= 1.0
		np.random.seed(seed)
		self.slope			= slope
		self.size_scale		= size_scale
		self.size_ratio		= size_ratio
		self.dust_template  = dust_template
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
		nside_galaxy = 2048
		# Now the centers will be defined by a template map
		# first normalize the map to the total number of filaments
		map_original = hp.read_map(self.dust_template,field=0,nest=False)
		map_nside = hp.ud_grade(map_original,nside_galaxy)
		# each pixel is mult by C --> Nfil = C*Sum(map)
		C = self.Nfil / np.sum(map_nside)
		number_fil = np.random.poisson(C*map_nside,12*nside_galaxy**2)
		self.realNfil = np.sum(number_fil)
		centers	= np.zeros((self.realNfil,3))
		# recipe to generate random centers
		# Usually this will be done in spherical coordinates
		l_rand	 = np.random.uniform(0.2,1.0,self.realNfil)
		radii_arr = (0.5*self.magfield.size)*l_rand**(1.0/3.0)
		counter = 0
		for n in range(12*nside_galaxy**2):
			if number_fil[n] > 0:
				for k in range(number_fil[n]):
					centers[counter,:] = radii_arr[counter]*np.array(hp.pix2vec(nside_galaxy,n,nest=False))
					counter = counter + 1
			else:
				continue
		print('Real number of fil',self.realNfil,'counter',counter)
		return centers
	def get_angles(self):
		angles			= np.zeros((self.realNfil,2))
		# get the euler angles according to the local magnetic field in the center pixel. The hatZ vector of the filament (long axis) follows local B
		local_magfield	= np.array([self.magfield.interp_fn((self.centers[n,0],self.centers[n,1],self.centers[n,2])) for n in range(self.realNfil)])
		#for n in range(self.realNfil):
			#print(local_magfield[n])
		if self.theta_LH_RMS == None:
			# unit vector along the local mag field
			hatZ			= np.array([local_magfield[n,:]/np.linalg.norm(local_magfield[n,:]) for n in range(self.realNfil)])
			# alpha angle
			angles[:,1]		= np.arccos(hatZ[:,2])
			# beta angle
			angles[:,0]		= np.arctan2(hatZ[:,1],hatZ[:,0])
			return angles,hatZ
		elif self.theta_LH_RMS == -1:
			# we want a unit vector that is ort to center vector
			ort_vec 		= np.array([np.cross(self.centers[n],np.array([1,1,1])) for n in range(self.realNfil)])
			# hatk is the unit vector along the LOS 
			hatk = np.array([self.centers[n]/np.linalg.norm(self.centers[n]) for n in range(self.realNfil)])
			# we rotate the ort vector by a random angle between 0 and 2pi
			phi_angle   = np.random.uniform(0,2*np.pi,self.realNfil)
			ort_vec_rotated		= np.array([ort_vec[n]*np.cos(phi_angle[n]) + np.cross(hatk[n],ort_vec[n])*np.sin(phi_angle[n]) + hatk[n]*np.dot(hatk[n],ort_vec[n])*(1 - np.cos(phi_angle[n])) for n in range(self.realNfil)])
			ort_vec_unit	= np.array([ort_vec_rotated[n]/np.linalg.norm(ort_vec_rotated[n]) for n in range(self.realNfil)])
			# alpha angle
			angles[:,1]		= np.arccos(ort_vec_unit[:,2])
			# beta angle
			angles[:,0]		= np.arctan2(ort_vec_unit[:,1],ort_vec_unit[:,0])
			return angles,ort_vec_unit
		else:
			# unit vector along the local mag field
			hatZ			= np.array([local_magfield[n,:]/np.linalg.norm(local_magfield[n,:]) for n in range(self.realNfil)])
			# we need a second unit vector hatY perpendicular to hatZ
			random_vectors  = np.random.uniform(0.0,1.0,size=(self.realNfil,3))
			vecY			= np.cross(hatZ,random_vectors)
			hatY			= np.array([vecY[n,:]/np.linalg.norm(vecY[n,:]) for n in range(self.realNfil)])
			# This is in radians
			theta_LH		= np.fabs(np.random.normal(0,self.theta_LH_RMS,self.realNfil))
			phi				= np.random.uniform(0,2*np.pi,self.realNfil)
			#phi = np.zeros(self.realNfil)
			# We rotate hatZ around hatY by theta_LH using Rodrigues formula
			hatZprime		= np.array([hatZ[n,:]*np.cos(theta_LH[n]) + np.cross(hatY[n,:],hatZ[n,:])*np.sin(theta_LH[n]) + hatY[n,:]*np.dot(hatY[n,:],hatZ[n,:])*(1 - np.cos(theta_LH[n])) for n in range(self.realNfil)])
			# We rotate hatZprime around hatZ by phi using Rodrigues formula
			hatZprime2		= np.array([hatZprime[n,:]*np.cos(phi[n]) + np.cross(hatZ[n,:],hatZprime[n,:])*np.sin(phi[n]) + hatZ[n,:]*np.dot(hatZ[n,:],hatZprime[n,:])*(1 - np.cos(phi[n])) for n in range(self.realNfil)])
			# Now hatZprime2 is the direction of the long axis of the filament
			norm_hatZprime2	= np.linalg.norm(hatZprime2,axis=1)
			# beta angle
			angles[:,1]		= np.arccos(hatZprime2[:,2]/norm_hatZprime2)
			# alpha angle
			angles[:,0]		= np.arctan2(hatZprime2[:,1],hatZprime2[:,0])
			return angles,hatZprime2
	def get_sizes(self):
		# The sizes will be the ellipsoid semi axes a,b,c with a=b<c
		sizes			= np.zeros((self.realNfil,3))
		c_semiaxis		= self.size_scale*(1.0+np.random.pareto(self.slope-1,size=self.realNfil))
		#sizes[:,2]		= np.clip(c_semiaxis,0,self.max_length)
		sizes[:,2]		= c_semiaxis
		sizes[:,0]		= self.size_ratio*sizes[:,2]
		sizes[:,1]		= self.size_ratio*sizes[:,2]
		return sizes

#---------------------------------------------------------------------