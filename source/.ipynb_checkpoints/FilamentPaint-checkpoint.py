import numpy as np
import healpy as hp
import Bpowspec
from scipy.interpolate import RegularGridInterpolator

class FilPop(object):
	def __init__(self,Nfil,theta_LH_RMS,size_ratio,size_scale,magfield,fixed_distance=False):
		self.Nfil			= Nfil
		self.magfield		= magfield
		self.fixed_distance	= fixed_distance
		self.max_length		= 15.0
		self.size_scale		= size_scale
		self.size_ratio		= size_ratio
		if theta_LH_RMS == None:
			self.theta_LH_RMS	= None
		else:
			self.theta_LH_RMS	= np.radians(theta_LH_RMS)
		self.centers				= self.get_centers()	
		self.angles,self.long_vec	= self.get_angles()
		self.sizes					= self.get_sizes()
	def get_centers(self):
		centers	= np.zeros((self.Nfil,3))
		# recipe to generate random centers
		# Usually this will be done in spherical coordinates
		# the center cannot leave the box within where B is defined
		# maximum radial distance of a center is 0.5*size - 5*max_length
		if self.fixed_distance:
			radii_random	= np.ones(self.Nfil) * 0.4*self.magfield.size
		else:
			radii_random	= np.random.uniform((0.05*self.magfield.size)**3,(0.5*self.magfield.size - 5*self.max_length)**3,self.Nfil)**(1./3.)
		phi_random		= 2*np.pi*np.random.uniform(0.0,1.0,self.Nfil)
		theta_random	= np.arccos(1.0 - 2*np.random.uniform(0.0,1.0,self.Nfil))
		centers[:,0]	= radii_random*np.sin(theta_random)*np.cos(phi_random)
		centers[:,1]	= radii_random*np.sin(theta_random)*np.sin(phi_random)
		centers[:,2]	= radii_random*np.cos(theta_random)
		return centers
	def get_angles(self):
		angles			= np.zeros((self.Nfil,2))		
		long_axis_vec 	= np.zeros((self.Nfil,3))

		# get the euler angles according to the local magnetic field in the center pixel. The hatZ vector of the filament (long axis) follows local B
		local_magfield	= np.array([self.magfield.interp_fn((self.centers[n,0],self.centers[n,1],self.centers[n,2])) for n in range(self.Nfil)])
		if self.theta_LH_RMS == None:
			# unit vector along the local mag field
			hatZ			= np.array([local_magfield[n,:]/np.linalg.norm(local_magfield[n,:]) for n in range(self.Nfil)])
			# alpha angle
			angles[:,1]		= np.arccos(hatZ[:,2])
			# beta angle
			angles[:,0]		= np.arctan2(hatZ[:,1],hatZ[:,0])
			return angles,hatZ
		else:
			# unit vector along the local mag field
			hatZ			= np.array([local_magfield[n,:]/np.linalg.norm(local_magfield[n,:]) for n in range(self.Nfil)])
			# we need a second unit vector hatY perpendicular to hatZ
			vecY			= np.cross(hatZ,np.array([0,1,0]))
			hatY			= np.array([vecY[n,:]/np.linalg.norm(vecY[n,:]) for n in range(self.Nfil)])
			# This is in radians
			theta_LH		= np.fabs(np.random.normal(0,self.theta_LH_RMS,self.Nfil))
			#theta_LH = np.zeros(self.Nfil)
			phi				= np.random.uniform(0,2*np.pi,self.Nfil)
			#phi = np.zeros(self.Nfil)
			# We rotate hatZ around hatY by theta_LH using Rodrigues formula
			hatZprime		= np.array([hatZ[n,:]*np.cos(theta_LH[n]) + np.cross(hatY[n,:],hatZ[n,:])*np.sin(theta_LH[n]) + hatY[n,:]*np.dot(hatY[n,:],hatZ[n,:])*(1 - np.cos(theta_LH[n])) for n in range(self.Nfil)])
			# We rotate hatZprime around hatZ by phi using Rodrigues formula
			hatZprime2		= np.array([hatZprime[n,:]*np.cos(phi[n]) + np.cross(hatZ[n,:],hatZprime[n,:])*np.sin(phi[n]) + hatZ[n,:]*np.dot(hatZ[n,:],hatZprime[n,:])*(1 - np.cos(phi[n])) for n in range(self.Nfil)])
			# Now hatZprime2 is the direction of the long axis of the filament
			norm_hatZprime2	= np.linalg.norm(hatZprime2,axis=1)
			# alpha angle
			angles[:,1]		= np.arccos(hatZprime2[:,2]/norm_hatZprime2)
			# beta angle
			angles[:,0]		= np.arctan2(hatZprime2[:,1],hatZprime2[:,0])
			return angles,hatZprime2
	def get_sizes(self):
		# The sizes will be the ellipsoid semi axes a,b,c with a=b<c
		sizes			= np.zeros((self.Nfil,3))
		c_semiaxis		= self.size_scale*(1.0+np.random.pareto(4.6-1,size=self.Nfil))
		sizes[:,2]		= np.clip(c_semiaxis,0,self.max_length)
		sizes[:,0]		= self.size_ratio*sizes[:,2]
		sizes[:,1]		= self.size_ratio*sizes[:,2]
		return sizes

class MagField(object):
	def __init__(self,size,pixels,seed):
		# size is a physical size
		self.size			= size
		# pixels is the number of pixels on each dimension
		self.pixels			= pixels
		self.seed			= seed
		self.Bcube			= self.get_MagField()
		self.interp_fn		= self.get_interpolator()
	
	def get_MagField(self):
		size3d 					= np.array([self.size,self.size,self.size])
		N 						= np.array([self.pixels,self.pixels,self.pixels],dtype=np.int32)
		Delta 					= size3d/N
		Deltak 					= Bpowspec.Delta2k(Delta,N)
		kmax 					= np.amax(N*Deltak)
		k 						= np.linspace(0,kmax, 500);
		Pk 						= np.exp(-k**2/2/(kmax/40)**2)
		kx,ky,kz 				= Bpowspec.kvecs(N,Deltak)
		Bharmx,Bharmy,Bharmz 	= Bpowspec.Pk2harm(k,Pk,N,kmax,Deltak,self.seed)
		Bcube					= np.zeros((self.pixels,self.pixels,self.pixels,3))
		Bcube[:,:,:,0] 			= Bpowspec.harm2map(Bharmx,Delta)
		Bcube[:,:,:,1] 			= Bpowspec.harm2map(Bharmy,Delta)
		Bcube[:,:,:,2] 			= Bpowspec.harm2map(Bharmz,Delta)
		return Bcube
	
	def get_interpolator(self):
		real_units				= np.linspace(-0.5*self.size,+0.5*self.size,self.pixels)
		interp_fn			 	= RegularGridInterpolator((real_units,real_units,real_units),self.Bcube,method='linear',fill_value=None)
		return interp_fn

class MagField_1fil:
	# This is to test with a single filament
	def __init__(self,size,pixels,seed,direction):
		# size is a physical size
		self.size			= size
		# pixels is the number of pixels on each dimension
		self.pixels			= pixels
		self.direction		= direction
		self.seed			= seed
		self.Bcube			= self.get_MagField()
		self.interp_fn		= self.get_interpolator()
	
	def get_MagField(self):
		Bcube					= np.zeros((self.pixels,self.pixels,self.pixels,3))
		
		if self.direction=='+z':
			Bcube[:,:,:,0]			= np.zeros((self.pixels,self.pixels,self.pixels))
			Bcube[:,:,:,1]			= np.zeros((self.pixels,self.pixels,self.pixels))
			Bcube[:,:,:,2]			= np.ones((self.pixels,self.pixels,self.pixels))
		if self.direction=='-z':
			Bcube[:,:,:,0]			= np.zeros((self.pixels,self.pixels,self.pixels))
			Bcube[:,:,:,1]			= np.zeros((self.pixels,self.pixels,self.pixels))
			Bcube[:,:,:,2]			= -1.0*np.ones((self.pixels,self.pixels,self.pixels))
		if self.direction=='+y':
			Bcube[:,:,:,0]			= np.zeros((self.pixels,self.pixels,self.pixels))
			Bcube[:,:,:,1]			= np.ones((self.pixels,self.pixels,self.pixels))
			Bcube[:,:,:,2]			= np.zeros((self.pixels,self.pixels,self.pixels))
		if self.direction=='-y':
			Bcube[:,:,:,0]			= np.zeros((self.pixels,self.pixels,self.pixels))
			Bcube[:,:,:,1]			= -1*np.ones((self.pixels,self.pixels,self.pixels))
			Bcube[:,:,:,2]			= np.zeros((self.pixels,self.pixels,self.pixels))
		if self.direction=='45deg':
			Bcube[:,:,:,0]			= np.zeros((self.pixels,self.pixels,self.pixels))
			Bcube[:,:,:,1]			= np.ones((self.pixels,self.pixels,self.pixels))
			Bcube[:,:,:,2]			= -1*np.ones((self.pixels,self.pixels,self.pixels))
		return Bcube
	
	def get_interpolator(self):
		real_units				= np.linspace(-0.5*self.size,+0.5*self.size,self.pixels)
		interp_fn			 	= RegularGridInterpolator((real_units,real_units,real_units),self.Bcube,method='linear',fill_value=None)
		return interp_fn

class Sky(object):
	def __init__(self,nside):
		self.nside								= nside
		self.Tmap								= np.zeros(12*self.nside**2)
		self.Qmap								= np.zeros(12*self.nside**2)
		self.Umap								= np.zeros(12*self.nside**2)
		self.mask								= np.zeros(12*self.nside**2)
		self.r_unit_vectors						= self.get_r_unit_vectors()
		self.local_triad						= self.get_local_triad()
			
	def get_r_unit_vectors(self):
		#get the r unit vector for each pixel in a ring pixelization and the angles
		Npix	= 12*self.nside**2
		r_unit_vectors			= np.zeros((Npix,3))
		result 					= hp.pix2vec(self.nside,range(Npix),nest=False)
		r_unit_vectors[:,0]		= np.array(result[0])
		r_unit_vectors[:,1]		= np.array(result[1])
		r_unit_vectors[:,2]		= np.array(result[2])
		return r_unit_vectors
	
	def get_local_triad(self):
		# Return local coordinate system at specified vector nhat, according to Healpix polarization convention
		# note: fails for nhat is zhat0 = North Pole
		Npix				= 12*self.nside**2
		# the indices are (N_pix,Ntriad_vectors,Ndimensions)
		local_triad			= np.zeros((Npix,3,3))
		nhat				= self.r_unit_vectors
		zhat0 				= np.array([0,0,1])
		local_triad[:,2,:] 	= nhat
		tmp 				= -zhat0 + np.einsum('i,ij->ij',np.dot(nhat,zhat0),nhat)
		local_triad[:,0,:] 	= tmp / np.linalg.norm(tmp,axis=1)[:,None]
		local_triad[:,1,:] 	= np.cross(local_triad[:,2],local_triad[:,0])
		return local_triad
		
	def save_sky(self,name):
		hp.write_map(name,[self.Tmap,self.Qmap,self.Umap],nest=False,overwrite=True)