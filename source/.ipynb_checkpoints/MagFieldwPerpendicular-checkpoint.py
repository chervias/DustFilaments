import Bpowspec
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class MagField:
	def __init__(self,size,pixels,seed):
		# size is a physical size
		self.size			= size
		# pixels is the number of pixels on each dimension
		self.pixels			= pixels
		self.seed			= seed
		self.Bcube			= self.get_MagField()
		self.interp_fn		= self.get_interpolator()
	
	def get_MagField(self):
		# WARNING !!!!!!!
		# Kevin's code will output a B cube with indices Bcube[iz,iy,ix], so we have to transpose to put it in the format
		# [ix,iy,iz]
		# !!!!!!!!!!!!!!!
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
		Bcube[:,:,:,0] 			= np.transpose(Bpowspec.harm2map(Bharmx,Delta),axes=(2,1,0))
		Bcube[:,:,:,1] 			= np.transpose(Bpowspec.harm2map(Bharmy,Delta),axes=(2,1,0))
		Bcube[:,:,:,2] 			= np.transpose(Bpowspec.harm2map(Bharmz,Delta),axes=(2,1,0))
		# I will put the mag field to be the hat phi vector from spherical coordinates
		Bcube2					= np.zeros((self.pixels,self.pixels,self.pixels,3))
		for i in range(self.pixels):
			for j in range(self.pixels):
				for k in range(self.pixels):
					# vector in real coordinates
					X = self.size*i/(self.pixels-1.0) - 0.5*self.size
					Y = self.size*j/(self.pixels-1.0) - 0.5*self.size
					Z = self.size*k/(self.pixels-1.0) - 0.5*self.size
					r = np.array([X,Y,Z])
					rhat = r/np.linalg.norm(r)
					proj_B = Bcube[i,j,k] - np.dot(Bcube[i,j,k],rhat)*rhat
					Bcube2[i,j,k,:] = proj_B
		return Bcube2

	def get_interpolator(self):
		real_units				= np.linspace(-0.5*self.size,+0.5*self.size,self.pixels)
		interp_fn			 	= RegularGridInterpolator((real_units,real_units,real_units),self.Bcube,method='linear',fill_value=None)
		return interp_fn