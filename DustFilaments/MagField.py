from Bpowspec import Pk2harm,harm2map,Delta2k
import numpy as np

def get_MagField(size,pixels,seed,method,path):
	# the size has to be in kpc
	if method==1:
		Bcube_ls = np.load(path)['Bcube'] # this is in [iz,iy,ix] order
		size3d 					= np.array([size,size,size])
		N 						= np.array([pixels,pixels,pixels],dtype=np.int32)
		Delta 					= size3d/N
		Deltak 					= Delta2k(Delta,N)
		kmax 					= np.amax(N*Deltak)
		k 						= np.linspace(0,kmax, 1000);
		# I create a P(k)
		#filter = 0.5+0.5*np.tanh(0.015*(k - 900))
		#Pk_small = 1e0*k**-4*filter
		#Pk_small[0] = 0.0
		# This is for the cube with 20kpc per side
		filter =  0.5+0.5*np.tanh(1.0*(k - 15))
		Pk_small = 6e-3*k**-4*filter
		Pk_small[0] = 0.0
		Bharmx,Bharmy,Bharmz 	= Pk2harm(k,Pk_small,N,kmax,Deltak,seed)
		Bcube					= np.zeros((pixels,pixels,pixels,3),dtype=np.double)
		Bcube[:,:,:,0] 			= harm2map(Bharmx,Delta) + Bcube_ls[:,:,:,0]
		Bcube[:,:,:,1] 			= harm2map(Bharmy,Delta) + Bcube_ls[:,:,:,1]
		Bcube[:,:,:,2] 			= harm2map(Bharmz,Delta) + Bcube_ls[:,:,:,2]
		return np.ascontiguousarray(Bcube)
	if method==2:
		size3d 					= np.array([size,size,size])
		N 						= np.array([pixels,pixels,pixels],dtype=np.int32)
		Delta 					= size3d/N
		Deltak 					= Delta2k(Delta,N)
		kmax 					= np.amax(N*Deltak)
		k 						= np.linspace(0,kmax, 500);
		#Pk 						= np.exp(-k**2/2/(kmax/40)**2)
		Pk 						= np.exp(-k**2/2/(kmax/100)**2)
		Bharmx,Bharmy,Bharmz 	= Pk2harm(k,Pk,N,kmax,Deltak,seed)
		Bcube					= np.zeros((pixels,pixels,pixels,3))
		Bcube[:,:,:,0] 			= harm2map(Bharmx,Delta)
		Bcube[:,:,:,1] 			= harm2map(Bharmy,Delta)
		Bcube[:,:,:,2] 			= harm2map(Bharmz,Delta)
		return np.ascontiguousarray(Bcube)
	if method==3:
		size3d 					= np.array([size,size,size])
		N 						= np.array([pixels,pixels,pixels],dtype=np.int32)
		Delta 					= size3d/N
		Deltak 					= Delta2k(Delta,N)
		kmax 					= np.amax(N*Deltak)
		k 						= np.linspace(0,kmax, 1000);
		import healpy as hp
		beam = hp.gauss_beam(np.radians(0.35),lmax=999,pol=False)
		Pk = k**-4*beam**2
		Pk[0] = 0.0
		Pk[-1] = 0.0
		Bharmx,Bharmy,Bharmz 	= Pk2harm(k,Pk,N,kmax,Deltak,seed)
		Bcube					= np.zeros((pixels,pixels,pixels,3),dtype=np.double)
		Bx = harm2map(Bharmx,Delta) ; By = harm2map(Bharmy,Delta) ; Bz = harm2map(Bharmz,Delta)
		#print("New: Bx ",Bx[2,1,0])
		Bcube[:,:,:,0] 			= Bx
		Bcube[:,:,:,1] 			= By
		Bcube[:,:,:,2] 			= Bz
		return np.ascontiguousarray(Bcube)
	if method == 4:
		# This will be the Jansson & Farrar model as the background, plus random P(k) = k**-4, normalized such RMS(B) = 3 uG
		Bcube_ls = np.load(path)['Bcube'] # this is in [iz,iy,ix] order, the file must be /home/chervias/CMB-work/Filaments/3dfilament-project-healpix/Bcube_Galactic_large-scale_100pc_JnFmodel_Npix256.npz
		size3d 					= np.array([size,size,size])
		N 						= np.array([pixels,pixels,pixels],dtype=np.int32)
		Delta 					= size3d/N
		Deltak 					= Delta2k(Delta,N)
		kmax 					= np.amax(N*Deltak)
		k 						= np.linspace(0, kmax, 1000);
		Pk = k**-4
		Pk[0] = 0.0
		Pk[-1] = 0.0
		Bharmx,Bharmy,Bharmz 	= Pk2harm(k,Pk,N,kmax,Deltak,seed)
		Bcube					= np.zeros((pixels,pixels,pixels,3))
		Bcube[:,:,:,0] 			= harm2map(Bharmx,Delta)
		Bcube[:,:,:,1] 			= harm2map(Bharmy,Delta)
		Bcube[:,:,:,2] 			= harm2map(Bharmz,Delta)
		# rms
		rms = np.sqrt(np.average(np.linalg.norm(Bcube,axis=3)**2))
		constant = 3.0 / rms  # this is 3 because we want to make the rms = 3 uG
		return np.ascontiguousarray(Bcube*constant + Bcube_ls)
	if method == 5:
		# This will a white spectrum magnetci field
		size3d 					= np.array([size,size,size])
		N 						= np.array([pixels,pixels,pixels],dtype=np.int32)
		Delta 					= size3d/N
		Deltak 					= Delta2k(Delta,N)
		kmax 					= np.amax(N*Deltak)
		k 						= np.linspace(0,kmax, 1000);
		Pk = k**0.0 # 1 at all k's, except the first and last
		Pk[0] = 0.0
		Pk[-1] = 0.0
		Bharmx,Bharmy,Bharmz 	= Pk2harm(k,Pk,N,kmax,Deltak,seed)
		Bcube					= np.zeros((pixels,pixels,pixels,3),dtype=np.double)
		Bcube[:,:,:,0] 			= harm2map(Bharmx,Delta) 
		Bcube[:,:,:,1] 			= harm2map(Bharmy,Delta) 
		Bcube[:,:,:,2] 			= harm2map(Bharmz,Delta)
		return np.ascontiguousarray(Bcube)