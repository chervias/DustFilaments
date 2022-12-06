from Bpowspec import Pk2harm,harm2map,Delta2k
import numpy as np

def get_MagField(size,pixels=256,seed=1,path_cube=None,only_path=False,rms_random=3.0,):
	# size is the size of the box
	# pixels is the number of pixels per side in the cube
	# seed is the random seed
	# path_cube, if provided, must contain a npz file with the large-scale B cube, with the label Bcube. This must have uG units
	# only_path if True, then you only want the Bcube in path to be passed to the filament painting, bypassing the entire process of generating a correlated magnetic field
	# rms_random is the rms of the isotropic random component we want to generate
	if path_cube is not None:
		Bcube_ls = np.load(path_cube)['Hcube'] # this is in [iz,iy,ix] order
		assert pixels == Bcube_ls.shape[0], 'The large-scale Hcube is not %i^3 pixels'%pixels
	size3d 					= np.array([size,size,size])
	N 						= np.array([pixels,pixels,pixels],dtype=np.int32)
	Delta 					= size3d/N
	Deltak 					= Delta2k(Delta,N)
	kmax 					= np.amax(N*Deltak)
	k 						= np.linspace(0, kmax, 1000);
	#Pk = k**-4
	#k_knee = 1.0/100
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
	constant = rms_random / rms  # this is 3 because we want to make the rms = 3 uG
	if path_cube is not None:
		if only_path == False:
			return np.ascontiguousarray(Bcube*constant + Bcube_ls)
		else:
			return np.ascontiguousarray( Bcube_ls )
	else:
		return np.ascontiguousarray(Bcube*constant)