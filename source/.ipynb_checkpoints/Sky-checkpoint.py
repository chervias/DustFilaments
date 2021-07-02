import numpy as np
import healpy as hp

def get_r_unit_vectors(nside):
	#get the r unit vector for each pixel in a ring pixelization and the angles
	Npix	= 12*nside**2
	r_unit_vectors			= np.zeros((Npix,3))
	result 					= hp.pix2vec(nside,range(Npix),nest=False)
	r_unit_vectors[:,0]		= np.array(result[0])
	r_unit_vectors[:,1]		= np.array(result[1])
	r_unit_vectors[:,2]		= np.array(result[2])
	return r_unit_vectors
	
def get_local_triad(nside,r_unit_vectors):
	# Return local coordinate system at specified vector nhat, according to Healpix polarization convention
	# note: fails for nhat is zhat0 = North Pole
	Npix				= 12*nside**2
	# the indices are (N_pix,Ntriad_vectors,Ndimensions)
	local_triad			= np.zeros((Npix,3,3))
	nhat				= r_unit_vectors
	zhat0 				= np.array([0,0,1])
	local_triad[:,2,:] 	= nhat
	tmp 				= -zhat0 + np.einsum('i,ij->ij',np.dot(nhat,zhat0),nhat)
	local_triad[:,0,:] 	= tmp / np.linalg.norm(tmp,axis=1)[:,None]
	local_triad[:,1,:] 	= np.cross(local_triad[:,2],local_triad[:,0])
	return local_triad