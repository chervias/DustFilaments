from DustFilaments.FilamentPaint import Get_Angles,Reject_Big_Filaments
import numpy as np
import healpy as hp

H_PLANCK =  6.6260755e-34
K_BOLTZ = 1.380658e-23
T_CMB =  2.72548

def get_centers(galactic_plane,null_Gplane,fixed_distance,dust_template,nside,Nfil,size,mask_file):
	# recipe to generate random centers
	# Usually this will be done in spherical coordinates
	if galactic_plane:
		# Now the centers will be defined by a template map
		# first normalize the map to the total number of filaments
		map_original = hp.read_map(dust_template,field=0)
		# make sure there are no 0s in this map
		map_original[map_original < 0.0] = 0.0
		map_nside = hp.ud_grade(map_original,nside)
		if null_Gplane:
			if mask_file not None:
				mask = hp.read_map(mask_file,field=0) # we will degrade the mask to 512
				mask = hp.ud_grade(mask,nside)
			else:
				print('You must provide a mask if you want to null the Galactic plane')
				exit()
			# you only want to mask the pixels that are = 0, everything >0 should be brought to 1
			# because of this we use ceil
			mask = np.ceil(mask)
			map_nside = map_nside * mask
		# each pixel is mult by C --> Nfil = C*Sum(map)
		C = Nfil / np.sum(map_nside)
		number_fil = np.random.poisson(C*map_nside,12*nside**2)
		real_number = int(np.sum(number_fil))
		centers	= np.zeros((real_number,3),dtype=np.double)
		l_rand	 = np.random.uniform(0.15,1.0,real_number)
		radii_arr = (0.45*size)*l_rand**(1.0/3.0)
		counter = 0
		for n in range(12*nside**2):
			if number_fil[n] > 0:
				for k in range(number_fil[n]):
					centers[counter,:] = radii_arr[counter] * np.array(hp.pix2vec(nside,n,nest=False))
					counter += 1 
			else:
				# continue since pixel n does not have any filament
				continue
		print('Real number of filaments is ',real_number,'counter',counter)
		return real_number, np.ascontiguousarray(centers,dtype=np.double)
	elif fixed_distance:
		centers	= np.zeros((Nfil,3),dtype=np.double)
		radii_random	= np.ones(Nfil) * 0.4*size
		phi_random		= 2*np.pi*np.random.uniform(0.0,1.0,Nfil)
		theta_random	= np.arccos(1.0 - 2*np.random.uniform(0.0,1.0,Nfil))
		centers[:,0]	= radii_random*np.sin(theta_random)*np.cos(phi_random)
		centers[:,1]	= radii_random*np.sin(theta_random)*np.sin(phi_random)
		centers[:,2]	= radii_random*np.cos(theta_random)
		return int(Nfil), np.ascontiguousarray(centers,dtype=np.double)
	else:
		centers = np.zeros((Nfil,3),dtype=np.double)
		l_rand	 = np.random.uniform(0.1,1.0,Nfil)
		u_rand   = np.random.uniform(-1.0,1.0,Nfil)
		phi_rand = np.random.uniform(0.0,2.0*np.pi,Nfil)
		centers[:,0] = (0.45*size)*l_rand**(1.0/3.0)*np.sqrt(1. - u_rand**2)*np.cos(phi_rand)
		centers[:,1] = (0.45*size)*l_rand**(1.0/3.0)*np.sqrt(1. - u_rand**2)*np.sin(phi_rand)
		centers[:,2] = (0.45*size)*l_rand**(1.0/3.0)*u_rand
		return int(Nfil), np.ascontiguousarray(centers,dtype=np.double)

def get_sizes(Nfil,fixed_size,size_scale,slope,size_ratio):
	# The sizes will be the ellipsoid semi axes a,b,c with a=b<c
	sizes			= np.zeros((Nfil,3))
	if fixed_size:
		c_semiaxis = size_scale*np.ones(Nfil)
	else:
		c_semiaxis = size_scale*(1.0+np.random.pareto(slope-1.0,size=Nfil))
	sizes[:,2]		= c_semiaxis
	# the size_ratio has a La^0.1/(size_scale)^0.1 dependance
	sizes[:,0]		= size_ratio*(sizes[:,2]/size_scale) ** +0.122 * sizes[:,2]
	sizes[:,1]		= size_ratio*(sizes[:,2]/size_scale) ** +0.122 * sizes[:,2]
	return sizes

def get_fpol(alpha,beta,Nfil,sizes,size_scale):
	# create a beta distribution for fpol0
	return np.ascontiguousarray(np.random.beta(alpha,beta,size=Nfil)*(sizes[:,2]/size_scale) ** -0.1,dtype=np.double)

def dust_sed_Jysr(nu,beta,Tdust):
	x = H_PLANCK*nu*1.e9/(K_BOLTZ*Tdust)
	return (nu*1e9)**(beta+3.0)/(np.exp(x)-1.0)

def get_beta_T(beta_template,T_template,nside,Nfil,centers,sigma_rho):
	# load maps from data
	beta_map_original = hp.read_map(beta_template,field=(0,1))
	beta_map_nside = hp.ud_grade(beta_map_original,nside)
	T_map_original = hp.read_map(T_template,field=(0,1))
	T_map_nside = hp.ud_grade(T_map_original,nside)
	# These are the pixels where the centers of each filaments are
	pixels = hp.vec2pix(nside,centers[:,0],centers[:,1],centers[:,2])
	# New method
	# From Pelgrims et al 2021
	# Eq 14 and 15
	random_rho = np.random.normal(loc=0.0,scale=sigma_rho,size=Nfil)
	# random rho CANNOT go under -1, if it does then beta_dust = nan
	random_rho = np.clip(random_rho,-0.99,None)
	alpha_per_fil = dust_sed_Jysr(217.,beta_map_nside[0][pixels],T_map_nside[0][pixels]) / dust_sed_Jysr(353.,beta_map_nside[0][pixels],T_map_nside[0][pixels])
	x_217_per_fil = H_PLANCK*217*1.e9/(K_BOLTZ*T_map_nside[0][pixels])
	x_353_per_fil = H_PLANCK*353*1.e9/(K_BOLTZ*T_map_nside[0][pixels])
	beta_array = np.log(alpha_per_fil*(1+random_rho)*(np.exp(x_217_per_fil)-1.0)/(np.exp(x_353_per_fil)-1.0))/np.log(217.0/353.0) - 3.0
	# Fixed T from Planck fit
	T_array = T_map_nside[0][pixels]
	return np.ascontiguousarray(beta_array),np.ascontiguousarray(T_array)

def get_FilPop(Nfil,theta_LH_RMS,size_ratio,size_scale,slope,Bcube,size,seed,alpha,beta,nside,dust_template,beta_template,T_template,ell_limit,sigma_rho,Nthreads=8,mask_file=None,fixed_distance=False,fixed_size=False,galactic_plane=False,null_Gplane=False):
	Npix_box = int(Bcube.shape[0])
	max_length		= 1.0
	nu_index = +0.122
	np.random.seed(seed)
	theta_LH_RMS_radians = np.radians(theta_LH_RMS)
	realNfils, centers = get_centers(galactic_plane,null_Gplane,fixed_distance,dust_template,nside,Nfil,size,mask_file) # this method will determine the total number of filaments, which is different if we do the poisson thing
	# get angles now is on C
	random_vectors  = np.zeros((realNfils,3)) ; random_vectors[:,0] = 1.0 
	theta_LH		= np.fabs(np.random.normal(loc=0,scale=theta_LH_RMS_radians,size=(realNfils)))
	random_azimuth = np.random.uniform(0.0,2.0*np.pi,size=(realNfils))
	results = Get_Angles( realNfils, Bcube, Npix_box, random_vectors, random_azimuth, theta_LH, size, centers , theta_LH_RMS_radians, Nthreads)
	(angles, long_vec, psi_LH, thetaH, thetaL) = results
	sizes = get_sizes(realNfils,fixed_size,size_scale,slope,size_ratio)
	print("I have calculated the centers, angles, sizes")
	mask,theta_a = Reject_Big_Filaments(sizes, thetaL, size_ratio, centers, realNfils, ell_limit, Nthreads, size_scale, nu_index)
	fpol0 = get_fpol(alpha,beta,realNfils,sizes,size_scale)
	beta_array,T_array = get_beta_T(beta_template,T_template,nside,realNfils,centers,sigma_rho)
	final_Nfils = int(realNfils)
	return centers, angles, sizes, psi_LH, thetaH, thetaL, fpol0, beta_array, T_array, final_Nfils, mask, theta_a