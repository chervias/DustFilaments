from DustFilaments.FilamentPaint import Get_Angles, Get_Angles_Asymmetry, Reject_Big_Filaments#, Permutations
import numpy as np
import healpy as hp
from scipy.stats import norm, laplace_asymmetric
#import os

H_PLANCK =  6.6260755e-34
K_BOLTZ = 1.380658e-23
T_CMB =  2.72548

def get_centers(galactic_plane,null_Gplane,fixed_distance,dust_template,nside,Nfil,size,mask_file):
	# recipe to generate random centers
	# Usually this will be done in spherical coordinates
	if galactic_plane:
		# Now the centers will be defined by a template map
		# first normalize the map to the total number of filaments
		if dust_template is not None:
			map_original = hp.read_map(dust_template,field=0,dtype=np.double)
			# make sure there are no 0s in this map
			map_original[map_original < 0.0] = 0.0
			map_nside = hp.ud_grade(map_original,nside)
		else:
			print('You must provide a dust_template if you want galactic_plane=True ')
			exit()
		if null_Gplane:
			if mask_file is not None:
				mask = hp.read_map(mask_file,field=0,dtype=np.double)
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
		indices = np.zeros((real_number),dtype=int)
		counter = 0
		for n in range(12 * nside **2):
			if number_fil[n] > 0:
				indices[counter:counter+number_fil[n]] = int(n)
				counter += number_fil[n]
			else:
				continue
		if not fixed_distance:
			l_rand	 = np.random.uniform(0.15,1.0,real_number)
			centers = (0.45*size)*l_rand**(1.0/3.0) * np.array(hp.pix2vec(nside, indices, nest=False))
		else:
			centers = 0.4 * size * np.array(hp.pix2vec(nside, indices, nest=False))
		print('Real number of filaments is ',real_number,'counter',counter)
		return real_number, np.ascontiguousarray(centers.T,dtype=np.double)
	else:
		centers = np.zeros((Nfil,3),dtype=np.double)
		if not fixed_distance:
			l_rand	 = np.random.uniform(0.1,1.0,Nfil)
			u_rand   = np.random.uniform(-1.0,1.0,Nfil)
			phi_rand = np.random.uniform(0.0,2.0*np.pi,Nfil)
			centers[:,0] = (0.45*size)*l_rand**(1.0/3.0)*np.sqrt(1. - u_rand**2)*np.cos(phi_rand)
			centers[:,1] = (0.45*size)*l_rand**(1.0/3.0)*np.sqrt(1. - u_rand**2)*np.sin(phi_rand)
			centers[:,2] = (0.45*size)*l_rand**(1.0/3.0)*u_rand
		else:
			u_rand   = np.random.uniform(-1.0,1.0,Nfil)
			phi_rand = np.random.uniform(0.0,2.0*np.pi,Nfil)
			centers[:,0] = (0.4*size) * np.sqrt(1. - u_rand**2)*np.cos(phi_rand)
			centers[:,1] = (0.4*size) * np.sqrt(1. - u_rand**2)*np.sin(phi_rand)
			centers[:,2] = (0.4*size) * u_rand
		return int(Nfil), np.ascontiguousarray(centers,dtype=np.double)

def get_sizes(Nfil,fixed_size,size_scale,slope,eta_eps,size_ratio):
	# The sizes will be the ellipsoid semi axes a,b,c with a=b<c
	sizes			= np.zeros((Nfil,3))
	if fixed_size:
		c_semiaxis = size_scale*np.ones(Nfil)
	else:
		c_semiaxis = size_scale*(1.0+np.random.pareto(slope-1.0,size=Nfil))
	sizes[:,2]		= c_semiaxis
	sizes[:,0]	= size_ratio*(sizes[:,2]/size_scale) ** eta_eps * sizes[:,2]
	sizes[:,1]	= size_ratio*(sizes[:,2]/size_scale) ** eta_eps * sizes[:,2]
	return sizes

def get_fpol(alpha,beta,Nfil,sizes,size_scale,nside,centers,eta_fpol,random=True,fpol_template=None):
	if random:
		# create a beta distribution for fpol0
		return np.ascontiguousarray(np.random.beta(alpha,beta,size=Nfil)*(sizes[:,2]/size_scale) ** eta_fpol, dtype=np.double)
	else:
		if fpol_template is not None:
			# load the map
			fpol_map = hp.read_map(fpol_template,field=0,dtype=np.double)
			pixels = hp.vec2pix(nside,centers[:,0],centers[:,1],centers[:,2])
			return np.ascontiguousarray(fpol_map[pixels],dtype=np.double)
		else:
			print('You must supply a template for the polarization fraction, exiting')
			exit()

def dust_sed_Jysr(nu,beta,Tdust):
	x = H_PLANCK*nu*1.e9/(K_BOLTZ*Tdust)
	return (nu*1e9)**(beta+3.0)/(np.exp(x)-1.0)

def get_beta_T(beta_template,T_template,nside,Nfil,centers,sigma_rho):
	# load maps from data
	beta_map_original = hp.read_map(beta_template,field=(0,1),dtype=np.double)
	beta_map_nside = hp.ud_grade(beta_map_original,nside)
	T_map_original = hp.read_map(T_template,field=(0,1),dtype=np.double)
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
def get_FilPop(Nfil,theta_LH_RMS,size_ratio,size_scale,slope,eta_eps,eta_fpol,Bcube,size,seed,alpha,beta,nside,beta_template,T_template,ell_limit,sigma_rho,dust_template=None,mask_file=None,fixed_distance=False,fixed_size=False,galactic_plane=False,null_Gplane=False,random_fpol=True,fpol_template=None, asymmetry=False, asymmetry_method=None, lambda_asymmetry=1.0, kappa_asymmetry=1.0, center_asymmetry=0.0, sigma_asymmetry=1.0 ):
    Npix_box = int(Bcube.shape[0])
    max_length		= 1.0
    np.random.seed(seed)
    #os.environ["GSL_RNG_SEED"] = str(seed)
    theta_LH_RMS_radians = np.radians(theta_LH_RMS)
    realNfils, centers = get_centers(galactic_plane,null_Gplane,fixed_distance,dust_template,nside,Nfil,size,mask_file) # this method will determine the total number of filaments, which is different if we do the poisson thing
    sizes = get_sizes(realNfils,fixed_size,size_scale,slope,eta_eps,size_ratio)
    fpol0 = get_fpol(alpha,beta,realNfils,sizes,size_scale,nside,centers,eta_fpol,random_fpol,fpol_template)
    beta_array,T_array = get_beta_T(beta_template,T_template,nside,realNfils,centers,sigma_rho)
    if asymmetry:
        # random_vectors will be the small vector x pointing down in the triad system
        #colat, long = hp.vec2ang(centers)
        #random_vectors = np.array([np.cos(colat)*np.cos(long), np.cos(colat)*np.sin(long), -np.sin(colat)]).T
        u_rand   = np.random.uniform(-1.0,1.0,realNfils)
        phi_rand = np.random.uniform(0.0,2.0*np.pi,realNfils)
        random_vectors = np.zeros((realNfils,3))
        random_vectors[:,0] = np.sqrt(1. - u_rand**2)*np.cos(phi_rand)
        random_vectors[:,1] = np.sqrt(1. - u_rand**2)*np.sin(phi_rand)
        random_vectors[:,2] = u_rand
        # now we use a normal and a asymmetric laplace
        N_good = 0 ; psi_final = [] ; theta_final = []
        while(N_good <= realNfils):
            U = np.random.uniform(0.0,1.0,realNfils)
            #U2 = rng.uniform(0.0,1.0,N)
            theta_LH = norm.ppf(U, loc=0.0, scale=theta_LH_RMS_radians)
            if asymmetry_method == 'ALD' or asymmetry_method == None:
                psi_LH_random = laplace_asymmetric.ppf(U, kappa_asymmetry, loc=0.0, scale=lambda_asymmetry**-1 ) 
            elif asymmetry_method == 'norm':
                psi_LH_random = norm.ppf(U, loc=np.radians(center_asymmetry), scale=np.radians(sigma_asymmetry) )
            theta_LH = np.array(sorted(theta_LH,key=np.fabs))
            psi_LH_random = np.array(sorted(psi_LH_random,key=np.fabs))
            # these are the values I want to keep
            mask_ = np.fabs(psi_LH_random) < np.fabs(theta_LH)
            psi_final.append(psi_LH_random[mask_])
            theta_final.append(theta_LH[mask_])
            N_good += np.sum(mask_)
        psi_LH_random = np.concatenate(psi_final)
        theta_LH = np.concatenate(theta_final)
        p = np.random.permutation(len(theta_LH))
        theta_LH = theta_LH[p] ; psi_LH_random = psi_LH_random[p]
        theta_LH = theta_LH[:realNfils] ; psi_LH_random = psi_LH_random[:realNfils]
        #print("For fil 4461901 the angles are psi=%.8E theta=%.8E"%(psi_LH_random[4461901],theta_LH[4461901]))
        print("shape of indices of imposible angles = ",np.argwhere(np.fabs(psi_LH_random) > np.fabs(theta_LH)).shape)
        results = Get_Angles_Asymmetry( realNfils, Bcube, Npix_box, random_vectors, psi_LH_random, theta_LH, size, centers , theta_LH_RMS_radians)
        (angles, long_vec, psi_LH, phi_LH, phi_LH_1, phi_LH_2, thetaH, thetaL, fn_evaluated) = results
    else:
        u_rand   = np.random.uniform(-1.0,1.0,realNfils)
        phi_rand = np.random.uniform(0.0,2.0*np.pi,realNfils)
        random_vectors = np.zeros((realNfils,3))
        random_vectors[:,0] = np.sqrt(1. - u_rand**2)*np.cos(phi_rand)
        random_vectors[:,1] = np.sqrt(1. - u_rand**2)*np.sin(phi_rand)
        random_vectors[:,2] = u_rand
        theta_LH		= np.fabs(np.random.normal(loc=0,scale=theta_LH_RMS_radians,size=(realNfils)))
        random_azimuth = np.random.uniform(0.0,2.0*np.pi,size=(realNfils))
        results = Get_Angles( realNfils, Bcube, Npix_box, random_vectors, random_azimuth, theta_LH, size, centers , theta_LH_RMS_radians)
        (angles, long_vec, psi_LH, thetaH, thetaL, vecY) = results
    mask,theta_a = Reject_Big_Filaments(sizes, thetaL, size_ratio, centers, realNfils, ell_limit, size_scale, eta_eps)
    final_Nfils = int(realNfils)
    if asymmetry:
        mask_fils = np.fabs(theta_LH) >= np.fabs(psi_LH)
        return centers, angles, sizes, psi_LH, psi_LH_random, phi_LH, phi_LH_1, phi_LH_2, theta_LH, thetaH, thetaL, fpol0, beta_array, T_array, final_Nfils, mask, theta_a, fn_evaluated, mask_fils
    else:
        return centers, angles, sizes, psi_LH, thetaH, thetaL, fpol0, beta_array, T_array, final_Nfils, mask, theta_a, vecY

