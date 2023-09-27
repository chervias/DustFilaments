import argparse
import numpy as np
import healpy as hp
from mpi4py import MPI
from DustFilaments.MagField import get_MagField
from DustFilaments.FilPop import get_FilPop
import time
import yaml
from yaml import Loader

# We need to set the seed for the random generator in gsl

start_time_absolute = time.time_ns()

parser = argparse.ArgumentParser()
parser.add_argument("input", help="indicate yaml file where all the inputs are defined")
args = parser.parse_args()

# read the yaml file
stream = open(args.input, 'r')
dict = yaml.load(stream,Loader=Loader)

freq_array = np.array(dict['freqs'],dtype=np.double)
Nfreqs = len(freq_array)

try:
	seed_population = int(dict['seed_population'])
except KeyError:
	seed_population = 1234

try:
	seed_magfield = int(dict['seed_magfield'])
except KeyError:
	seed_magfield = 2345

try:
	Npixels_magfield = int(dict['Npixels_magfield'])
except KeyError:
	Npixels_magfield = 256

try:
	if dict['skip_Bcube']:
		skip_Bcube = 1
	else:
		skip_Bcube = 0
except KeyError:
	skip_Bcube = 0

try:
	if dict['galactic_plane']:
		galactic_plane = True
	else:
		galactic_plane = False
except KeyError:
	galactic_plane = False

try:
	if dict['null_Gplane']:
		null_Gplane = True
	else:
		null_Gplane = False
except KeyError:
	null_Gplane = False

try:
	if dict['fixed_distance']:
		fixed_distance = True
	else:
		fixed_distance = False
except KeyError:
	fixed_distance = False

try:
	if dict['fixed_size']:
		fixed_size = True
	else:
		fixed_size = False
except KeyError:
	fixed_size = False

try:
	if dict['random_fpol']:
		random_fpol = True
	else:
		random_fpol = False
except KeyError:
	random_fpol = True

try:
	fpol_template = dict['fpol_template']
except KeyError:
	fpol_template = None
    
try:
	if dict['only_path']:
		only_path = True
	else:
		only_path = False
except KeyError:
	only_path = False

try:
	if dict['asymmetry']:
		asymmetry = True
	else:
		asymmetry = False
except KeyError:
	asymmetry = False
try:
	if dict['correct_impossible_angles']:
		correct_impossible_angles = True
	else:
		correct_impossible_angles = False
except KeyError:
	correct_impossible_angles = False

# now make the population object

size_magfield = (Npixels_magfield,Npixels_magfield,Npixels_magfield,3)
Bcube = get_MagField(float(dict['size_box']),Npixels_magfield,seed_magfield,path_cube=str(dict['path_Bcube']),)

# Create the filament population object in rank 0
if asymmetry:
	centers,angles,sizes,psi_LH, psi_LH_random,phi_LH,phi_LH_1,phi_LH_2,theta_LH,thetaH,thetaL,fpol0,beta_array,T_array,final_Nfils, filaments_mask, theta_a, fn_evaluated, mask_fils = get_FilPop(int(dict['Nfil']), float(dict['theta_LH_RMS']), float(dict['size_ratio']), float(dict['size_scale']), float(dict['slope']), float(dict['eta_eps']), float(dict['eta_fpol']), Bcube, float(dict['size_box']), seed_population, float(dict['alpha']), float(dict['beta']), int(dict['nside']), str(dict['beta_template']), str(dict['T_template']), float(dict['ell_limit']), float(dict['sigma_rho']), dust_template=str(dict['dust_template']), mask_file=str(dict['mask_file']), galactic_plane=galactic_plane, null_Gplane=null_Gplane, fixed_distance=fixed_distance, fixed_size=fixed_size, random_fpol=random_fpol, fpol_template=fpol_template, asymmetry=asymmetry, kappa_asymmetry=float(dict['kappa_asymmetry']), lambda_asymmetry=float(dict['lambda_asymmetry']), correct_impossible_angles=correct_impossible_angles)
else:
	centers, angles, sizes, psi_LH, thetaH, thetaL, fpol0, beta_array, T_array, final_Nfils, filaments_mask, theta_a, vecY = get_FilPop(int(dict['Nfil']), float(dict['theta_LH_RMS']), float(dict['size_ratio']), float(dict['size_scale']), float(dict['slope']), float(dict['eta_eps']), float(dict['eta_fpol']), Bcube, float(dict['size_box']), seed_population, float(dict['alpha']), float(dict['beta']), int(dict['nside']), str(dict['beta_template']), str(dict['T_template']), float(dict['ell_limit']), float(dict['sigma_rho']), dust_template=str(dict['dust_template']), mask_file=str(dict['mask_file']), galactic_plane=galactic_plane, null_Gplane=null_Gplane, fixed_distance=fixed_distance, fixed_size=fixed_size, random_fpol=random_fpol, fpol_template=str(dict['fpol_template']))
print('finished with population')