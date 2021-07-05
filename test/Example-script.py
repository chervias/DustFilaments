import argparse
import numpy as np
import healpy as hp
from mpi4py import MPI
from DustFilaments.MagField import get_MagField
from DustFilaments.FilPop import get_FilPop
from DustFilaments.FilamentPaint import Paint_Filament
import time
import yaml

start_time_absolute = time.time_ns()

parser = argparse.ArgumentParser()
parser.add_argument("input", help="indicate yaml file where all the inputs are defined")
args = parser.parse_args()

# read the yaml file
stream = open(args.input, 'r')
dictionary = yaml.load(stream)
for key, value in dictionary.items():
	if key == 'nside':
		nside = int(value)
	else:
		nside = 2048
	if key == 'theta_LH_RMS':
		theta_LH_RMS = float(value)
	if key == 'size_scale':
		size_scale = float(value)
	if key == 'size_ratio':
		size_ratio = float(value)
	if key == 'slope':
		slope = float(value)
	if key == 'Nfil':
		Nfil = int(value)
	if key == 'size_box':
		size_box = float(value)
	if key == 'label':
		label = str(value)
	if key == 'resolution':
		resolution = int(value)
	if key == 'alpha':
		alpha = float(value)
	if key == 'beta':
		beta = float(value)
	if key == 'freqs':
		freqs = np.array(value,dtype=np.double) 
	if key == 'ell_limit':
		ell_limit = float(value)
	if key == 'sigma_rho':
		sigma_rho = float(value)
	if key == 'Nthreads':
		Nthreads = int(value)
	else:
		Nthreads = 8
	if key == 'seed_population':
		seed_population = int(value)
	if key == 'seed_magfield':
		seed_magfield = int(value)
	if key == 'Npixels_magfield':
		Npixels_magfield = int(value)
	else:
		Npixels_magfield = 256
	if key == 'skip_Bcube':
		if value:
			skip_Bcube = 1
		else:
			skip_Bcube = 0
	if key == 'method':
		method = int(value)
	if key == 'mask_file':
		mask_file = str(value)
	if key == 'dust_template':
		dust_template = str(value)
	if key == 'beta_template':
		beta_template = str(value)
	if key == 'T_template':
		T_template = str(value)
	if key == 'path_Bcube':
		path_Bcube = str(value)
	if key == 'outdir':
		outdir = str(value)
	if key == 'galactic_plane':
		galactic_plane = value
	else:
		galactic_plane = False
	if key == 'null_Gplane':
		null_Gplane = value
	else:
		null_Gplane = False
	if key == 'fixed_distance':
		fixed_distance = value
	else:
		fixed_distance = False
	if key == 'fixed_size':
		fixed_size = value
	else:
		fixed_size = False

shared_comm = MPI.COMM_WORLD
size_pool = shared_comm.Get_size()
rank = shared_comm.Get_rank()

start_time_first = time.time_ns()

if rank==0:
	output_tqumap	= '%s/tqumap_ns%s_%s_maa%s_sr%s_sl%s_min-size%s_ell-limit%i_%s'%(outdir,str(nside),str(Nfil),str(theta_LH_RMS).replace('.','p'),str(size_ratio).replace('.','p'),str(slope).replace('.','p'),str(size_scale).replace('.','p'),int(ell_limit),label)

size_magfield = (Npixels_magfield,Npixels_magfield,Npixels_magfield,3)

if rank == 0:
	Bcube = get_MagField(size_box,Npixels_magfield,seed_magfield,path_Bcube)
else:
	Bcube = np.empty(size_magfield,dtype=np.double)
shared_comm.Bcast(Bcube, root=0)

shared_comm.Barrier()

if rank==0:
	# Create the filament population object in rank 0
	centers,angles,sizes,psi_LH,thetaH,thetaL,fpol0,beta_array,T_array,final_Nfils,filaments_mask,theta_a = get_FilPop(Nfil,theta_LH_RMS,size_ratio,size_scale,slope,Bcube,size_box,seed_population,alpha,beta,nside,beta_template,T_template,ell_limit,sigma_rho,Nthreads=int(Nthreads),dust_template=dust_template,mask_file=mask_file,galactic_plane=galactic_plane,null_Gplane=null_Gplane,fixed_distance=fixed_distance,fixed_size=fixed_size)
	print('finish with population')
# Now I split the total number of filaments in the ranks
if rank == 0:
	# create the numpy array of ints
	# make sure it is C-contiguous
	fils_indices = np.arange(0,final_Nfils,dtype='int32')
	fils_indices_split = np.array_split(fils_indices, size_pool)
	shapes = []
	for rank_id in range(size_pool):
		shapes.append(len(fils_indices_split[rank_id]))
	# I need to broadcast the shape to all ranks
else:
	shapes = None
shapes = shared_comm.bcast(shapes, root=0)

if rank==0:
	# I go over each rank from 1 to (poolsize-1)
	for rank_id in range(1,size_pool):
		# this mask is the indices of the filaments that correspond to rank=rank_id
		mask = fils_indices_split[rank_id]
		shared_comm.Send([fils_indices_split[rank_id], MPI.INT], dest=rank_id, tag=1) # indices
		shared_comm.Send([centers[mask], MPI.DOUBLE], dest=rank_id, tag=2) # centers
		shared_comm.Send([sizes[mask], MPI.DOUBLE], dest=rank_id, tag=3) # sizes
		shared_comm.Send([angles[mask], MPI.DOUBLE], dest=rank_id, tag=4) # angles
		shared_comm.Send([fpol0[mask], MPI.DOUBLE], dest=rank_id, tag=5) # fpol0
		shared_comm.Send([thetaH[mask], MPI.DOUBLE], dest=rank_id, tag=6) # thetaH
		shared_comm.Send([beta_array[mask], MPI.DOUBLE], dest=rank_id, tag=7) # beta_dust
		shared_comm.Send([T_array[mask], MPI.DOUBLE], dest=rank_id, tag=8) # Tdust
		shared_comm.Send([theta_a[mask], MPI.DOUBLE], dest=rank_id, tag=9) # theta_a
	# finally I need to put the sub-arrays in new names
	fils_indices_split_rank = fils_indices_split[rank]
	centers_rank = centers[fils_indices_split_rank]
	sizes_rank = sizes[fils_indices_split_rank]
	angles_rank = angles[fils_indices_split_rank]
	fpol0_rank = fpol0[fils_indices_split_rank]
	thetaH_rank = thetaH[fils_indices_split_rank]
	beta_array_rank = beta_array[fils_indices_split_rank]
	T_array_rank = T_array[fils_indices_split_rank]
	theta_a_rank = theta_a[fils_indices_split_rank]
else:
	fils_indices_split_rank = np.empty(shapes[rank], dtype='int32')
	centers_rank = np.empty((shapes[rank],3), dtype='float64')
	sizes_rank = np.empty((shapes[rank],3), dtype='float64')
	angles_rank = np.empty((shapes[rank],2), dtype='float64')
	fpol0_rank = np.empty(shapes[rank], dtype='float64')
	thetaH_rank = np.empty(shapes[rank], dtype='float64')
	beta_array_rank = np.empty(shapes[rank], dtype='float64')
	T_array_rank = np.empty(shapes[rank], dtype='float64')
	theta_a_rank = np.empty(shapes[rank], dtype='float64')
	
	shared_comm.Recv([fils_indices_split_rank, MPI.INT], source=0, tag=1)
	shared_comm.Recv([centers_rank, MPI.DOUBLE], source=0, tag=2)
	shared_comm.Recv([sizes_rank, MPI.DOUBLE], source=0, tag=3)
	shared_comm.Recv([angles_rank, MPI.DOUBLE], source=0, tag=4)
	shared_comm.Recv([fpol0_rank, MPI.DOUBLE], source=0, tag=5)
	shared_comm.Recv([thetaH_rank, MPI.DOUBLE], source=0, tag=6)
	shared_comm.Recv([beta_array_rank, MPI.DOUBLE], source=0, tag=7)
	shared_comm.Recv([T_array_rank, MPI.DOUBLE], source=0, tag=8)
	shared_comm.Recv([theta_a_rank, MPI.DOUBLE], source=0, tag=9)

# Check the lengths
if len(fils_indices_split_rank) != shapes[rank]:
	print('Lengths do not match, existing')
	exit(1)
if centers_rank.shape[0] != shapes[rank]:
	print('Lengths do not match, existing')
	exit(1)
if sizes_rank.shape[0] != shapes[rank]:
	print('Lengths do not match, existing')
	exit(1)
if angles_rank.shape[0] != shapes[rank]:
	print('Lengths do not match, existing')
	exit(1)
if len(fpol0_rank) != shapes[rank]:
	print('Lengths do not match, existing')
	exit(1)
if len(thetaH_rank) != shapes[rank]:
	print('Lengths do not match, existing')
	exit(1)
if len(beta_array_rank) != shapes[rank]:
	print('Lengths do not match, existing')
	exit(1)
if len(T_array_rank) != shapes[rank]:
	print('Lengths do not match, existing')
	exit(1)
if len(theta_a_rank) != shapes[rank]:
	print('Lengths do not match, existing')
	exit(1)

shared_comm.Barrier()

# this tqu_total must be a dictionary, where the keys are the Nsides from 32 to nside
tqu_total = {}
nside_moving = int(nside)
while nside_moving >= 128:
	tqu_total[nside_moving] = np.ascontiguousarray(np.zeros((len(freqs),3,12*nside_moving**2),dtype=np.double))
	# this way you get 2048 at the beginning, then 1024, then 512, etc. all the way to 128
	nside_moving = int(0.5*nside_moving)

first_part_time = time.time_ns() - start_time_first

counter = 0
rank_time = 0.0

for n_rank,n_general in enumerate(fils_indices_split_rank):
	# we need to decide what is the size of the a filament. We will decide it is 2*Theta_a, which is one total length (remember that La and Lb are semi-axes)
	if np.pi/theta_a_rank[n_rank] < ell_limit:
		continue
	try:
		time_start = time.time_ns()
		Paint_Filament(n_rank,nside,sizes_rank,centers_rank,angles_rank,float(fpol0_rank[n_rank]),float(thetaH_rank[n_rank]),float(beta_array_rank[n_rank]),float(T_array_rank[n_rank]),Bcube,size_box,Npixels_magfield,resolution,freqs,len(freqs),tqu_total,Nthreads,skip_Bcube,rank)
		time_end = time.time_ns()
		# divide by 1e6 to transform to ms
		rank_time += (time_end-time_start)/1e6
	except:
		print("Rank:%i, Error in filament rank index=%i general index=%i, skipping"%(rank,n_rank,n_general))
		counter += 1
		continue

start_time_second = time.time_ns()
if rank==0:
	# only processor 0 will actually get the data
	# tqu_final must also be a dict like tqu_total
	tqu_final = {}
	nside_moving = int(nside)
	while nside_moving >= 128:
		tqu_final[nside_moving] = np.zeros((len(freqs),3,12*nside_moving**2),dtype=np.double)
		# this way you get 2048 at the beginning, then 1024, then 512, etc. all the way to 128
		nside_moving = int(0.5*nside_moving)
else:
	# tqu_final must also be a dict like tqu_total
	tqu_final = {}
	nside_moving = int(nside)
	while nside_moving >= 128:
		tqu_final[nside_moving] = None
		# this way you get 2048 at the beginning, then 1024, then 512, etc. all the way to 128
		nside_moving = int(0.5*nside_moving)
# put a barrier to make sure all processeses are finished
shared_comm.Barrier()
# use MPI to get the totals, but I need to loop over the tqu_total dict, to reduce into the tqu_final map
for nside_var,tqumap in tqu_total.items():
	# we reduce into the rank=0 process
	shared_comm.Reduce([tqumap, MPI.DOUBLE],[tqu_final[nside_var], MPI.DOUBLE],op = MPI.SUM,root = 0)

# reduce the counter of skipped filaments
counter_total = shared_comm.reduce(counter,op=MPI.SUM,root=0)
# reduce the total time running on FilamentPaint.Paint_Filament
time_total = shared_comm.reduce(rank_time,op=MPI.SUM,root=0)

if rank==0:
	# I need to upgrade the resolution for every nside between 128 and nside
	for n in range(len(freqs)):
		tqu_final_final = np.zeros((3,12*nside**2),dtype=np.double)
		for nside_var,tqumap in tqu_final.items():
			if nside_var == nside:
				# in this case only sum into tqu_final_final, do not upgrade
				# I keep a copy of this map also
				tqu_final_final += tqumap[n]
			else:
				# we have to upgrade the resolution in harmonic space
				print('Im upgrading the resolution in harm space at nside=%i for frequency %.1f'%(nside_var,freqs[n]))
				tqumap_ring = hp.reorder(tqumap[n],n2r=True)
				(almT, almE, almB) = hp.map2alm(tqumap_ring,pol=True,lmax=(3*nside_var))
				fwhm = np.sqrt(np.pi/3.0)/nside_var
				almT_conv = hp.almxfl(almT, hp.gauss_beam(fwhm,lmax=(4*nside)), inplace=False)
				almE_conv = hp.almxfl(almE, hp.gauss_beam(fwhm,lmax=(4*nside)), inplace=False)
				almB_conv = hp.almxfl(almB, hp.gauss_beam(fwhm,lmax=(4*nside)), inplace=False)
				tqumap_nside = hp.alm2map((almT_conv,almE_conv,almB_conv),nside,pol=True)
				tqumap_nside_nest = hp.reorder(tqumap_nside,r2n=True)
				tqu_final_final += tqumap_nside_nest
		# I save the final map at freq nn
		hp.write_map(output_tqumap+'_f%s.fits'%str(freqs[n]).replace('.','p'),tqu_final_final,nest=True,overwrite=True,dtype=np.double)
	second_part_time = time.time_ns() - start_time_second
	
	#f = open(output_params_file,'a') # I append the times
	#f.write('The total number of skipped filaments is %i\n'%(counter_total))
	#f.write('The time for the first part is %f s\n'%(first_part_time/1e9))
	#f.write('The total time running is %f s\n'%(time_total/1e3))
	#f.write('The total time running (per filament) is %f ms\n'%(time_total/final_Nfils))
	#f.write('The time for the second part is %f s\n'%(second_part_time/1e9))
	#f.write('The total time of execution is %f s\n'%((time.time_ns() - start_time_absolute)/1e9))
	#f.close()
