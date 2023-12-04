import argparse
import numpy as np
import healpy as hp
from mpi4py import MPI
from DustFilaments.MagField import get_MagField
from DustFilaments.FilPop import get_FilPop
from DustFilaments.FilamentPaint import Paint_Filament_Shells
import time
import yaml
from yaml import Loader

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
    if dict['asymmetry_method']=='ALD':
        asymmetry_method = 'ALD'
    elif dict['asymmetry_method']=='norm':
        asymmetry_method = 'norm'
    else:
        asymmetry_method = None
except KeyError:
    asymmetry_method = None

try:
	if dict['fpol_template']:
		fpol_template = dict['fpol_template']
	else:
		fpol_template = None
except KeyError:
	fpol_template = None

shared_comm = MPI.COMM_WORLD
size_pool = shared_comm.Get_size()
rank = shared_comm.Get_rank()

start_time_first = time.time_ns()

if rank==0:
    output_tqumap = '%s/DustFilaments_TQU_%s'%(str(dict['outdir']),str(dict['label']))
    output_tshells = '%s/DustFilaments_T_shells_%s'%(str(dict['outdir']),str(dict['label']))

size_magfield = (Npixels_magfield,Npixels_magfield,Npixels_magfield,3)

if rank == 0:
	Bcube = get_MagField(float(dict['size_box']),Npixels_magfield,seed_magfield,path_cube=str(dict['path_Bcube']),only_path=only_path)
else:
	Bcube = np.empty(size_magfield,dtype=np.double)
shared_comm.Bcast(Bcube, root=0)
shared_comm.Barrier()
if rank==0:
    if asymmetry:
        if asymmetry_method=='ALD':
        	centers,angles,sizes,psi_LH, psi_LH_random,phi_LH,phi_LH_1,phi_LH_2,theta_LH,thetaH,thetaL,fpol0,beta_array,T_array,final_Nfils, filaments_mask, theta_a, fn_evaluated, mask_fils = get_FilPop(int(dict['Nfil']), float(dict['theta_LH_RMS']), float(dict['size_ratio']), float(dict['size_scale']), float(dict['slope']), float(dict['eta_eps']), float(dict['eta_fpol']), Bcube, float(dict['size_box']), seed_population, float(dict['alpha']), float(dict['beta']), int(dict['nside']), str(dict['beta_template']), str(dict['T_template']), float(dict['ell_limit']), float(dict['sigma_rho']), dust_template=str(dict['dust_template']), mask_file=str(dict['mask_file']), galactic_plane=galactic_plane, null_Gplane=null_Gplane, fixed_distance=fixed_distance, fixed_size=fixed_size, random_fpol=random_fpol, fpol_template=fpol_template, asymmetry=asymmetry, asymmetry_method=asymmetry_method, kappa_asymmetry=float(dict['kappa_asymmetry']), lambda_asymmetry=float(dict['lambda_asymmetry']),)
        elif asymmetry_method=='norm':
            centers,angles,sizes,psi_LH, psi_LH_random,phi_LH,phi_LH_1,phi_LH_2,theta_LH,thetaH,thetaL,fpol0,beta_array,T_array,final_Nfils, filaments_mask, theta_a, fn_evaluated, mask_fils = get_FilPop(int(dict['Nfil']), float(dict['theta_LH_RMS']), float(dict['size_ratio']), float(dict['size_scale']), float(dict['slope']), float(dict['eta_eps']), float(dict['eta_fpol']), Bcube, float(dict['size_box']), seed_population, float(dict['alpha']), float(dict['beta']), int(dict['nside']), str(dict['beta_template']), str(dict['T_template']), float(dict['ell_limit']), float(dict['sigma_rho']), dust_template=str(dict['dust_template']), mask_file=str(dict['mask_file']), galactic_plane=galactic_plane, null_Gplane=null_Gplane, fixed_distance=fixed_distance, fixed_size=fixed_size, random_fpol=random_fpol, fpol_template=fpol_template, asymmetry=asymmetry, asymmetry_method=asymmetry_method, center_asymmetry=float(dict['center_asymmetry']), sigma_asymmetry=float(dict['sigma_asymmetry']),)
        else:
            exit('No asymmetry method was chosen, exiting.')
    else:
    	centers, angles, sizes, psi_LH, thetaH, thetaL, fpol0, beta_array, T_array, final_Nfils, filaments_mask, theta_a, vecY = get_FilPop(int(dict['Nfil']), float(dict['theta_LH_RMS']), float(dict['size_ratio']), float(dict['size_scale']), float(dict['slope']), float(dict['eta_eps']), float(dict['eta_fpol']), Bcube, float(dict['size_box']), seed_population, float(dict['alpha']), float(dict['beta']), int(dict['nside']), str(dict['beta_template']), str(dict['T_template']), float(dict['ell_limit']), float(dict['sigma_rho']), dust_template=str(dict['dust_template']), mask_file=str(dict['mask_file']), galactic_plane=galactic_plane, null_Gplane=null_Gplane, fixed_distance=fixed_distance, fixed_size=fixed_size, random_fpol=random_fpol, fpol_template=fpol_template)
    print('finished with population')
    # I define the bins for the shells
    dist_all = np.linalg.norm(centers,axis=1)
    shells_radii = np.linspace(dist_all.min(),dist_all.max()*(1.01),21,dtype=np.double)
    print('finished with population, the distance are min=%.2f max=%.2f'%(dist_all.min(),dist_all.max()))
else:
    shells_radii = np.empty(21,dtype=np.double)

# broadcast the radii for shells
shared_comm.Bcast(shells_radii, root=0)
Nshells = 21 - 1

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
    #print('the shapes are ',shapes)
else:
	shapes = None
shapes = shared_comm.bcast(shapes, root=0)

print('rank %i will run %i filaments'%(rank,shapes[rank]))

if rank==0:
    # I go over each rank from 1 to (poolsize-1)
    for rank_id in range(1,size_pool):
        # this mask is the indices of the filaments that correspond to rank=rank_id
        mask = fils_indices_split[rank_id]
        #print('I am rank 0 and I will send to rank %i the T_array with length %i'%(rank_id,len(T_array[mask])))
        shared_comm.Send([fils_indices_split[rank_id], MPI.INT], dest=rank_id, tag=1) # indices
        shared_comm.Send([centers[mask], MPI.DOUBLE], dest=rank_id, tag=2) # centers
        shared_comm.Send([sizes[mask], MPI.DOUBLE], dest=rank_id, tag=3) # sizes
        shared_comm.Send([angles[mask], MPI.DOUBLE], dest=rank_id, tag=4) # angles
        shared_comm.Send([fpol0[mask], MPI.DOUBLE], dest=rank_id, tag=5) # fpol0
        shared_comm.Send([thetaH[mask], MPI.DOUBLE], dest=rank_id, tag=6) # thetaH
        shared_comm.Send([beta_array[mask], MPI.DOUBLE], dest=rank_id, tag=7) # beta_dust
        shared_comm.Send([T_array[mask], MPI.DOUBLE], dest=rank_id, tag=8) # Tdust
        #print('here')
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

    #print('I am rank %i and I will receive from rank 0 the T_array with length %i'%(rank,len(T_array_rank)))

    shared_comm.Recv([fils_indices_split_rank, MPI.INT], source=0, tag=1)
    shared_comm.Recv([centers_rank, MPI.DOUBLE], source=0, tag=2)
    shared_comm.Recv([sizes_rank, MPI.DOUBLE], source=0, tag=3)
    shared_comm.Recv([angles_rank, MPI.DOUBLE], source=0, tag=4)
    shared_comm.Recv([fpol0_rank, MPI.DOUBLE], source=0, tag=5)
    shared_comm.Recv([thetaH_rank, MPI.DOUBLE], source=0, tag=6)
    shared_comm.Recv([beta_array_rank, MPI.DOUBLE], source=0, tag=7)
    shared_comm.Recv([T_array_rank, MPI.DOUBLE], source=0, tag=8)
    shared_comm.Recv([theta_a_rank, MPI.DOUBLE], source=0, tag=9)
    #print('rank %i received all arrays'%rank)

if True:
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
    nside_moving = int(dict['nside'])
    while nside_moving >= 128:
        tqu_total[nside_moving] = np.ascontiguousarray(np.zeros((Nfreqs,3,12*nside_moving**2),dtype=np.double))
        # this way you get 2048 at the beginning, then 1024, then 512, etc. all the way to 128
        nside_moving = int(0.5*nside_moving)
    # this t_total must be an array, with one dictionary for each shell, where the keys are the Nsides from 32 to nside
    t_total = [None] * Nshells
    for i_shell in range(Nshells):
        t_total[i_shell] = {}
        nside_moving = int(dict['nside'])
        while nside_moving >= 128:
            t_total[i_shell][nside_moving] = np.ascontiguousarray(np.zeros((Nfreqs,12*nside_moving**2),dtype=np.double))
            # this way you get 2048 at the beginning, then 1024, then 512, etc. all the way to 128
            nside_moving = int(0.5*nside_moving)

    first_part_time = time.time_ns() - start_time_first

    counter = 0
    rank_time = 0.0

    for n_rank,n_general in enumerate(fils_indices_split_rank):
        # we need to decide what is the size of the a filament. We will decide it is 2*Theta_a, which is one total length (remember that La and Lb are semi-axes)
        if np.pi/theta_a_rank[n_rank] < float(dict['ell_limit']):
            continue
        try:
            time_start = time.time_ns()
            #determine to which shell the filament belongs to
            distance_to_fil = np.linalg.norm(centers_rank[n_rank])
            i_fil_shell = 0
            while not ( distance_to_fil>=shells_radii[i_fil_shell] and distance_to_fil<shells_radii[i_fil_shell+1] ):
                i_fil_shell += 1
                if i_fil_shell == Nshells:
                    print('Something is wrong !!!!!')
            Paint_Filament_Shells(n_rank,int(dict['nside']),sizes_rank,centers_rank,angles_rank,float(fpol0_rank[n_rank]),float(thetaH_rank[n_rank]),float(beta_array_rank[n_rank]),float(T_array_rank[n_rank]),Bcube,float(dict['size_box']),Npixels_magfield,int(dict['resolution_low']),int(dict['resolution_high']),freq_array,Nfreqs,tqu_total,t_total[i_fil_shell],skip_Bcube,rank)
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
        nside_moving = int(dict['nside'])
        while nside_moving >= 128:
            tqu_final[nside_moving] = np.zeros((Nfreqs,3,12*nside_moving**2),dtype=np.double)
            # this way you get 2048 at the beginning, then 1024, then 512, etc. all the way to 128
            nside_moving = int(0.5*nside_moving)
        
        # t_final must also be a dict like t_total
        t_final = [None] * Nshells
        for i_shell in range(Nshells):
            t_final[i_shell] = {}
            nside_moving = int(dict['nside'])
            while nside_moving >= 128:
                t_final[i_shell][nside_moving] = np.zeros((Nfreqs,12*nside_moving**2),dtype=np.double)
                # this way you get 2048 at the beginning, then 1024, then 512, etc. all the way to 128
                nside_moving = int(0.5*nside_moving)
    else:
        # tqu_final must also be a dict like tqu_total
        tqu_final = {}
        nside_moving = int(dict['nside'])
        while nside_moving >= 128:
            tqu_final[nside_moving] = None
            # this way you get 2048 at the beginning, then 1024, then 512, etc. all the way to 128
            nside_moving = int(0.5*nside_moving)
        # t_final must also be a dict like t_total
        t_final = [None] * Nshells
        for i_shell in range(Nshells):
            t_final[i_shell] = {}
            nside_moving = int(dict['nside'])
            while nside_moving >= 128:
                t_final[i_shell][nside_moving] = None
                # this way you get 2048 at the beginning, then 1024, then 512, etc. all the way to 128
                nside_moving = int(0.5*nside_moving)
    # put a barrier to make sure all processeses are finished
    shared_comm.Barrier()
    # use MPI to get the totals, but I need to loop over the tqu_total dict, to reduce into the tqu_final map
    for nside_var,tqumap in tqu_total.items():
        # we reduce into the rank=0 process
        shared_comm.Reduce([tqumap, MPI.DOUBLE],[tqu_final[nside_var], MPI.DOUBLE],op = MPI.SUM,root = 0)
    # use MPI to get the totals, but I need to loop over the t_total dict, to reduce into the t_final map
    for i_shell in range(Nshells):
        for nside_var,tqumap in t_total[i_shell].items():
            # we reduce into the rank=0 process
            shared_comm.Reduce([tqumap, MPI.DOUBLE],[t_final[i_shell][nside_var], MPI.DOUBLE],op = MPI.SUM,root = 0)

    # reduce the counter of skipped filaments
    counter_total = shared_comm.reduce(counter,op=MPI.SUM,root=0)
    # reduce the total time running on FilamentPaint.Paint_Filament
    time_total = shared_comm.reduce(rank_time,op=MPI.SUM,root=0)

    if rank==0:
        # I need to upgrade the resolution for every nside between 128 and nside
        for n in range(Nfreqs):
            tqu_final_final = np.zeros((3,12*int(dict['nside'])**2),dtype=np.double)
            for nside_var,tqumap in tqu_final.items():
                if nside_var == int(dict['nside']):
                    # in this case only sum into tqu_final_final, do not upgrade
                    # I keep a copy of this map also
                    tqu_final_final += tqumap[n]
                else:
                    # we have to upgrade the resolution in harmonic space
                    print('Im upgrading the resolution in harm space at nside=%i for frequency %.1f'%(nside_var,freq_array[n]))
                    tqumap_ring = hp.reorder(tqumap[n],n2r=True)
                    (almT, almE, almB) = hp.map2alm(tqumap_ring,pol=True,lmax=(3*nside_var))
                    fwhm = np.sqrt(np.pi/3.0)/nside_var
                    almT_conv = hp.almxfl(almT, hp.gauss_beam(fwhm,lmax=(4*int(dict['nside']))), inplace=False)
                    almE_conv = hp.almxfl(almE, hp.gauss_beam(fwhm,lmax=(4*int(dict['nside']))), inplace=False)
                    almB_conv = hp.almxfl(almB, hp.gauss_beam(fwhm,lmax=(4*int(dict['nside']))), inplace=False)
                    tqumap_nside = hp.alm2map((almT_conv,almE_conv,almB_conv),int(dict['nside']),pol=True)
                    tqumap_nside_nest = hp.reorder(tqumap_nside,r2n=True)
                    tqu_final_final += tqumap_nside_nest
            # I save the final map at freq nn
            hp.write_map(output_tqumap+'_f%s.fits'%str(freq_array[n]).replace('.','p'),tqu_final_final,nest=True,overwrite=True,dtype=np.single)
        # I need to upgrade the resolution for every nside between 128 and nside
        for n in range(Nfreqs):
            for i_shell in range(Nshells):
                t_final_final = np.zeros((12*int(dict['nside'])**2),dtype=np.double)
                for nside_var,tqumap in t_final[i_shell].items():
                    if nside_var == int(dict['nside']):
                        # in this case only sum into t_final_final, do not upgrade
                        # I keep a copy of this map also
                        t_final_final += tqumap[n]
                    else:
                        # we have to upgrade the resolution in harmonic space
                        print('Im upgrading the resolution in harm space at nside=%i for frequency %.1f'%(nside_var,freq_array[n]))
                        tqumap_ring = hp.reorder(tqumap[n],n2r=True)
                        almT = hp.map2alm(tqumap_ring,pol=False,lmax=(3*nside_var))
                        fwhm = np.sqrt(np.pi/3.0)/nside_var
                        almT_conv = hp.almxfl(almT, hp.gauss_beam(fwhm,lmax=(4*int(dict['nside']))), inplace=False)
                        tqumap_nside = hp.alm2map((almT_conv),int(dict['nside']),pol=False)
                        tqumap_nside_nest = hp.reorder(tqumap_nside,r2n=True)
                        t_final_final += tqumap_nside_nest
                # I save the final map at freq nn and shell i_shell
                hp.write_map(output_tqumap+'_f%s_RadShell%02i.fits'%(str(freq_array[n]).replace('.','p'),i_shell),t_final_final,nest=True,overwrite=True,dtype=np.single)
        second_part_time = time.time_ns() - start_time_second

        #f = open(output_params_file,'a') # I append the times
        print('The total number of skipped filaments is %i\n'%(counter_total))
        print('The time for the first part is %f s\n'%(first_part_time/1e9))
        print('The total time running is %f s\n'%(time_total/1e3))
        print('The total time running (per filament) is %f ms\n'%(time_total/final_Nfils))
        print('The time for the second part is %f s\n'%(second_part_time/1e9))
        print('The total time of execution is %f s\n'%((time.time_ns() - start_time_absolute)/1e9))
        #f.close()

