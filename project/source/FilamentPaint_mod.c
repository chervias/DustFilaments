#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <numpy/ndarrayobject.h>
#include <omp.h>

#include <FilamentPaint.h>

#define H_PLANCK 6.6260755e-34
#define K_BOLTZ 1.380658e-23
#define T_CMB 2.72548
#define PI 3.14159265358979323846

static PyObject *Reject_Big_Filaments(PyObject *self, PyObject *args){
	PyObject *sizes = NULL;
	PyObject *thetaL = NULL;
	PyObject *size_ratio = NULL;
	PyObject *centers = NULL;
	PyObject *Nfil = NULL;
	PyObject *ell_limit = NULL;
	PyObject *size_scale = NULL;
	PyObject *nu_index = NULL;
	if (!PyArg_ParseTuple(args, "OOOOOOOO",&sizes, &thetaL, &size_ratio, &centers, &Nfil, &ell_limit, &size_scale, &nu_index)) 
		return NULL;
	
	int n;
	
	int Nfil_ = (int) PyLong_AsLong(Nfil);
	double size_ratio_ = PyFloat_AsDouble(size_ratio);
	double size_scale_ = PyFloat_AsDouble(size_scale);
	double ell_limit_ = PyFloat_AsDouble(ell_limit);
	double nu_index_ = PyFloat_AsDouble(nu_index);
	
	double *centers_ptr = PyArray_DATA(centers);
	double *sizes_ptr = PyArray_DATA(sizes);
	double *thetaL_ptr = PyArray_DATA(thetaL);
	
	long *mask = calloc(Nfil_,sizeof(long));
	double *theta_a = calloc(Nfil_,sizeof(double));
	
	#pragma omp parallel for schedule(static)
	for (n=0;n<Nfil_;n++){
		double centers_mod = 0.0 ;
		for(int j=0;j<3;j++) centers_mod += pow(centers_ptr[n*3+j],2) ;
		double size_ratio_new = size_ratio_ * pow( sizes_ptr[n*3+2] / size_scale_ , nu_index_) ;
		//#pragma omp critical
		{
			// La goes with the sin, Lb goes with the cos
			theta_a[n] = 2.0 * sqrt(pow(sizes_ptr[n*3+2]*sin(atan(tan(thetaL_ptr[n])/size_ratio_new)),2) + pow(sizes_ptr[n*3+1]*cos(atan(tan(thetaL_ptr[n])/size_ratio_new)),2))/sqrt(centers_mod) ;
		}
	}
	
	// mask
	npy_intp npy_shape_mask[1] = {Nfil_};
	PyObject *arr_mask = PyArray_SimpleNewFromData(1,npy_shape_mask, NPY_INT, mask);
	PyArray_ENABLEFLAGS((PyArrayObject *)arr_mask, NPY_OWNDATA);
	
	// theta_a
	npy_intp npy_shape_theta_a[1] = {Nfil_};
	PyObject *arr_theta_a = PyArray_SimpleNewFromData(1,npy_shape_theta_a, NPY_DOUBLE, theta_a);
	PyArray_ENABLEFLAGS((PyArrayObject *)arr_theta_a, NPY_OWNDATA);
	
	PyObject *rtrn = PyTuple_New(2);
	PyTuple_SetItem(rtrn, 0, arr_mask);
	PyTuple_SetItem(rtrn, 1, arr_theta_a);
	return(rtrn);
}

static PyObject *Get_Angles(PyObject *self, PyObject *args){
	/* Getting the elements */
	PyObject *Nfil = NULL ;
	PyObject *Bcube = NULL ;
	PyObject *Npix_box = NULL;
	PyObject *random_vector = NULL;
	PyObject *random_azimuth = NULL;
	PyObject *random_thetaLH = NULL;
	PyObject *size = NULL;
	PyObject *centers = NULL;
	PyObject *theta_LH_RMS = NULL;
	if (!PyArg_ParseTuple(args, "OOOOOOOOO",&Nfil, &Bcube, &Npix_box, &random_vector, &random_azimuth, &random_thetaLH, &size, &centers , &theta_LH_RMS ))
		return NULL;
	int Nfil_ = (int) PyLong_AsLong(Nfil);
	int Npix_box_ = (int) PyLong_AsLong(Npix_box);
	
	// The Bcube will follow the ordering of Kevin's code [iz,iy,ix,]
	double *Bcube_ptr = PyArray_DATA(Bcube);
	double *centers_ptr = PyArray_DATA(centers);
	double *random_vector_ptr = PyArray_DATA(random_vector);
	double *random_azimuth_ptr = PyArray_DATA(random_azimuth);
	double *random_thetaLH_ptr = PyArray_DATA(random_thetaLH);
	
	double size_ = PyFloat_AsDouble(size);
	double theta_LH_RMS_ = PyFloat_AsDouble(theta_LH_RMS);
	
	// define the angles array to output
	double *angles = calloc(Nfil_*2,sizeof(double));
	double *hatZprime2 = calloc(Nfil_*3,sizeof(double));
	double *psi_LH = calloc(Nfil_,sizeof(double));
	double *thetaH = calloc(Nfil_,sizeof(double));
	double *thetaL = calloc(Nfil_,sizeof(double));
	
	#pragma omp parallel for schedule(static)
	for(int n=0;n<Nfil_;n++){
		double local_magfield[3], center[3] ;
		int j;
		for(j=0;j<3;j++) center[j] = centers_ptr[n*3 + j];
		FilamentPaint_TrilinearInterpolation(Bcube_ptr, size_, Npix_box_, center, local_magfield) ;
		double local_magfield_mod = 0.0;
		for(j=0;j<3;j++) local_magfield_mod += pow(local_magfield[j],2);
		if (theta_LH_RMS_ == 0.0){
			//#pragma omp critical
			{
				for(j=0;j<3;j++) hatZprime2[n*3+j] = local_magfield[j] / sqrt(local_magfield_mod) ;
				angles[n*2 + 1] = acos(hatZprime2[n*3+2]);
				angles[n*2 + 0] = atan2(hatZprime2[n*3+1],hatZprime2[n*3+0]);
			}
		}
		else{
			double centers_mod = 0.0;
			double hatZ[3], rhat[3], local_B_proj[3], vecY[3], hatY[3], hatZprime[3], filament_vec_proj[3] ;
			for(j=0;j<3;j++) hatZ[j] = local_magfield[j] / sqrt(local_magfield_mod) ;
			
			for(j=0;j<3;j++) centers_mod += pow(center[j],2) ;
			for(j=0;j<3;j++) rhat[j] = center[j] / sqrt(centers_mod) ;
			
			double dot_product = 0.0 ;
			for(j=0;j<3;j++) dot_product += local_magfield[j] * rhat[j] ;
			for(j=0;j<3;j++) local_B_proj[j] = local_magfield[j]  - dot_product * rhat[j] ;
			// this is cross product 
			vecY[0] = (hatZ[1]*random_vector_ptr[n*3+2] - hatZ[2]*random_vector_ptr[n*3+1]) ;
			vecY[1] = (hatZ[2]*random_vector_ptr[n*3+0] - hatZ[0]*random_vector_ptr[n*3+2]) ;
			vecY[2] = (hatZ[0]*random_vector_ptr[n*3+1] - hatZ[1]*random_vector_ptr[n*3+0]) ;
			double vecY_mod = 0.0;
			for(j=0;j<3;j++) vecY_mod += pow(vecY[j],2) ;
			for(j=0;j<3;j++) hatY[j] = vecY[j] / sqrt(vecY_mod) ;
			// rotate hatZ around hatY by theta_LH using Rodrigues formula
			double cross_product[3] ;
			cross_product[0] = (hatY[1]*hatZ[2] - hatY[2]*hatZ[1]) ;
			cross_product[1] = (hatY[2]*hatZ[0] - hatY[0]*hatZ[2]) ;
			cross_product[2] = (hatY[0]*hatZ[1] - hatY[1]*hatZ[0]) ;
			dot_product = 0.0 ;
			for(j=0;j<3;j++) dot_product += hatY[j]*hatZ[j] ;
			for(j=0;j<3;j++) hatZprime[j] = hatZ[j]*cos(random_thetaLH_ptr[n]) + cross_product[j]*sin(random_thetaLH_ptr[n]) + hatY[j]*dot_product*(1.0 - cos(random_thetaLH_ptr[n])) ;
			
			// We rotate hatZprime around hatZ by phi using Rodrigues formula
			cross_product[0] = (hatZ[1]*hatZprime[2] - hatZ[2]*hatZprime[1]) ;
			cross_product[1] = (hatZ[2]*hatZprime[0] - hatZ[0]*hatZprime[2]) ;
			cross_product[2] = (hatZ[0]*hatZprime[1] - hatZ[1]*hatZprime[0]) ;
			dot_product = 0.0 ;
			for(j=0;j<3;j++) dot_product += hatZ[j]*hatZprime[j] ;
			//#pragma omp critical
			{
				for(j=0;j<3;j++) hatZprime2[n*3+j] = hatZprime[j]*cos(random_azimuth_ptr[n]) + cross_product[j]*sin(random_azimuth_ptr[n]) + hatZ[j]*dot_product*(1.0 - cos(random_azimuth_ptr[n])) ;
			}
			// project the vector along the long axis of the filament towards rhat
			dot_product = 0.0 ;
			for(j=0;j<3;j++) dot_product += hatZprime2[n*3+j]*rhat[j] ;
			for(j=0;j<3;j++) filament_vec_proj[j] = hatZprime2[n*3+j] - dot_product * rhat[j] ;
			
			cross_product[0] = (filament_vec_proj[1]*local_B_proj[2] - filament_vec_proj[2]*local_B_proj[1]) ;
			cross_product[1] = (filament_vec_proj[2]*local_B_proj[0] - filament_vec_proj[0]*local_B_proj[2]) ;
			cross_product[2] = (filament_vec_proj[0]*local_B_proj[1] - filament_vec_proj[1]*local_B_proj[0]) ;
			dot_product = 0.0 ;
			for(j=0;j<3;j++) dot_product += rhat[j]*cross_product[j] ;
			double dot_product2 = 0.0 ;
			for(j=0;j<3;j++) dot_product2 += filament_vec_proj[j]*local_B_proj[j];
			//#pragma omp critical
			{
				psi_LH[n] = atan2(dot_product,dot_product2) ;
			}
			
			dot_product = 0.0 ;
			for(j=0;j<3;j++) dot_product += local_magfield[j] * rhat[j] ;
			//#pragma omp critical
			{
				thetaH[n] = acos( dot_product / sqrt(local_magfield_mod) ) ;
			}
			
			dot_product = 0.0 ;
			for(j=0;j<3;j++) dot_product += hatZprime2[n*3+j] * rhat[j] ;
			double hatZprime2_norm = 0.0;
			for(j=0;j<3;j++) hatZprime2_norm += pow(hatZprime2[n*3+j],2) ;
			//#pragma omp critical
			{
				thetaL[n] = acos( dot_product / sqrt(hatZprime2_norm) ) ;
				angles[n*2 + 0] = atan2(hatZprime2[n*3+1],hatZprime2[n*3+0]) ;
				angles[n*2 + 1] = acos(hatZprime2[n*3+2] / sqrt(hatZprime2_norm) ) ;
			}
		}
	}
	
	// return the arrays angles, hatZprime2 , psi_LH , thetaH , thetaL
	
	// angles
	npy_intp npy_shape_angles[2] = {Nfil_,2};
	PyObject *arr_angles = PyArray_SimpleNewFromData(2,npy_shape_angles, NPY_DOUBLE, angles);
	
	// hatZprime2
	npy_intp npy_shape_hatZprime2[2] = {Nfil_,3};
	PyObject *arr_hatZprime2 = PyArray_SimpleNewFromData(2,npy_shape_hatZprime2, NPY_DOUBLE, hatZprime2);
	
	// psi_LH 
	npy_intp npy_shape_psi_LH[1] = {Nfil_};
	PyObject *arr_psi_LH = PyArray_SimpleNewFromData(1,npy_shape_psi_LH, NPY_DOUBLE, psi_LH);
	
	// thetaH 
	npy_intp npy_shape_thetaH[1] = {Nfil_};
	PyObject *arr_thetaH = PyArray_SimpleNewFromData(1,npy_shape_thetaH, NPY_DOUBLE, thetaH);
	
	// thetaL
	npy_intp npy_shape_thetaL[1] = {Nfil_};
	PyObject *arr_thetaL = PyArray_SimpleNewFromData(1,npy_shape_thetaL, NPY_DOUBLE, thetaL);
	
	PyArray_ENABLEFLAGS((PyArrayObject *)arr_angles, NPY_OWNDATA);
	PyArray_ENABLEFLAGS((PyArrayObject *)arr_hatZprime2, NPY_OWNDATA);
	PyArray_ENABLEFLAGS((PyArrayObject *)arr_psi_LH, NPY_OWNDATA);
	PyArray_ENABLEFLAGS((PyArrayObject *)arr_thetaH, NPY_OWNDATA);
	PyArray_ENABLEFLAGS((PyArrayObject *)arr_thetaL, NPY_OWNDATA);
	
	PyObject *rtrn = PyTuple_New(5);
	PyTuple_SetItem(rtrn, 0, arr_angles);
	PyTuple_SetItem(rtrn, 1, arr_hatZprime2);
	PyTuple_SetItem(rtrn, 2, arr_psi_LH);
	PyTuple_SetItem(rtrn, 3, arr_thetaH);
	PyTuple_SetItem(rtrn, 4, arr_thetaL);
	return rtrn ;
}

static PyObject *Paint_Filament(PyObject *self, PyObject *args){
	/* Getting the elements */
	PyObject *n = NULL;
	PyObject *nside = NULL;
	
	PyObject *Sizes_arr = NULL;
	PyObject *Centers_arr = NULL;
	PyObject *Angles_arr = NULL;
	PyObject *fpol0_arr = NULL;
	PyObject *thetaH_arr = NULL;
	PyObject *betadust_arr = NULL;
	PyObject *Tdust_arr = NULL;

	PyObject *Bcube=NULL;
	PyObject *size=NULL;
	PyObject *Npix_magfield=NULL;
	PyObject *resolution=NULL;
	PyObject *freqs_arr=NULL;
	PyObject *Nfreqs=NULL;
	PyObject *tqu_dict=NULL; // this is the dict with the tqu_maps, the keys are the nsides from nside_fixed to 128
	PyObject *skip_Bcube=NULL;
	PyObject *rank=NULL;
	
	int ii, flag;
	int isInside=1, sucess;
	
	if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOOO",&n, &nside, &Sizes_arr, &Centers_arr, &Angles_arr, &fpol0_arr, &thetaH_arr, &betadust_arr, &Tdust_arr, &Bcube, &size, &Npix_magfield,&resolution,&freqs_arr,&Nfreqs,&tqu_dict,&skip_Bcube,&rank))
		return NULL;
	
	// Check arrays
	
	flag = PyArray_IS_C_CONTIGUOUS(Centers_arr);
	if (!flag){printf("Centers arr from dict. is not contiguous, exiting ...\n");exit(EXIT_FAILURE);}
	flag = PyArray_IS_C_CONTIGUOUS(Angles_arr);
	if (!flag){printf("Angles arr from dict. is not contiguous, exiting ...\n");exit(EXIT_FAILURE);}
	flag = PyArray_IS_C_CONTIGUOUS(Sizes_arr);
	if (!flag){printf("Sizes arr from dict. is not contiguous, exiting ...\n");exit(EXIT_FAILURE);}
	flag = PyArray_IS_C_CONTIGUOUS(Bcube);
	if (!flag){printf("Bcube arr from dict. is not contiguous, exiting ...\n");exit(EXIT_FAILURE);}
	
	int nside_fixed = (int) PyLong_AsLong(nside);
	int npix_fixed  = 12*nside_fixed*nside_fixed; // this cannot be over Nside 65k, but it is highly unlikely that we will create a map at that resolution
	int resol_int   = (int) PyLong_AsLong(resolution) ; 
	int Npix_magfield_int = (int) PyLong_AsLong(Npix_magfield);
	double Size_double    =  PyFloat_AsDouble(size);
	int rank_ = (int) PyLong_AsLong(rank);

	/* We want the filament n*/
	int n_fil = (int) PyLong_AsLong(n);
	
	double *Centers_ptr = PyArray_DATA(Centers_arr); // Centers is an array with shape (3)
	double *Angles_ptr = PyArray_DATA(Angles_arr); // Angles is an array with shape (2)
	double *Sizes_ptr = PyArray_DATA(Sizes_arr); // Angles is an array with shape (3)
	double *Bcube_ptr = PyArray_DATA(Bcube);
	
	/* sample the Bcube at the center of the filament, I always do this */
	double vec_center[3];
	double Bcenter[3] ;
	for(int k=0;k<3;k++) vec_center[k] = Centers_ptr[n_fil*3+k] ;
	
	FilamentPaint_TrilinearInterpolation(Bcube_ptr, Size_double, Npix_magfield_int, vec_center, Bcenter);

	// Calculate which resolution I need to sample in 50x50 pixels at least. 2^n_nside is the nside necesary for the sampling 
	int n_nside = (int) round(log(0.1*resol_int*sqrt(PI/3.0)*sqrt(pow(Centers_ptr[n_fil*3+0],2)+pow(Centers_ptr[n_fil*3+1],2)+pow(Centers_ptr[n_fil*3+2],2))/Sizes_ptr[n_fil*3+0])/log(2.0)) ;
	int nside_variable = pow(2,n_nside);
	long npix_variable = 12*nside_variable*nside_variable ;
	int skip_Bcube_ = (int) PyLong_AsLong(skip_Bcube);

	// This is for testing if the cuboid is outside the box
	/* Calculate the rot matrix */
	double rot_matrix[3*3],inv_rot_matrix[3*3],xyz_vertices[8*3], xyz_normal_to_faces[6*3], xyz_faces[6*4*3], xyz_edges[6*2*3], xyz_edges_unit[6*2*3];
	FilamentPaint_RotationMatrix(Angles_ptr,n_fil,rot_matrix);
	FilamentPaint_InvertRotMat(rot_matrix,inv_rot_matrix);
	/* Calculate the 8 vertices in the xyz coordinates */
	FilamentPaint_xyzVertices(rot_matrix,Sizes_ptr,Centers_ptr,Size_double,&isInside,xyz_vertices,n_fil);
	// We skip the big filaments that would be sampled at nside 64 or lower
	if (isInside==1 && nside_variable >= 128){
		/* Calculate normal to faces rotated */
		FilamentPaint_xyzNormalToFaces(rot_matrix,xyz_normal_to_faces);
		/* Calculate the faces matrix*/
		FilamentPaint_xyzFaces(xyz_vertices,xyz_faces);
		// Calculate the edges vectors
		FilamentPaint_xyzEdgeVectors(xyz_faces,xyz_edges);
		FilamentPaint_xyzEdgeVectorsUnit(xyz_edges,xyz_edges_unit);
		// Calculate the polygon in the variable nside map
		long* ipix = calloc(200000,sizeof(long));
		if (!ipix){
			printf("mem failure, could not allocate ipix, exiting \n");
			exit(EXIT_FAILURE);
		}
		int nipix;
		//double querypolygon_start_time = omp_get_wtime();
		FilamentPaint_DoQueryPolygon(nside_variable,xyz_vertices,ipix,&nipix,Centers_ptr,&sucess,n_fil);
		double* localtriad = calloc(nipix*3*3,sizeof(double));
		if (!localtriad){
			printf("mem failure, could not allocate localtriad, exiting \n");
			exit(EXIT_FAILURE);
		}
		//double querypolygon_run_time = omp_get_wtime() - querypolygon_start_time;
		if(sucess == 0){
			free(ipix);
			free(localtriad);
			PyErr_SetString(PyExc_TypeError, "Oh no!");
			return (PyObject *) NULL;
		}
		// Cycle through each of the pixels in ipix
		//printf("Filament %i has %u pixels in a nside=%i pixelization \n",n_fil,nipix,nside_variable) ;
		// Im going to parallelize the filling of the local triad, I'll time it
		//double localtriad_start_time = omp_get_wtime();
		FilamentPaint_DoLocalTriad(ipix,nipix,nside_variable,localtriad);
		//double localtriad_run_time = omp_get_wtime() - localtriad_start_time;
		// Now I won't use a full size map at nside_variable, because it would be a waste of memory. Instead, I will map the integ result into the nested nside_fixed map
		
		// Calculate the sed factor
		double fpol0 = PyFloat_AsDouble(fpol0_arr);
		double thetaH = PyFloat_AsDouble(thetaH_arr);
		double betadust = PyFloat_AsDouble(betadust_arr);
		double Tdust = PyFloat_AsDouble(Tdust_arr);
		double *freqs_arr_ = PyArray_DATA(freqs_arr);
		int Nfreqs_ = (int) PyLong_AsLong(Nfreqs);
		double x_d_353 = H_PLANCK*353.0*1.e9/(K_BOLTZ*Tdust);
		// The conversion is 
		double thermo_2_rj_353 = 0.0774279729042878 ; // this is fixed for 353 GHz
		// the sed factor for 353 GHz
		double sed_factor_353 = pow(353*1e9,betadust+1.0)/(exp(x_d_353)-1.0) / thermo_2_rj_353 ;
		double *sed_factor_nu = calloc(Nfreqs_,sizeof(double));
		for(int n=0;n<Nfreqs_;n++){
			double x_d_nu = H_PLANCK*freqs_arr_[n]*1.e9/(K_BOLTZ*Tdust);
			double x_cmb = H_PLANCK*freqs_arr_[n]*1.0e9/(K_BOLTZ*T_CMB);
			double thermo_2_rj = pow(x_cmb,2) * exp(x_cmb) / pow(exp(x_cmb) - 1.0,2) ; // multiplying by this factor transform thermo 2 RJ units, divide for the reverse conversion
			sed_factor_nu[n] = pow(freqs_arr_[n]*1e9,betadust+1.0)/(exp(x_d_nu)-1.0) / thermo_2_rj  / sed_factor_353 ;
		}
		
		// Now I will run FilamentPaint_CalculateDistances and FilamentPaint_RiemannIntegrator over the tqumap itself
		if (nside_fixed > nside_variable){
			// This means the filament is bigger than the fixed resolution and we have to upgrade the map
			// we will have to cycle over all children pixels in the nside_fixed pixelization, assigning the same value to each
			PyObject *nside_pointer = PyLong_FromLong((long) nside_variable);
			PyObject *tqu_array = PyDict_GetItem(tqu_dict, nside_pointer);
			if (!tqu_array){
				printf("Nside not in the tqu dictionary, exiting ...\n");
				exit(EXIT_FAILURE);
			}
			flag = PyArray_IS_C_CONTIGUOUS(tqu_array);
			if (!flag){printf("Nside arr from dict. is not contiguous, exiting ...\n");exit(EXIT_FAILURE);}
			Py_INCREF(tqu_array);
			Py_DECREF(nside_pointer);
			double *tqu_array_c = PyArray_DATA(tqu_array);
			Py_DECREF(tqu_array);
			//double pixelcycle_start_time = omp_get_wtime();
			#pragma omp parallel 
			{
			#pragma omp for schedule(static)
			for(ii=0;ii<nipix;ii++){
				int jj,skip_pix=1,nn; // skip_pixel will be True at the beggining, and FilamentPaint_CalculateDistances will change it to False if we find the intersection
				int index_pix = (int) ipix[ii];
				double rDistances[2];
				double integ[3];
				FilamentPaint_CalculateDistances(xyz_normal_to_faces,xyz_faces,xyz_edges,xyz_edges_unit,ii,localtriad,rDistances,&skip_pix);
				if (skip_pix == 1){
					// if skip_pix is still True, then we could not find the intersection with the face and we skip this particular pixel
					printf("There is a pixel for which we couldn't find the intersection with the filament box, we skip it \n") ;
					continue;
				}
				FilamentPaint_RiemannIntegrator(rDistances[0],rDistances[1],inv_rot_matrix,ii,localtriad,Centers_ptr,Sizes_ptr,Bcube_ptr,Size_double,Npix_magfield_int,integ,fpol0,thetaH,n_fil,skip_Bcube_,Bcenter);
				// set the integ values on to the numpy array whose pointer is tqu_array, which is a 2 dimensional array with shape (3,12*nside**2)
				if (index_pix<0 || index_pix>npix_variable-1){printf("The pixel %i arr is invalid in pixelization Nside=%i, exiting (nside_fixed>nside_var)...\n",index_pix,nside_variable);exit(EXIT_FAILURE);}
				// tqu_array_c will have shape (Nfreqs,3,Npixels)
				for(jj=0;jj<3;jj++){
					for(nn=0;nn<Nfreqs_;nn++) tqu_array_c[ nn*3*((int) npix_variable) + jj*((int) npix_variable) + index_pix] += sed_factor_nu[nn] * integ[jj];
				}
			}
			}
		}
		else{
			// else, we are degrading or copying the same
			if (nside_fixed == nside_variable){
				// this means we modify the numpy array with the key nside_fixed
				PyObject *nside_pointer = PyLong_FromLong((long) nside_fixed);
				PyObject *tqu_array = PyDict_GetItem(tqu_dict, nside_pointer);
				if (!tqu_array){
					printf("Nside not in the tqu dictionary, exiting ...\n");
					exit(EXIT_FAILURE);
				}
				flag = PyArray_IS_C_CONTIGUOUS(tqu_array);
				if (!flag){printf("Nside arr from dict. is not contiguous, exiting ...\n");exit(EXIT_FAILURE);}
				Py_INCREF(tqu_array);
				Py_DECREF(nside_pointer);
				double *tqu_array_c = PyArray_DATA(tqu_array);
				Py_DECREF(tqu_array);
				//double pixelcycle_start_time = omp_get_wtime();
				#pragma omp parallel
				{
				#pragma omp for schedule(static)
				for(ii=0;ii<nipix;ii++){
					int jj, skip_pix=1,nn;
					int index_pix = (int) ipix[ii];
					double rDistances[2];
					double integ[3];
					FilamentPaint_CalculateDistances(xyz_normal_to_faces,xyz_faces,xyz_edges,xyz_edges_unit,ii,localtriad,rDistances,&skip_pix);
					if (skip_pix == 1){
						// if skip_pix is still True, then we could not find the intersection with the face and we skip this particular pixel
						printf("There is a pixel for which we couldn't find the intersection with the filament box, we skip it \n") ;
						continue;
					}
					FilamentPaint_RiemannIntegrator(rDistances[0],rDistances[1],inv_rot_matrix,ii,localtriad,Centers_ptr,Sizes_ptr,Bcube_ptr,Size_double,Npix_magfield_int,integ,fpol0,thetaH,n_fil,skip_Bcube_,Bcenter);
					// set the integ values on to the numpy array whose pointer is tqu_array, which is a 2 dimensional array with shape (3,12*nside**2)
					// check that index_pix has a valid range
					if (index_pix<0 || index_pix>npix_fixed-1){printf("The pixel %i arr is invalid in pixelization Nside=%i, exiting (nside_fixed=nside_var) ...\n",index_pix,nside_fixed);exit(EXIT_FAILURE);}
					for(jj=0;jj<3;jj++){
						for(nn=0;nn<Nfreqs_;nn++) tqu_array_c[ nn*3*npix_fixed + jj*npix_fixed + index_pix] += sed_factor_nu[nn] * integ[jj];
					}
				}
				}
			}
			else if (nside_fixed < nside_variable){
				// This means the filament is smaller than the fixed resolution and we have to degrade the map
				// step will determine how many steps in nside I went, e.g. from 2048 to 8192 I went 2 steps
				int step = (int)(log(nside_variable)/log(2.0) - log(nside_fixed)/log(2.0)) ;
				// This is for filaments sampled at a resolution > nside_fixed, so we will acumulate them in the nside_fixed numpy array
				PyObject *nside_pointer = PyLong_FromLong((long) nside_fixed);
				PyObject *tqu_array = PyDict_GetItem(tqu_dict, nside_pointer);
				if (!tqu_array){
					printf("Nside not in the tqu dictionary, exiting ...\n");
					exit(EXIT_FAILURE);
				}
				flag = PyArray_IS_C_CONTIGUOUS(tqu_array);
				if (!flag){printf("Nside arr from dict. is not contiguous, exiting ...\n");exit(EXIT_FAILURE);}
				Py_INCREF(tqu_array);
				Py_DECREF(nside_pointer);
				double *tqu_array_c = PyArray_DATA(tqu_array);
				Py_DECREF(tqu_array);
				//double pixelcycle_start_time = omp_get_wtime();
				#pragma omp parallel
				{
				#pragma omp for schedule(static)
				for(ii=0;ii<nipix;ii++){
					int jj, skip_pix = 1,nn;
					long index_parent_pix_long = ipix[ii] >> 2*step ; // this has to be a long, since we could have Nside >= 65k
					int index_parent_pixel = (int) index_parent_pix_long ; // this can be a int, since nside_fixed will not be >=65k
					double rDistances[2];
					double integ[3];
					FilamentPaint_CalculateDistances(xyz_normal_to_faces,xyz_faces,xyz_edges,xyz_edges_unit,ii,localtriad,rDistances,&skip_pix);
					if (skip_pix == 1){
						// if skip_pix is still True, then we could not find the intersection with the face and we skip this particular pixel
						printf("There is a pixel for which we couldn't find the intersection with the filament box, we skip it \n") ;
						continue;
					}
					FilamentPaint_RiemannIntegrator(rDistances[0],rDistances[1],inv_rot_matrix,ii,localtriad,Centers_ptr,Sizes_ptr,Bcube_ptr,Size_double,Npix_magfield_int,integ,fpol0,thetaH,n_fil,skip_Bcube_,Bcenter);
					// set the integ values on to the numpy array whose pointer is tqu_array, which is a 2 dimensional array with shape (3,12*nside**2)
					// we have to protect this with a omp critical, since several small pixels can end up in the same big pixel = race condition
					#pragma omp critical
					{
						// check that index_parent_pixel has a valid range
						if (index_parent_pixel<0 || index_parent_pixel>npix_fixed-1){printf("The pixel %i arr is invalid in pixelization Nside=%i, exiting (nside_fixed<nside_var) ...\n",index_parent_pixel,nside_fixed);exit(EXIT_FAILURE);}
						for(jj=0;jj<3;jj++){
							for(nn=0;nn<Nfreqs_;nn++) tqu_array_c[ nn*3*npix_fixed + jj*npix_fixed + index_parent_pixel] += sed_factor_nu[nn] * integ[jj] / pow(4,step) ;
						}
					}
				}
				}
			}
		}
		free(ipix);
		free(localtriad);
		free(sed_factor_nu);
		// We have nothing to return 
		Py_RETURN_NONE;
	}// this is the end of (isInside==1),
	else{
		// this means that the box overflows the volume, so we just exit, we don't do anything else
		Py_RETURN_NONE;
	}
}









static PyObject *Paint_Filament_OnlyIntensity(PyObject *self, PyObject *args){
	/* Getting the elements */
	PyObject *n = NULL;
	PyObject *nside = NULL;
	PyObject *Sizes_arr = NULL;
	PyObject *Centers_arr = NULL;
	PyObject *Angles_arr = NULL;
	PyObject *betadust_arr = NULL;
	PyObject *Tdust_arr = NULL;
	PyObject *size=NULL;
	PyObject *resolution=NULL;
	PyObject *freqs_arr=NULL;
	PyObject *Nfreqs=NULL;
	PyObject *t_dict=NULL; // this is the dict with the tqu_maps, the keys are the nsides from nside_fixed to 128
	PyObject *rank=NULL;
	
	int ii, flag;
	int isInside=1, sucess;
	
	if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOO",&n, &nside, &Sizes_arr, &Centers_arr, &Angles_arr, &betadust_arr, &Tdust_arr, &size,&resolution,&freqs_arr,&Nfreqs,&t_dict,&rank))
		return NULL;
	
	// Check arrays
	
	flag = PyArray_IS_C_CONTIGUOUS(Centers_arr);
	if (!flag){printf("Centers arr from dict. is not contiguous, exiting ...\n");exit(EXIT_FAILURE);}
	flag = PyArray_IS_C_CONTIGUOUS(Angles_arr);
	if (!flag){printf("Angles arr from dict. is not contiguous, exiting ...\n");exit(EXIT_FAILURE);}
	flag = PyArray_IS_C_CONTIGUOUS(Sizes_arr);
	if (!flag){printf("Sizes arr from dict. is not contiguous, exiting ...\n");exit(EXIT_FAILURE);}

	int nside_fixed = (int) PyLong_AsLong(nside);
	int npix_fixed  = 12*nside_fixed*nside_fixed; // this cannot be over Nside 65k, but it is highly unlikely that we will create a map at that resolution
	int resol_int   = (int) PyLong_AsLong(resolution) ; 
	double Size_double    =  PyFloat_AsDouble(size);
	int rank_ = (int) PyLong_AsLong(rank);

	/* We want the filament n*/
	int n_fil = (int) PyLong_AsLong(n);
	
	double *Centers_ptr = PyArray_DATA(Centers_arr); // Centers is an array with shape (3)
	double *Angles_ptr = PyArray_DATA(Angles_arr); // Angles is an array with shape (2)
	double *Sizes_ptr = PyArray_DATA(Sizes_arr); // Angles is an array with shape (3)

	// Calculate which resolution I need to sample in 50x50 pixels at least. 2^n_nside is the nside necesary for the sampling 
	int n_nside = (int) round(log(0.1*resol_int*sqrt(PI/3.0)*sqrt(pow(Centers_ptr[n_fil*3+0],2)+pow(Centers_ptr[n_fil*3+1],2)+pow(Centers_ptr[n_fil*3+2],2))/Sizes_ptr[n_fil*3+0])/log(2.0)) ;
	int nside_variable = pow(2,n_nside);
	long npix_variable = 12*nside_variable*nside_variable ;

	// This is for testing if the cuboid is outside the box
	/* Calculate the rot matrix */
	double rot_matrix[3*3],inv_rot_matrix[3*3],xyz_vertices[8*3], xyz_normal_to_faces[6*3], xyz_faces[6*4*3], xyz_edges[6*2*3], xyz_edges_unit[6*2*3];
	FilamentPaint_RotationMatrix(Angles_ptr,n_fil,rot_matrix);
	FilamentPaint_InvertRotMat(rot_matrix,inv_rot_matrix);
	/* Calculate the 8 vertices in the xyz coordinates */
	FilamentPaint_xyzVertices(rot_matrix,Sizes_ptr,Centers_ptr,Size_double,&isInside,xyz_vertices,n_fil);
	// We skip the big filaments that would be sampled at nside 64 or lower
	if (isInside==1 && nside_variable >= 128){
		/* Calculate normal to faces rotated */
		FilamentPaint_xyzNormalToFaces(rot_matrix,xyz_normal_to_faces);
		/* Calculate the faces matrix*/
		FilamentPaint_xyzFaces(xyz_vertices,xyz_faces);
		// Calculate the edges vectors
		FilamentPaint_xyzEdgeVectors(xyz_faces,xyz_edges);
		FilamentPaint_xyzEdgeVectorsUnit(xyz_edges,xyz_edges_unit);
		// Calculate the polygon in the variable nside map
		long* ipix = calloc(200000,sizeof(long));
		if (!ipix){
			printf("mem failure, could not allocate ipix, exiting \n");
			exit(EXIT_FAILURE);
		}
		int nipix;
		//double querypolygon_start_time = omp_get_wtime();
		FilamentPaint_DoQueryPolygon(nside_variable,xyz_vertices,ipix,&nipix,Centers_ptr,&sucess,n_fil);
		double* localtriad = calloc(nipix*3*3,sizeof(double));
		if (!localtriad){
			printf("mem failure, could not allocate localtriad, exiting \n");
			exit(EXIT_FAILURE);
		}
		//double querypolygon_run_time = omp_get_wtime() - querypolygon_start_time;
		if(sucess == 0){
			free(ipix);
			free(localtriad);
			PyErr_SetString(PyExc_TypeError, "Oh no!");
			return (PyObject *) NULL;
		}
		// Cycle through each of the pixels in ipix
		//printf("Filament %i has %u pixels in a nside=%i pixelization \n",n_fil,nipix,nside_variable) ;
		// Im going to parallelize the filling of the local triad, I'll time it
		//double localtriad_start_time = omp_get_wtime();
		FilamentPaint_DoLocalTriad(ipix,nipix,nside_variable,localtriad);
		//double localtriad_run_time = omp_get_wtime() - localtriad_start_time;
		// Now I won't use a full size map at nside_variable, because it would be a waste of memory. Instead, I will map the integ result into the nested nside_fixed map
		
		// Calculate the sed factor
		double betadust = PyFloat_AsDouble(betadust_arr);
		double Tdust = PyFloat_AsDouble(Tdust_arr);
		double *freqs_arr_ = PyArray_DATA(freqs_arr);
		int Nfreqs_ = (int) PyLong_AsLong(Nfreqs);
		double x_d_353 = H_PLANCK*353.0*1.e9/(K_BOLTZ*Tdust);
		// The conversion is 
		double thermo_2_rj_353 = 0.0774279729042878 ; // this is fixed for 353 GHz
		// the sed factor for 353 GHz
		double sed_factor_353 = pow(353*1e9,betadust+1.0)/(exp(x_d_353)-1.0) / thermo_2_rj_353 ;
		double *sed_factor_nu = calloc(Nfreqs_,sizeof(double));
		for(int n=0;n<Nfreqs_;n++){
			double x_d_nu = H_PLANCK*freqs_arr_[n]*1.e9/(K_BOLTZ*Tdust);
			double x_cmb = H_PLANCK*freqs_arr_[n]*1.0e9/(K_BOLTZ*T_CMB);
			double thermo_2_rj = pow(x_cmb,2) * exp(x_cmb) / pow(exp(x_cmb) - 1.0,2) ; // multiplying by this factor transform thermo 2 RJ units, divide for the reverse conversion
			sed_factor_nu[n] = pow(freqs_arr_[n]*1e9,betadust+1.0)/(exp(x_d_nu)-1.0) / thermo_2_rj  / sed_factor_353 ;
		}
		
		// Now I will run FilamentPaint_CalculateDistances and FilamentPaint_RiemannIntegrator over the tqumap itself
		if (nside_fixed > nside_variable){
			// This means the filament is bigger than the fixed resolution and we have to upgrade the map
			// we will have to cycle over all children pixels in the nside_fixed pixelization, assigning the same value to each
			PyObject *nside_pointer = PyLong_FromLong((long) nside_variable);
			PyObject *t_array = PyDict_GetItem(t_dict, nside_pointer);
			if (!t_array){
				printf("Nside not in the tqu dictionary, exiting ...\n");
				exit(EXIT_FAILURE);
			}
			flag = PyArray_IS_C_CONTIGUOUS(t_array);
			if (!flag){printf("Nside arr from dict. is not contiguous, exiting ...\n");exit(EXIT_FAILURE);}
			Py_INCREF(t_array);
			Py_DECREF(nside_pointer);
			double *t_array_c = PyArray_DATA(t_array);
			Py_DECREF(t_array);
			//double pixelcycle_start_time = omp_get_wtime();
			#pragma omp parallel 
			{
			#pragma omp for schedule(static)
			for(ii=0;ii<nipix;ii++){
				int jj,skip_pix=1,nn; // skip_pixel will be True at the beggining, and FilamentPaint_CalculateDistances will change it to False if we find the intersection
				int index_pix = (int) ipix[ii];
				double rDistances[2];
				FilamentPaint_CalculateDistances(xyz_normal_to_faces,xyz_faces,xyz_edges,xyz_edges_unit,ii,localtriad,rDistances,&skip_pix);
				if (skip_pix == 1){
					// if skip_pix is still True, then we could not find the intersection with the face and we skip this particular pixel
					printf("There is a pixel for which we couldn't find the intersection with the filament box, we skip it \n") ;
					continue;
				}
				double integ = FilamentPaint_RiemannIntegrator_OnlyIntensity(rDistances[0],rDistances[1],inv_rot_matrix,ii,localtriad,Centers_ptr,Sizes_ptr,n_fil);
				// set the integ values on to the numpy array whose pointer is tqu_array, which is a 2 dimensional array with shape (3,12*nside**2)
				if (index_pix<0 || index_pix>npix_variable-1){printf("The pixel %i arr is invalid in pixelization Nside=%i, exiting (nside_fixed>nside_var)...\n",index_pix,nside_variable);exit(EXIT_FAILURE);}
				// t_array_c will have shape (Nfreqs,Npixels)
				for(nn=0;nn<Nfreqs_;nn++) t_array_c[ nn*((int) npix_variable) + index_pix] += sed_factor_nu[nn] * integ;
			}
			}
		}
		else{
			// else, we are degrading or copying the same
			if (nside_fixed == nside_variable){
				// this means we modify the numpy array with the key nside_fixed
				PyObject *nside_pointer = PyLong_FromLong((long) nside_fixed);
				PyObject *t_array = PyDict_GetItem(t_dict, nside_pointer);
				if (!t_array){
					printf("Nside not in the tqu dictionary, exiting ...\n");
					exit(EXIT_FAILURE);
				}
				flag = PyArray_IS_C_CONTIGUOUS(t_array);
				if (!flag){printf("Nside arr from dict. is not contiguous, exiting ...\n");exit(EXIT_FAILURE);}
				Py_INCREF(t_array);
				Py_DECREF(nside_pointer);
				double *t_array_c = PyArray_DATA(t_array);
				Py_DECREF(t_array);
				//double pixelcycle_start_time = omp_get_wtime();
				#pragma omp parallel
				{
				//int n_thr_max = omp_get_num_threads();
				//printf("I got %i threads \n",n_thr_max);
				#pragma omp for schedule(static)
				for(ii=0;ii<nipix;ii++){
					int jj, skip_pix=1,nn;
					int index_pix = (int) ipix[ii];
					double rDistances[2];
					FilamentPaint_CalculateDistances(xyz_normal_to_faces,xyz_faces,xyz_edges,xyz_edges_unit,ii,localtriad,rDistances,&skip_pix);
					if (skip_pix == 1){
						// if skip_pix is still True, then we could not find the intersection with the face and we skip this particular pixel
						printf("There is a pixel for which we couldn't find the intersection with the filament box, we skip it \n") ;
						continue;
					}
					double integ = FilamentPaint_RiemannIntegrator_OnlyIntensity(rDistances[0],rDistances[1],inv_rot_matrix,ii,localtriad,Centers_ptr,Sizes_ptr,n_fil);
					// set the integ values on to the numpy array whose pointer is tqu_array, which is a 2 dimensional array with shape (3,12*nside**2)
					// check that index_pix has a valid range
					if (index_pix<0 || index_pix>npix_fixed-1){printf("The pixel %i arr is invalid in pixelization Nside=%i, exiting (nside_fixed=nside_var) ...\n",index_pix,nside_fixed);exit(EXIT_FAILURE);}
					// t_array_c will have shape (Nfreqs,Npixels)
					for(nn=0;nn<Nfreqs_;nn++) t_array_c[ nn*npix_fixed + index_pix] += sed_factor_nu[nn] * integ ;
				}
				}
			}
			else if (nside_fixed < nside_variable){
				// This means the filament is smaller than the fixed resolution and we have to degrade the map
				// step will determine how many steps in nside I went, e.g. from 2048 to 8192 I went 2 steps
				int step = (int)(log(nside_variable)/log(2.0) - log(nside_fixed)/log(2.0)) ;
				// This is for filaments sampled at a resolution > nside_fixed, so we will acumulate them in the nside_fixed numpy array
				PyObject *nside_pointer = PyLong_FromLong((long) nside_fixed);
				PyObject *t_array = PyDict_GetItem(t_dict, nside_pointer);
				if (!t_array){
					printf("Nside not in the tqu dictionary, exiting ...\n");
					exit(EXIT_FAILURE);
				}
				flag = PyArray_IS_C_CONTIGUOUS(t_array);
				if (!flag){printf("Nside arr from dict. is not contiguous, exiting ...\n");exit(EXIT_FAILURE);}
				Py_INCREF(t_array);
				Py_DECREF(nside_pointer);
				double *t_array_c = PyArray_DATA(t_array);
				Py_DECREF(t_array);
				//double pixelcycle_start_time = omp_get_wtime();
				#pragma omp parallel
				{
				//int n_thr_max = omp_get_num_threads();
				//printf("I got %i threads \n",n_thr_max);
				#pragma omp for schedule(static)
				for(ii=0;ii<nipix;ii++){
					int jj, skip_pix = 1,nn;
					long index_parent_pix_long = ipix[ii] >> 2*step ; // this has to be a long, since we could have Nside >= 65k
					int index_parent_pixel = (int) index_parent_pix_long ; // this can be a int, since nside_fixed will not be >=65k
					double rDistances[2];
					FilamentPaint_CalculateDistances(xyz_normal_to_faces,xyz_faces,xyz_edges,xyz_edges_unit,ii,localtriad,rDistances,&skip_pix);
					if (skip_pix == 1){
						// if skip_pix is still True, then we could not find the intersection with the face and we skip this particular pixel
						printf("There is a pixel for which we couldn't find the intersection with the filament box, we skip it \n") ;
						continue;
					}
					double integ = FilamentPaint_RiemannIntegrator_OnlyIntensity(rDistances[0],rDistances[1],inv_rot_matrix,ii,localtriad,Centers_ptr,Sizes_ptr,n_fil);
					// set the integ values on to the numpy array whose pointer is tqu_array, which is a 2 dimensional array with shape (3,12*nside**2)
					// we have to protect this with a omp critical, since several small pixels can end up in the same big pixel = race condition
					#pragma omp critical
					{
						// check that index_parent_pixel has a valid range
						if (index_parent_pixel<0 || index_parent_pixel>npix_fixed-1){printf("The pixel %i arr is invalid in pixelization Nside=%i, exiting (nside_fixed<nside_var) ...\n",index_parent_pixel,nside_fixed);exit(EXIT_FAILURE);}
						// t_array_c will have shape (Nfreqs,Npixels)
						for(nn=0;nn<Nfreqs_;nn++) t_array_c[ nn*npix_fixed + index_parent_pixel] += sed_factor_nu[nn] * integ / pow(4,step) ;
					}
				}
				}
			}
		}
		free(ipix);
		free(localtriad);
		free(sed_factor_nu);
		// We have nothing to return 
		Py_RETURN_NONE;
	}// this is the end of (isInside==1),
	else{
		// this means that the box overflows the volume, so we just exit, we don't do anything else
		Py_RETURN_NONE;
	}
}

static PyMethodDef FilamentPaintMethods[] = {
  {"Paint_Filament", Paint_Filament, METH_VARARGS,NULL},
  {"Paint_Filament_OnlyIntensity", Paint_Filament_OnlyIntensity, METH_VARARGS,NULL},
  {"Get_Angles", Get_Angles, METH_VARARGS,NULL},
  {"Reject_Big_Filaments", Reject_Big_Filaments, METH_VARARGS,NULL},
 {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef FilamentPaint_module = {
    PyModuleDef_HEAD_INIT,
    "FilamentPaint",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    FilamentPaintMethods
};

PyMODINIT_FUNC PyInit_FilamentPaint(void){
  PyObject *m;
  m = PyModule_Create(&FilamentPaint_module);
  import_array();  // This is important for using the numpy_array api, otherwise segfaults!
  return(m);
}