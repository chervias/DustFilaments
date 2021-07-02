#include <healpix_cxx/healpix_base.h>
#include <healpix_cxx/fitshandle.h>
#include <healpix_cxx/healpix_map.h>
#include <healpix_cxx/rangeset.h>
#include <healpix_cxx/pointing.h>
#include <healpix_cxx/vec3.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <omp.h>

using namespace std;

extern "C" {
	void query_polygon_wrapper(double* x_vert, double* y_vert, double* z_vert, int hsize, int nside, long* ipix_arr, int* nipix, int* sucess){
		// theta and phi are arrays with size hsize
		std::vector<pointing> vertices;
		//for(std::size_t j = 0; j<hsize ; j++){
		for(size_t j = 0; j< (unsigned int) hsize ; j++){
			vec3 v(x_vert[j],y_vert[j],z_vert[j]);
			pointing a(v);
			vertices.push_back(a);
			//printf("vertex %i = %.10E %.10E \n",j,a.phi,a.theta);
		}
		//printf("The nside for query polygon is %d \n",nside);
		T_Healpix_Base<long> hp_base(nside,NEST,SET_NSIDE); 
		rangeset<long> ipix;
		try{
			hp_base.query_polygon(vertices,ipix);
			std::vector<long> v = ipix.toVector();
			*nipix	= (int) v.size();
			// the size of ipix is 200k longs, if we get more pixels than that it will seg fault
			if (v.size()>200000){printf("The number of pixels from Query polygon is too high, exiting \n");exit(EXIT_FAILURE);}
			for(size_t i = 0; i < v.size(); i++){
			//for(std::size_t i = 0; i < v.size(); i++){
				ipix_arr[i]	= (long) v[i];
			}
			*sucess = 1;
		}
		catch (PlanckError e){
			*sucess = 0;
		}
	}
}

extern "C" {
	void localtriad_wrapper(long* ipix_final, int nipix, int nside, double* localtriad, int Nthreads){
		// localtriad will be nipix by 3 (hat(x),hat(y),hat(z)) by 3 (x,y,z components) matrix
		// localtriad[nipix * 3 * 3] = local_triad[i][j][k] = localtriad[i*3*3 + j*3 + k ]
		// set the healpix base object
		T_Healpix_Base<long> hp_base(nside,NEST,SET_NSIDE);
		//printf("I got %d threads available\n",nthreads);
		omp_set_num_threads(Nthreads);
		#pragma omp parallel for schedule(static)
		for(size_t i = 0; i < (unsigned int) nipix; i++){
		//for(std::size_t i = 0; i < nipix; i++){
			vec3 vec = hp_base.pix2vec(ipix_final[i]);
			double norm = sqrt(pow(vec.x,2)+pow(vec.y,2)+pow(vec.z,2)) ;
			double theta = acos(vec.z/norm);
			double phi = atan2(vec.y,vec.x);
			// fill the hat(x) vector from local triad (equivalent to hat(theta) from spherical coordinates)
			localtriad[i*3*3 + 0*3 + 0] = cos(theta)*cos(phi);
			localtriad[i*3*3 + 0*3 + 1] = cos(theta)*sin(phi);
			localtriad[i*3*3 + 0*3 + 2] = -sin(theta);
			// fill the hat(y) vector from local triad (equivalent to hat(phi) from spherical coordinates)
			localtriad[i*3*3 + 1*3 + 0] = -sin(phi);
			localtriad[i*3*3 + 1*3 + 1] = cos(phi);
			localtriad[i*3*3 + 1*3 + 2] = 0.0;
			// fill the hat(z) vector from local triad (equivalent to hat(r) from spherical coordinates or the output from pix2vec)
			localtriad[i*3*3 + 2*3 + 0] = vec.x/norm;
			localtriad[i*3*3 + 2*3 + 1] = vec.y/norm;
			localtriad[i*3*3 + 2*3 + 2] = vec.z/norm;
		}
	}
}
