// Convex hull code from here https://rosettacode.org/wiki/Convex_hull#C

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "query_polygon_wrapper.h"
#include <assert.h>
#include <stdbool.h>
#define PI 3.14159265358979323846

typedef struct tPoint{
	long double x, y;
	int index;
}Point;

bool ccw(const Point *a, const Point *b, const Point *c) {
    return (b->x - a->x) * (c->y - a->y) > (b->y - a->y) * (c->x - a->x);
}

int comparePoints(const void *lhs, const void *rhs) {
    const Point* lp = lhs;
    const Point* rp = rhs;
    if (lp->x < rp->x)
        return -1;
    if (rp->x < lp->x)
        return 1;
    if (lp->y < rp->y)
        return -1;
    if (rp->y < lp->y)
        return 1;
    return 0;
}
 
void fatal(const char* message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}
 
void* xmalloc(size_t n) {
    void* ptr = malloc(n);
    if (ptr == NULL)
        fatal("Out of memory");
    return ptr;
}
 
void* xrealloc(void* p, size_t n) {
    void* ptr = realloc(p, n);
    if (ptr == NULL)
        fatal("Out of memory");
    return ptr;
}
 
Point* convexHull(Point p[], int len, int* hsize) {
    if (len == 0) {
        *hsize = 0;
        return NULL;
    }
 
    int i, size = 0, capacity = 4;
    Point* hull = xmalloc(capacity * sizeof(Point));
 
    qsort(p, len, sizeof(Point), comparePoints);
 
    /* lower hull */
    for (i = 0; i < len; ++i) {
        while (size >= 2 && !ccw(&hull[size - 2], &hull[size - 1], &p[i]))
            --size;
        if (size == capacity) {
            capacity *= 2;
            hull = xrealloc(hull, capacity * sizeof(Point));
        }
        assert(size >= 0 && size < capacity);
        hull[size++] = p[i];
    }
 
    /* upper hull */
    int t = size + 1;
    for (i = len - 1; i >= 0; i--) {
        while (size >= t && !ccw(&hull[size - 2], &hull[size - 1], &p[i]))
            --size;
        if (size == capacity) {
            capacity *= 2;
            hull = xrealloc(hull, capacity * sizeof(Point));
        }
        assert(size >= 0 && size < capacity);
        hull[size++] = p[i];
    }
    --size;
    assert(size >= 0);
    hull = xrealloc(hull, size * sizeof(Point));
    *hsize = size;
    return hull;
}

void FilamentPaint_RotationMatrix(double* angles_arr, int n_fil, double* rot_matrix){
	double ca = cos(angles_arr[n_fil*2+0]); 
	double cb = cos(angles_arr[n_fil*2+1]);
	double sa = sin(angles_arr[n_fil*2+0]); 
	double sb = sin(angles_arr[n_fil*2+1]);
	rot_matrix[0*3+0] = ca*cb;
	rot_matrix[1*3+0] = sa*cb;
	rot_matrix[2*3+0] = -sb;
	rot_matrix[0*3+1] = -sa;
	rot_matrix[1*3+1] = ca;
	rot_matrix[2*3+1] = 0;
	rot_matrix[0*3+2] = ca*sb;
	rot_matrix[1*3+2] = sa*sb;
	rot_matrix[2*3+2] = cb;
}
void FilamentPaint_InvertRotMat(double* rot_matrix, double* inv_rot_matrix){
	// Following wikipedia https://en.wikipedia.org/wiki/Invertible_matrix#Methods_of_matrix_inversion
	//c
	double determinant=0.0;
	int i,j;
	for (i=0;i<3;i++){
		determinant = determinant + (rot_matrix[0*3+i]*(rot_matrix[1*3+(i+1)%3]*rot_matrix[2*3+(i+2)%3] - rot_matrix[1*3+(i+2)%3]*rot_matrix[2*3+(i+1)%3]));
	}
	for(j=0;j<3;j++){
		for(i=0;i<3;i++){
			inv_rot_matrix[j*3+i] = ((rot_matrix[((i+1)%3)*3+(j+1)%3] * rot_matrix[((i+2)%3)*3+(j+2)%3]) - (rot_matrix[((i+1)%3)*3+(j+2)%3]*rot_matrix[((i+2)%3)*3+(j+1)%3]))/ determinant;
		}
	}
}

void FilamentPaint_xyzVertices(double* rot_matrix, double* sizes_arr, double* centers_arr, double Size, int* isInside,double* xyz_vertices, int n_fil){
	/* This calculates the vertices of the cuboid in the xyz fixed coord */
	// xyz_vertices has shape xyz_vertices[8][3] = xyz_vertices[i][j]
	int i,j,k;
	double XYZ_vertices[8*3];
	XYZ_vertices[0*3+0] = XYZ_vertices[1*3+0] = XYZ_vertices[2*3+0]=XYZ_vertices[3*3+0] = +5*sizes_arr[n_fil*3+0];
	/* X element of vertices 5,6,7,8 */
	XYZ_vertices[4*3+0] = XYZ_vertices[5*3+0] = XYZ_vertices[6*3+0]=XYZ_vertices[7*3+0] = -5*sizes_arr[n_fil*3+0];
	/* Y element of vertices 1,4,5,8 */
	XYZ_vertices[0*3+1] = XYZ_vertices[3*3+1] = XYZ_vertices[4*3+1]=XYZ_vertices[7*3+1] = -5*sizes_arr[n_fil*3+1];
	/* Y elemeny of vertices 2,3,6,7 */
	XYZ_vertices[1*3+1] = XYZ_vertices[2*3+1] = XYZ_vertices[5*3+1]=XYZ_vertices[6*3+1] = +5*sizes_arr[n_fil*3+1];
	/* Z element of vertices 1,2,5,6 */
	XYZ_vertices[0*3+2] = XYZ_vertices[1*3+2] = XYZ_vertices[4*3+2]=XYZ_vertices[5*3+2] = +5*sizes_arr[n_fil*3+2];
	/* Z element of vertices 3,4,7,8 */
	XYZ_vertices[2*3+2] = XYZ_vertices[3*3+2] = XYZ_vertices[6*3+2]=XYZ_vertices[7*3+2] = -5*sizes_arr[n_fil*3+2];
	
	/* multiply rot matrix by XYZ vertex and move by center_arr */
	for (i=0;i<8;i++){
		// j is rows
		for (j=0;j<3;j++){
			double sum = 0.0;
			for (k=0;k<3;k++) sum += rot_matrix[j*3+k]*XYZ_vertices[i*3+k] ;
			xyz_vertices[i*3+j] = sum + centers_arr[n_fil*3+j];
		}
	}
	
	// Check if the vertices are outside the box
	// It will be 1. It will be changed to 0 if the condition is True
	for (i=0;i<8;i++){
		if (*isInside == 0) break;
		for (j=0;j<3;j++){
			if (xyz_vertices[i*3+j] < -0.5*Size || +0.5*Size < xyz_vertices[i*3+j]){
				*isInside = 0 ;
				break;
			}
		}
	}
}

void FilamentPaint_xyzNormalToFaces(double* rot_matrix, double* xyz_normal_to_faces){
	//xyz_normal_to_faces has shape xyz_normal_to_faces[6][3] = xyz_normal_to_faces[i][j]
	double XYZ_normals[6*3];
	int i,j,k;
	XYZ_normals[0*3+0] = 1.0 ; XYZ_normals[0*3+1] = 0.0 ; XYZ_normals[0*3+2] = 0.0 ;
	XYZ_normals[1*3+0] = 1.0 ; XYZ_normals[1*3+1] = 0.0 ; XYZ_normals[1*3+2] = 0.0 ;
	XYZ_normals[2*3+0] = 0.0 ; XYZ_normals[2*3+1] = 1.0 ; XYZ_normals[2*3+2] = 0.0 ;
	XYZ_normals[3*3+0] = 0.0 ; XYZ_normals[3*3+1] = 1.0 ; XYZ_normals[3*3+2] = 0.0 ;
	XYZ_normals[4*3+0] = 0.0 ; XYZ_normals[4*3+1] = 0.0 ; XYZ_normals[4*3+2] = 1.0 ;
	XYZ_normals[5*3+0] = 0.0 ; XYZ_normals[5*3+1] = 0.0 ; XYZ_normals[5*3+2] = 1.0 ;
	
	/* multiply rot matrix by XYZ normals */
	for (i=0;i<6;i++){
		// j is rows
		for (j=0;j<3;j++){
			double sum = 0.0;
			for (k=0;k<3;k++) sum += rot_matrix[j*3+k]*XYZ_normals[i*3+k] ;
			xyz_normal_to_faces[i*3+j] = sum;
		}
	}
}

void FilamentPaint_xyzFaces(double* xyz_vertices, double* xyz_faces){
	/* The faces are 6, and each one has 4 vertices, each vertex has x,y,z*/
	//The array has shape xyz_faces[6][4][3] = xyz_faces[i][j][k];
	// xyz_vertices is [8][3]
	int i;
	for (i=0;i<3;i++){
		/* face 0 has vertices 0,1,2,3 */
		xyz_faces[0*4*3+0*3+i]	= xyz_vertices[0*3+i];
		xyz_faces[0*4*3+1*3+i]	= xyz_vertices[1*3+i];
		xyz_faces[0*4*3+2*3+i]	= xyz_vertices[2*3+i];
		xyz_faces[0*4*3+3*3+i]	= xyz_vertices[3*3+i];
		/* face 1 has vertices 4,5,6,7 */
		xyz_faces[1*4*3+0*3+i]	= xyz_vertices[4*3+i];
		xyz_faces[1*4*3+1*3+i]	= xyz_vertices[5*3+i];
		xyz_faces[1*4*3+2*3+i]	= xyz_vertices[6*3+i];
		xyz_faces[1*4*3+3*3+i]	= xyz_vertices[7*3+i];
		/* face 2 has vertices 0,3,7,4 */
		xyz_faces[2*4*3+0*3+i]	= xyz_vertices[0*3+i];
		xyz_faces[2*4*3+1*3+i]	= xyz_vertices[3*3+i];
		xyz_faces[2*4*3+2*3+i]	= xyz_vertices[7*3+i];
		xyz_faces[2*4*3+3*3+i]	= xyz_vertices[4*3+i];
		/* face 3 has vertices 1,2,6,5 */
		xyz_faces[3*4*3+0*3+i]	= xyz_vertices[1*3+i];
		xyz_faces[3*4*3+1*3+i]	= xyz_vertices[2*3+i];
		xyz_faces[3*4*3+2*3+i]	= xyz_vertices[6*3+i];
		xyz_faces[3*4*3+3*3+i]	= xyz_vertices[5*3+i];
		/* face 4 has vertices 0,1,5,4 */
		xyz_faces[4*4*3+0*3+i]	= xyz_vertices[0*3+i];
		xyz_faces[4*4*3+1*3+i]	= xyz_vertices[1*3+i];
		xyz_faces[4*4*3+2*3+i]	= xyz_vertices[5*3+i];
		xyz_faces[4*4*3+3*3+i]	= xyz_vertices[4*3+i];
		/* face 5 has vertices 2,3,7,6 */
		xyz_faces[5*4*3+0*3+i]	= xyz_vertices[2*3+i];
		xyz_faces[5*4*3+1*3+i]	= xyz_vertices[3*3+i];
		xyz_faces[5*4*3+2*3+i]	= xyz_vertices[7*3+i];
		xyz_faces[5*4*3+3*3+i]	= xyz_vertices[6*3+i];
	}
}

void FilamentPaint_xyzEdgeVectors(double* xyz_faces, double* xyz_edges){
	//xyz_faces has shape [6][4][3]
	//xyz_edges has shape [6][2][3]
	int i,k;
	for (i=0;i<6;i++){
		for (k=0;k<3;k++){
			xyz_edges[i*2*3+0*3+k]	= xyz_faces[i*4*3+1*3+k] - xyz_faces[i*4*3+0*3+k];
			xyz_edges[i*2*3+1*3+k]	= xyz_faces[i*4*3+3*3+k] - xyz_faces[i*4*3+0*3+k];
		}
	}
}

void FilamentPaint_xyzEdgeVectorsUnit(double* xyz_edges, double* xyz_edges_unit){
	// xyz_edges has shape [6][2][3]
	//xyz_edges_unit has shape [6][2][3];
	int i,j ;
	for (i=0;i<6;i++){
		double norm0=0.0,norm1=0.0;
		for (j=0;j<3;j++){
			norm0 = norm0 + pow(xyz_edges[i*2*3+0*3+j],2);
			norm1 = norm1 + pow(xyz_edges[i*2*3+1*3+j],2);
		}
		for (j=0;j<3;j++){
			xyz_edges_unit[i*2*3+0*3+j] = xyz_edges[i*2*3+0*3+j]/sqrt(norm0);
			xyz_edges_unit[i*2*3+1*3+j] = xyz_edges[i*2*3+1*3+j]/sqrt(norm1);
		}
	}
}
void FilamentPaint_DoQueryPolygon(int nside, double* xyz_vertices, long* ipix, int* nipix, double* centers_arr, int* sucess, int n_fil){
	int i,k;
	int hsize;
	Point points_ch[8];
	// I will use this code https://rosettacode.org/wiki/Convex_hull#C
	// I need to transform the 8 corners from 3D points to longitude/colatitude
	// Remember that the colat must be converted to lat
	long double theta_center, phi_center, phi_p, theta_p;
	double radius_center = 0.0 ;
	phi_center = atan2(centers_arr[n_fil*3+1],centers_arr[n_fil*3+0]) ;
	for (i=0;i<3;i++) radius_center += pow(centers_arr[n_fil*3+i],2);
	theta_center = acos(centers_arr[n_fil*3+2]/sqrt(radius_center)) ;
	for (i=0;i<8;i++){
		double radius = 0.0;
		for (k=0;k<3;k++) radius += pow(xyz_vertices[i*3 + k],2);
		phi_p = atan2(xyz_vertices[i*3 + 1],xyz_vertices[i*3 + 0]) ; // This is phi from -pi to pi converted to 0 to 2pi
		theta_p = acos(xyz_vertices[i*3 + 2]/sqrt(radius)) ; // This is theta (colatitude) 0 to pi
		long double kk = 2.0L/( 1.0L + sin(0.5L*PI - theta_center)*sin(0.5L*PI-theta_p) + cos(0.5L*PI-theta_center)*cos(0.5L*PI-theta_p)*cos(phi_p-phi_center) );
		points_ch[i].x = kk*cos(0.5L*PI-theta_p)*sin(phi_p - phi_center) ;
		points_ch[i].y = kk*( cos(0.5L*PI-theta_center)*sin(0.5L*PI-theta_p) - sin(0.5L*PI-theta_center)*cos(0.5L*PI-theta_p)*cos(phi_p - phi_center)  );
		points_ch[i].index = (int) i;
	}
	Point* hull = convexHull(points_ch, sizeof(points_ch)/sizeof(Point), &hsize);
	//printf("I have %i points in the convex hull \n",hsize) ;
	// You need to know how many points are in the convex hull before defining x2,y2,z2
	double* x2 = calloc(hsize,sizeof(double));
	double* y2 = calloc(hsize,sizeof(double));
	double* z2 = calloc(hsize,sizeof(double));
	for (i=0;i<hsize;i++){ 
		int in = hull[i].index ;
		//printf("vertex %i is in the convex hull\n",in);
		x2[i] = xyz_vertices[in*3 + 0] ; y2[i] = xyz_vertices[in*3 + 1] ; z2[i] = xyz_vertices[in*3 + 2] ;
//		phi2[i] = phi1[in] ;
//		theta2[i] = theta1[in];
//		printf("Vertex %i has coordinates %.20E %.20E %.20E \n",in,xyz_vertices[in*3 + 0],xyz_vertices[in*3 + 1],xyz_vertices[in*3 + 2]);
	}
	query_polygon_wrapper(x2,y2,z2,hsize,nside,ipix,nipix,sucess);
	/*
	if (*sucess==0){
		// Print the coordinates if it fails
		// Print azimuth, colatitude
		// 0 for not printing, 1 for printing
		if (0){
			printf("phi1=[");
			for (i=0;i<8;i++) printf(",%.20E",phi1[i]);
			printf("]\n");
			printf("theta1=[");
			for (i=0;i<8;i++) printf(",%.20E",theta1[i]);
			printf("]\n");
			// ----------------------------------
			printf("phi2=[");
//			for (i=0;i<hsize;i++) printf(",%.20E",phi2[i]);
			printf("]\n");
			printf("theta2=[");
//			for (i=0;i<hsize;i++) printf(",%.20E",theta2[i]);
			printf("]\n");
			// Print stereo x,y
			printf("x1=[");
			for (i=0;i<8;i++) printf(",%.20E",(double) points_ch[i].x);
			printf("]\n");
			printf("y1=[");
			for (i=0;i<8;i++) printf(",%.20E",(double) points_ch[i].y);
			printf("]\n");
			// ----------------------------------
			printf("x2=[");
			for (i=0;i<hsize;i++) printf(",%.20E",(double) hull[i].x);
			printf("]\n");
			printf("y2=[");
			for (i=0;i<hsize;i++) printf(",%.20E",(double) hull[i].y);
			printf("]\n");
		}
	}
	*/
	free(hull);
	free(x2);
	free(y2);
	free(z2);
}

void FilamentPaint_DoLocalTriad(long* ipix_final, int nipix, int nside, double* localtriad){
	// localtriad is double* localtriad = calloc(nipix*3*3,sizeof(double)), which is created before
	localtriad_wrapper(ipix_final,nipix,nside,localtriad);
}

void FilamentPaint_CalculateDistances(double* xyz_normal_to_faces, double* xyz_faces, double* xyz_edges, double* xyz_edges_unit, int idx_pix, double* local_triad, double* distances, int* skip_pixel){
	// This function receives the Filament matrices and the local_triad and the pixel index and returns the 2 distances of intersection
	// shapes: xyz_normal_to_faces[6][3] , xyz_faces[6][4][3] , xyz_edges[6][2][3] , xyz_edges_unit[6][2][3]
	// skip_pixel will be true when the function is called. It will be changed to False when a intersect face is found
	double radii_intersect[6];
	int i,j,c=0,id_faces[2];
	for (i=0 ; i<6 ; i++){
		// iterate over the 6 faces
		// "bottom" is the dot product between the normal of face i and r_unit_vector, "top" is the dot product between the normal and some point in the plane
		double bottom=0.0,top=0.0;
		// here j indexes the x,y,z componentes
		for (j=0 ; j<3 ; j++) bottom = bottom + xyz_normal_to_faces[i*3+j]*local_triad[idx_pix*3*3 + 2*3 + j];
		if (bottom==0.0){
			// if bottom is 0.0, the radius will never intersect the plane
			radii_intersect[i] = -1.0;
		}
		else{
			for (j=0;j<3;j++) top = top + xyz_normal_to_faces[i*3+j]*xyz_faces[i*4*3 + 2*3 + j] ;
			radii_intersect[i] = top / bottom ;
			double VectCornerToIntersection[3];
			for (j=0;j<3;j++) VectCornerToIntersection[j] = radii_intersect[i]*local_triad[idx_pix*3*3 + 2*3 + j] - xyz_faces[i*4*3 + 0*3 + j] ;
			double proj10 = 0.0, proj30=0.0 , norm0=0.0 , norm1=0.0;
			for (j=0;j<3;j++){
				proj10 = proj10 + VectCornerToIntersection[j] * xyz_edges_unit[i*2*3 + 0*3 + j] ;
				proj30 = proj30 + VectCornerToIntersection[j] * xyz_edges_unit[i*2*3 + 1*3 + j] ;
				norm0  = norm0 + pow(xyz_edges[i*2*3 + 0*3 + j],2);
				norm1  = norm1 + pow(xyz_edges[i*2*3 + 1*3 + j],2);
			}
			if ((0.0<=proj10) && (proj10<=sqrt(norm0)) && (0.0<=proj30) && (proj30<=sqrt(norm1))){
				// face i does intersect the ray r*hat(r)
				id_faces[c] = i ;
				c++;

			}
		}
	}
	if (c == 2){
		// because we have found an intersection, we change skip_pixel to False
		*skip_pixel = 0;
		distances[0] = radii_intersect[id_faces[0]];
		distances[1] = radii_intersect[id_faces[1]];
	}
}

double FilamentPaint_Density(double r, double* rot_matrix, int idx_pix, double* local_triad, double* centers_arr, double* sizes_arr, int n_fil){
	// This function defines the density profile within the cuboid in the XYZ coordinates
	// rot_matrix must be the inverse rotation matrix
	int i,j;
	double XYZ_coord[3];
	double radius, profile;
	for (i=0;i<3;i++){
		double sum=0.0 ;
		for (j=0;j<3;j++){
			sum = sum + rot_matrix[i*3 + j]*(r*local_triad[idx_pix*3*3 + 2*3 + j] - centers_arr[n_fil*3+j]) ;
		}
		XYZ_coord[i] = sum ;
	}
	radius		= pow(XYZ_coord[0]/sizes_arr[n_fil*3+0],2)+pow(XYZ_coord[1]/sizes_arr[n_fil*3+1],2)+pow(XYZ_coord[2]/sizes_arr[n_fil*3+2],2);
	//if (radius <= 1.0){
	//	profile 	= exp(-sqrt(radius)) ;
	profile 	= exp(-0.5*radius) ;
	//}
	//else{
	//	profile 	= 0.0;
	//}
	return profile;
}

void FilamentPaint_TrilinearInterpolation(double* Bcube_ptr, double size_box, int nbox, double* vector, double* c){
	// Bcube is the cube with the values, this cube has dimensions nbox*nbox*nbox pixels, size_box is the physical size of each side
	// vector is the vector for which we want to know the interpolated value
	// This follows https://en.wikipedia.org/wiki/Trilinear_interpolation
	int i;
	// First, we need to get the indices of the cube where vector lives. These indice should be from 0 to nbox-1
	int idx_x1 	= ceil(vector[0]*(nbox-1)/size_box + 0.5*(nbox-1));
	int idx_x0 	= floor(vector[0]*(nbox-1)/size_box + 0.5*(nbox-1));
	int idx_y1 	= ceil(vector[1]*(nbox-1)/size_box + 0.5*(nbox-1));
	int idx_y0 	= floor(vector[1]*(nbox-1)/size_box + 0.5*(nbox-1));
	int idx_z1 	= ceil(vector[2]*(nbox-1)/size_box + 0.5*(nbox-1));
	int idx_z0 	= floor(vector[2]*(nbox-1)/size_box + 0.5*(nbox-1));
	// check the indices
	if (idx_x1<0 || idx_x1>nbox-1 || idx_x0<0 || idx_x0>nbox-1){
		printf("Error, X indices in the Bcube interpolation is outside the allowed value of (0,%i)\n",nbox-1) ;
		exit(EXIT_FAILURE);
	}
	if (idx_y1<0 || idx_y1>nbox-1 || idx_y0<0 || idx_y0>nbox-1){
		printf("Error, Y indices in the Bcube interpolation is outside the allowed value of (0,%i)\n",nbox-1) ;
		exit(EXIT_FAILURE);
	}
	if (idx_z1<0 || idx_z1>nbox-1 || idx_z0<0 || idx_z0>nbox-1){
		printf("Error, Z indices in the Bcube interpolation is outside the allowed value of (0,%i)\n",nbox-1) ;
		exit(EXIT_FAILURE);
	}
	// map the indices to real coordinates
	double x0		= size_box*idx_x0/(nbox-1) - 0.5*size_box ;
	double x1		= size_box*idx_x1/(nbox-1) - 0.5*size_box ;
	double y0		= size_box*idx_y0/(nbox-1) - 0.5*size_box ;
	double y1		= size_box*idx_y1/(nbox-1) - 0.5*size_box ;
	double z0		= size_box*idx_z0/(nbox-1) - 0.5*size_box ;
	double z1		= size_box*idx_z1/(nbox-1) - 0.5*size_box ;
	
	// Calculate xd,yd,zd
	double xd		= (vector[0] - x0)/(x1 - x0) ;
	double yd		= (vector[1] - y0)/(y1 - y0) ;
	double zd		= (vector[2] - z0)/(z1 - z0) ;
	
	// interpolate along x
	double c00[3],c01[3],c10[3],c11[3] ;
	for (i=0;i<3;i++){
		c00[i]		= Bcube_ptr[idx_z0*nbox*nbox*3+idx_y0*nbox*3+idx_x0*3+i]*(1.0 - xd) + Bcube_ptr[idx_z0*nbox*nbox*3+idx_y0*nbox*3+idx_x1*3+i]*xd ;
		c01[i]		= Bcube_ptr[idx_z1*nbox*nbox*3+idx_y0*nbox*3+idx_x0*3+i]*(1.0 - xd) + Bcube_ptr[idx_z1*nbox*nbox*3+idx_y0*nbox*3+idx_x1*3+i]*xd ;
		c10[i]		= Bcube_ptr[idx_z0*nbox*nbox*3+idx_y1*nbox*3+idx_x0*3+i]*(1.0 - xd) + Bcube_ptr[idx_z0*nbox*nbox*3+idx_y1*nbox*3+idx_x1*3+i]*xd ;
		c11[i]		= Bcube_ptr[idx_z1*nbox*nbox*3+idx_y1*nbox*3+idx_x0*3+i]*(1.0 - xd) + Bcube_ptr[idx_z1*nbox*nbox*3+idx_y1*nbox*3+idx_x1*3+i]*xd ;
	}
	// interpolate along y
	double c0[3],c1[3];
	for (i=0;i<3;i++){
		c0[i]		= c00[i]*(1.0 - yd) + c10[i]*yd ;
		c1[i]		= c01[i]*(1.0 - yd) + c11[i]*yd ;
	}
	// interpolate along z
	for (i=0;i<3;i++){
		c[i]		= c0[i]*(1.0-zd) + c1[i]*zd ;
	}
}

void FilamentPaint_Bxyz(double r, double* Bcube_ptr,double size_box, int nbox, int idx_pix, double* local_triad, double* result){
	// Get the local magnetic field in r*hat(r)
	int i;
	double vec[3], localB[3] ;
	for (i=0 ; i<3 ; i++) vec[i] = r*local_triad[idx_pix*3*3 + 2*3 + i] ;
	FilamentPaint_TrilinearInterpolation(Bcube_ptr,size_box,nbox,vec,localB);
	// The result is a 1D array with size 4: Bx, By, Bz, norm2
	double Bx=0.0,By=0.0,Bz=0.0;
	for (i=0;i<3;i++){
		Bx	= Bx + localB[i]*local_triad[idx_pix*3*3 + 0*3 + i] ;
		By	= By + localB[i]*local_triad[idx_pix*3*3 + 1*3 + i] ;
		Bz	= Bz + localB[i]*local_triad[idx_pix*3*3 + 2*3 + i] ;
	}
	result[0]	= Bx;
	result[1]	= By; 
	result[2]	= Bz;
	result[3]	= Bx*Bx + By*By + Bz*Bz;
}

void FilamentPaint_RiemannIntegrator(double r1, double r2, double* rot_matrix, int idx_pix, double* local_triad, double* centers_arr, double* sizes_arr, double* Bcube_ptr, double size_box, int nbox, double* integ, double fpol0, double thetaH, int n_fil, int skip_Bcube, double* Bcube_center){
	//Integrator
	double a,b ;
	int i;
	if (r1 > r2){a=r2;b=r1;}
	else{a=r1;b=r2;}
	double deltar = (b-a)/24.0;
	double sumT=0.0;
	double sumQ=0.0;
	double sumU=0.0;
	double fpol = fpol0*pow(sin(thetaH),2) ;
	double density_0 = pow(sizes_arr[n_fil*3+2],-1.1);
	double result[4];
	if (skip_Bcube==1){
		double Bx=0.0,By=0.0,Bz=0.0;
		for (i=0;i<3;i++){
			Bx	= Bx + Bcube_center[i]*local_triad[idx_pix*3*3 + 0*3 + i] ;
			By	= By + Bcube_center[i]*local_triad[idx_pix*3*3 + 1*3 + i] ;
			Bz	= Bz + Bcube_center[i]*local_triad[idx_pix*3*3 + 2*3 + i] ;
		}
		result[0]	= Bx;
		result[1]	= By; 
		result[2]	= Bz;
		result[3]	= Bx*Bx + By*By + Bz*Bz;
	}
	for (i=0;i<25;i++){
		double density	= FilamentPaint_Density(a+i*deltar,rot_matrix,idx_pix,local_triad,centers_arr,sizes_arr,n_fil) ;
		if (skip_Bcube==0) FilamentPaint_Bxyz(a+i*deltar,Bcube_ptr,size_box,nbox,idx_pix,local_triad,result);
		sumT += density_0*density ;
		sumQ += fpol*density_0*density*(pow(result[1],2) - pow(result[0],2))/result[3];
		sumU += fpol*density_0*density*(-2.0)*result[1]*result[0]/result[3];
	}
	integ[0] =  sumT*deltar ;
	integ[1] =  sumQ*deltar ;
	integ[2] =  sumU*deltar ;
}

double FilamentPaint_RiemannIntegrator_OnlyIntensity(double r1, double r2, double* rot_matrix, int idx_pix, double* local_triad, double* centers_arr, double* sizes_arr, int n_fil){
	//Integrator
	double a,b ;
	int i;
	if (r1 > r2){a=r2;b=r1;}
	else{a=r1;b=r2;}
	double deltar = (b-a)/24.0;
	double sumT=0.0;
	double density_0 = pow(sizes_arr[n_fil*3+2],-1.1);
	for (i=0;i<25;i++){
		double density	= FilamentPaint_Density(a+i*deltar,rot_matrix,idx_pix,local_triad,centers_arr,sizes_arr,n_fil) ;
		sumT += density_0*density ;
	}
	return sumT*deltar ;
}