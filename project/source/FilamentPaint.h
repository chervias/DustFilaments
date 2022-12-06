#include <stdbool.h>

typedef struct tPoint{
	long double x, y;
	int index;
}Point;

bool ccw(const Point *a, const Point *b, const Point *c) ;
int comparePoints(const void *lhs, const void *rhs) ;
void fatal(const char* message) ;
void* xmalloc(size_t n) ;
void* xrealloc(void* p, size_t n) ;
void printPoints(const Point* points, int len) ;
Point* convexHull(Point p[], int len, int* hsize) ;
void FilamentPaint_RotationMatrix(double* angles_arr, int n_fil, double* rot_matrix);
void FilamentPaint_InvertRotMat(double* rot_matrix, double* inv_rot_matrix);
void FilamentPaint_xyzVertices(double* rot_matrix, double* sizes_arr, double* centers_arr, double Size, int* isInside,double* xyz_vertices, int n_fil);
void FilamentPaint_xyzNormalToFaces(double* rot_matrix, double* xyz_normal_to_faces);
void FilamentPaint_xyzFaces(double* xyz_vertices, double* xyz_faces);
void FilamentPaint_xyzEdgeVectors(double* xyz_faces, double* xyz_edges);
void FilamentPaint_xyzEdgeVectorsUnit(double* xyz_edges, double* xyz_edges_unit);
void FilamentPaint_DoQueryPolygon(int nside, double* xyz_vertices, long* ipix, int* nipix, double* centers_arr, int* sucess, int n_fil);
void FilamentPaint_DoLocalTriad(long* ipix, int nipix, int nside, double* localtriad);
void FilamentPaint_CalculateDistances(double* xyz_normal_to_faces, double* xyz_faces, double* xyz_edges, double* xyz_edges_unit, int idx_pix, double* local_triad, double* distances, int* skip_pixel);
void FilamentPaint_RiemannIntegrator(double r1, double r2, double* rot_matrix, int idx_pix, double* local_triad, double* centers_arr, double* sizes_arr, double* Bcube_ptr, double size_box, int nbox, double* integ, double fpol0, double thetaH, int n_fil, int skip_Bcube, double* Bcube_center);
double FilamentPaint_RiemannIntegrator_OnlyIntensity(double r1, double r2, double* rot_matrix, int idx_pix, double* local_triad, double* centers_arr, double* sizes_arr, int n_fil);
double FilamentPaint_Density(double r, double* rot_matrix, int idx_pix, double* local_triad, double* centers_arr, double* sizes_arr, int n_fil);
void FilamentPaint_TrilinearInterpolation(double* Bcube_ptr, double size_box, int nbox, double* vector, double* c);
void FilamentPaint_Bxyz(double r, double* Bcube_ptr,double size_box, int nbox, int idx_pix, double* local_triad, double* result);
