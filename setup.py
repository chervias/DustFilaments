#from setuptools import setup, Extension
from setuptools import setup, find_packages, Extension
import numpy as np

module =  Extension('DustFilaments.FilamentPaint',
sources = ['project/source/FilamentPaint.c','project/source/FilamentPaint_mod.c','project/source/query_polygon_wrapper.cpp'],
include_dirs = ['project/source',"/global/homes/c/chervias/Software/include","/global/homes/c/chervias/Software/Healpix_3.80/include",np.get_include()],
#libraries=['gsl','healpix_cxx','cxxsupport','c_utils'],
libraries=['gsl','gslcblas','healpix_cxx'],
library_dirs = ["/global/homes/c/chervias/Software/lib","/global/homes/c/chervias/Software/Healpix_3.80/lib","lib"],
extra_compile_args=['-fPIC','-Wall','-g','-fopenmp','-std=c99'],
extra_link_args=['-fopenmp'],
)

setup(name='DustFilaments',
version='1.0',
package_dir = {'': 'project',},
packages=['DustFilaments'],
#packages=find_packages(),
#ext_package=PACKAGE_NAME,
ext_modules=[module],
#include_package_data=True,
)
