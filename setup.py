import os
from setuptools import setup, Extension

PACKAGE_NAME = 'DustFilaments'

module =  Extension('FilamentPaint',
sources = ['source/FilamentPaint.c','source/FilamentPaint_mod.c','source/query_polygon_wrapper.cpp'],
include_dirs = ['source'],
libraries=['gsl','gslcblas','healpix_cxx','cxxsupport','sharp','fftpack','c_utils','cfitsio'],
library_dirs = ["lib"],
extra_compile_args=['-fPIC','-Wall','-g','-fopenmp'],
extra_link_args=['-fopenmp'],
)

setup(name=PACKAGE_NAME,
      version='1.0',
      packages=[PACKAGE_NAME],
      ext_package=PACKAGE_NAME,
      ext_modules=[module],
      include_package_data=True)