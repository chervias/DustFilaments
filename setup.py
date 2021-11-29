from setuptools import setup, Extension

PACKAGE_NAME = 'DustFilaments'

module =  Extension(PACKAGE_NAME+'.FilamentPaint',
sources = ['source/FilamentPaint.c','source/FilamentPaint_mod.c','source/query_polygon_wrapper.cpp'],
include_dirs = ['source'],
libraries=['gsl','healpix_cxx','cxxsupport','c_utils'],
library_dirs = ["lib"],
extra_compile_args=['-fPIC','-Wall','-g','-fopenmp','-std=c99'],
extra_link_args=['-fopenmp'],
)

setup(name=PACKAGE_NAME,
      version='1.0',
      packages=[PACKAGE_NAME],
      ext_package=PACKAGE_NAME,
      ext_modules=[module],
      include_package_data=True)