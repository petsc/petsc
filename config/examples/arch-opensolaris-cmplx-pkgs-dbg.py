#!/usr/bin/env python

configure_options = [
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  #'--with-debugger=/bin/true',
  '--with-scalar-type=complex',
  'FFLAGS=-ftrap=%none',
  #'--with-clanguage=cxx', # solaris C++ compiler behave differently with PETSC_EXTERN stuff - and breaks

  # mpich does not build with -g - compiler bug?
  #'--download-mpich=1',
  '--with-mpi-dir=/export/home/petsc/soft/mpich-3.3.1',
  '--with-c2html=0',

  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-triangle=1',
  '--download-superlu=1',
  #'--download-superlu_dist=1',  # breaking with 'mpicc -lmpi'
  '--download-fblaslapack=1', # -lsunperf is insufficient for scalapack
  '--download-scalapack=1',
  '--download-mumps=1',
  #'--download-elemental=1', breaks with solaris compilers
  #'--download-hdf5',
  #'--download-sundials=1',
  #'--download-hypre=1',
  #'--download-suitesparse=1',
  #'--download-chaco=1',
  #'--download-spai=1',

  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
