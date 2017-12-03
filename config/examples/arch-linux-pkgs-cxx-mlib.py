#!/usr/bin/env python

configure_options = [
  '--with-single-library=0',
  '--with-clanguage=cxx',
  '--download-mpich=1',
  '--download-fblaslapack=1',
  '--download-hypre=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-suitesparse=1',
  '--download-triangle=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-parms=1',
  '--download-ctetgen=1',
  #'--download-elemental=1',
  #'--with-cxx-dialect=C++11', hdf5 conflicts with C++11
  '--download-spai=1',
  '--download-chaco=1',
  '--download-netcdf=1',
  '--download-hdf5=1',
  '--with-zlib=1',
  '--download-szlib=1',
  '--download-moab=1',
  '--download-petsc4py=1',
  '--download-mpi4py=1',
  '--download-saws',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
