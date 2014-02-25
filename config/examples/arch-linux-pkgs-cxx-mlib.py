#!/usr/bin/env python

configure_options = [
  '--with-single-library=0',
  '--with-clanguage=cxx',
  '--download-mpich=1',
  '--download-f-blas-lapack=1',
  '--download-hypre=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-umfpack=1',
  '--download-triangle=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  #'--download-elemental=1',
  #'--with-cxx-dialect=C++11', hdf5 conflicts with C++11
  '--download-spai=1',
  '--download-chaco=1',
  '--download-netcdf=1',
  '--download-hdf5=1',
  '--download-moab=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
