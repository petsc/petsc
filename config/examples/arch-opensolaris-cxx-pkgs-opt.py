#!/usr/bin/env python

configure_options = [
  '--with-debugger=/bin/true',
  '--with-clanguage=cxx',
  '--with-debugging=0',

  '--download-mpich=1',
  #'--with-mpi-dir=/export/home/petsc/soft/mpich2-1.2.1p1',
  '--with-c2html=0',

  #'-download-f-blas-lapack=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-triangle=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-blacs=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  #'--download-hdf5',
  #'--download-sundials=1',
  #'--download-hypre=1',
  '--download-umfpack=1',
  '--download-chaco=1',
  '--download-spai=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
