#!/usr/bin/env python

configure_options = [
  '--with-cc=gcc',
  '--with-fc=gfortran', # http://hpc.sourceforge.net/
  '--with-cxx=g++',

  '--with-clanguage=cxx',
  '--with-debugging=0',

  #'-download-f-blas-lapack=1',
  '--download-mpich=1',
  '--download-plapack=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-triangle=1',
  '--download-spooles=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-blacs=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  #'--download-hdf5',
  '--download-sundials=1',
  '--download-prometheus=1',
  '--download-hypre=1',
  '--download-umfpack=1',
  '--download-chaco=1',
  '--download-spai=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
