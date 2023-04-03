#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--useThreads=0', # for some reason cmake hangs when invoked from configure on bsd?

  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',

  '--with-clanguage=cxx',
  '--with-scalar-type=complex',

  #'-download-fblaslapack=1',
  '--download-mpich=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-triangle=1',
  #'--download-superlu=1',
  #'--download-superlu_dist=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-hdf5',
  '--with-zlib=1',
  # '--download-elemental=1', # disabled since its maxCxxVersion is c++14, but Kokkos-4.0's minCxxVersion is c++17
  #'--download-sundials2=1',
  #'--download-hypre=1',
  #'--download-suitesparse=1',
  #'--download-chaco=1',
  #'--download-spai=1',
  '--download-p4est=1',
  '--with-mpi-f90module-visibility=0',
  '--download-kokkos=1',
  '--download-kokkos-kernels=1',
  '--with-ssl=1',
  '--with-strict-petscerrorcode=0',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
