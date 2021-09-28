#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-cc=clang',
  '--with-cxx=clang++',
  '--with-fc=gfortran',
  '--with-debugging=1',
  '--debugLevel=4',
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  '--download-openmpi=1',
  '--download-fblaslapack=1',
  '--with-openmp=1',
  '--download-hwloc=1',
  #'--download-hypre=1', disabled as hypre produces wrong results when openmp is enabled
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-suitesparse=1',
  '--download-triangle=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-scalapack=1',
  '--download-strumpack=1',
  '--download-mumps=1',
  '--download-elemental=1',
  '--download-spai=1',
  '--download-parms=1',
  '--download-kokkos=1',
  '--download-kokkos-kernels=1',
  '--with-kokkos-init-warnings=0', # we want to avoid "Kokkos::OpenMP::initialize WARNING: You are likely oversubscribing your CPU cores" in test output
  '--download-chaco=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
