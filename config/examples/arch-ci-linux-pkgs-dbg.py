#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-coverage',
  #'--download-mpich=1', use system MPI as elemental fails with this
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
  # '--download-elemental=1', # disabled since its maxCxxVersion is c++14, but Kokkos-4.0's minCxxVersion is c++17
  '--download-spai=1',
  # '--download-moab=1', # disabled since its maxCxxVersion is c++14, but Kokkos-4.0's minCxxVersion is c++17
  '--download-parms=1',
  '--download-chaco=1',
  '--download-fftw=1',
  '--download-pastix=1',
  '--download-hwloc=1',
  '--download-ctetgen',
  '--download-netcdf',
  '--download-hdf5',
  '--with-zlib=1',
  '--download-med=1',
  '--download-exodusii',
  '--download-pnetcdf',
  '--download-party',
  '--download-yaml',
  '--download-ml',
  '--download-sundials2',
  '--download-p4est=1',
  '--download-eigen',
  '--download-pragmatic',
  '--download-mmg=1',
  '--download-parmmg=1',
  '--download-hpddm=1',
  '--download-bamg=1',
  '--download-htool=1',
  '--download-mfem=1',
  '--download-glvis=1',
  '--with-opengl=1',
  '--download-revolve=1',
  '--download-cams=1',
  '--download-slepc',
  '--download-kokkos',
  '--download-kokkos-kernels',
  '--with-dmlandau-3d',
  '--with-strict-petscerrorcode',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
