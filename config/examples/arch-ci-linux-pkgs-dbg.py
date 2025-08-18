#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-coverage',
  #'--download-mpich', use system MPI as elemental fails with this
  '--download-fblaslapack',
  '--download-hypre',
  '--download-cmake',
  '--download-metis',
  '--download-parmetis',
  '--download-ptscotch',
  '--download-suitesparse',
  '--download-triangle',
  '--download-triangle-build-exec',
  '--download-superlu',
  '--download-superlu_dist',
  '--download-scalapack',
  '--download-mumps',
  # '--download-elemental', # disabled since its maxCxxVersion is c++14, but Kokkos-4.0's minCxxVersion is c++17
  '--download-spai',
  '--download-moab',
  '--download-parms',
  '--download-chaco',
  '--download-fftw',
  '--download-hwloc',
  '--download-ctetgen',
  '--download-netcdf',
  '--download-hdf5',
  '--with-zlib',
  '--download-exodusii',
  '--with-exodusii-fortran-bindings',
  '--download-pnetcdf',
  '--download-party',
  '--download-yaml',
  '--download-ml',
  '--download-sundials2',
  '--download-p4est',
  '--download-eigen',
  '--download-pragmatic',
  '--download-mmg',
  '--download-parmmg',
  '--download-hpddm',
  '--download-bamg',
  '--download-htool',
  '--download-mfem',
  '--download-glvis',
  '--with-opengl',
  '--download-revolve',
  '--download-cams',
  '--download-slepc',
  '--download-kokkos',
  '--download-kokkos-cxx-std-threads',
  '--download-kokkos-kernels',
  '--with-dmlandau-3d',
  '--with-strict-petscerrorcode',
  '--download-mpi4py',
  '--with-petsc4py',
  '--with-debugging',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
