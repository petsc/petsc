#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-make-test-np=3',
    'COPTFLAGS=-g -O0',
    'FOPTFLAGS=-g -O0',
    'CXXOPTFLAGS=-g -O0',
    '--with-mpiexec-tail='+os.path.realpath("lib/petsc/bin/cuda_use_first_gpu.sh"),
    '--with-coverage',
    '--download-suitesparse',
    '--download-mumps',
    '--download-scalapack',
    '--download-chaco',
    '--download-ctetgen',
    '--download-exodusii',
    '--with-exodusii-fortran-bindings=1',
    '--download-pnetcdf',
    '--download-generator',
    '--download-hdf5',
    '--download-zlib',
    '--download-metis',
    '--download-ml',
    '--download-netcdf',
    '--download-parmetis',
    '--download-triangle',
    '--download-triangle-build-exec',
    '--download-p4est',
    '--download-mfem',
    '--with-cuda',
    '--with-openmp',
    '--with-shared-libraries',
    '--download-magma',
    '--download-kblas',
    '--download-h2opus',
    # '--download-kokkos', # Kokkos-5.0 requires c++20 and cuda-12.2 or above, which break many packages
    # '--download-kokkos-kernels',
    '--download-hwloc',
    '--download-umpire',
    '--download-hypre',
    '--download-caliper',
    '--download-raja',
    '--download-amgx',
    '--download-zfp',
    '--download-butterflypack',
    '--download-strumpack',
    '--with-strict-petscerrorcode',
  ]

  configure.petsc_configure(configure_options)
