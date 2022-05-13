#!/usr/bin/python

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
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--download-suitesparse',
    '--download-mumps',
    '--download-scalapack',
    '--download-chaco',
    '--download-ctetgen',
    '--download-exodusii',
    '--download-pnetcdf',
    '--download-generator',
    '--download-hdf5',
    '--download-zlib=1',
    '--download-metis',
    '--download-ml',
    '--download-netcdf',
    '--download-parmetis',
    '--download-triangle',
    '--download-p4est',
    '--with-cuda',
    '--with-shared-libraries',
    '--download-magma',
    '--with-magma-fortran-bindings=0',
    '--download-kblas',
    '--download-h2opus',
    '--download-kokkos',
    '--download-kokkos-kernels',
    '--download-hwloc',
    '--download-hypre',
    '--download-hypre-configure-arguments=--enable-unified-memory',
    '--download-raja',

  ]

  configure.petsc_configure(configure_options)
