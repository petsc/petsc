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
    '--with-mpi-dir=/home/users/balay/soft/mpich-3.4.2',
    '--with-blaslapack-dir=/home/users/balay/soft/fblaslapack',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-cuda=0',
    '--with-hip=1',
    '--with-hipc=/opt/rocm/bin/hipcc',
    '--with-hip-dir=/opt/rocm',
    '--with-precision=double',
    '--with-clanguage=c',
    '--download-kokkos',
    '--download-kokkos-kernels',
    #'--download-hypre',
    #'--download-hypre-configure-arguments=--enable-unified-memory',
    #'--with-hypre-gpuarch=gfx908',
    '--download-magma',
    '--with-magma-fortran-bindings=0',
    '--with-magma-gputarget=gfx908',
  ]

  configure.petsc_configure(configure_options)
