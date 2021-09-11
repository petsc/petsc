#!/usr/bin/python3
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-kokkos',
    '--download-kokkos-kernels',
    '--with-cuda=1',
    '--with-cxx-dialect=c++14',
    '--with-debugging=0',
    # Need to use g++ as host compiler for NVCC (tested with 7.5.0) to compile kokkos lambdas
    'CUDAFLAGS=-ccbin g++',
    # Uses NVC (PGI) compilers for MPI wrappers
    'CFLAGS=-g -nomp -tp p7-64',
    'CXXFLAGS=-g -nomp -tp p7-64',
    'FFLAGS=-g -nomp -tp p7-64',
    'PETSC_ARCH=arch-nvhpc',
  ]
  configure.petsc_configure(configure_options)
