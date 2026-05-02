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
    '--with-make-test-np=2',
    '--download-openmpi=1',
    '--download-hypre=1',
    '--download-hwloc=1',
    '--download-kokkos=1',
    '--download-kokkos-commit=1557870d70d5ac0a636d3e8873d5b4ce1bb0375b', # develop as of 5/1/2026
    '--download-kokkos-kernels=1',
    '--download-kokkos-kernels-commit=90ce916124f86173481944db6c810d67e8978bd0', # develop as of 5/1/2026
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-64-bit-indices=1',
    '--with-cuda-dir=/usr/local/cuda-12.6',
    '--with-precision=double',
    '--with-clanguage=c',
    # Note: If using nvcc with a host compiler other than the CUDA SDK default for your platform (GCC on Linux, clang
    # on Mac OS X, MSVC on Windows), you must set -ccbin appropriately in CUDAFLAGS, as in the example for PGI below:
    # 'CUDAFLAGS=-ccbin pgc++',
    '--with-strict-petscerrorcode',
    '--with-coverage',
  ]

  configure.petsc_configure(configure_options)
