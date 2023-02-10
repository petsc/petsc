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
    '--with-make-test-np=15',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-cuda=1',
    '--with-openmp',
    '--with-threadsafety',
    '--download-kokkos',
    '--download-kokkos-kernels',
    '--download-kokkos-commit=36f65d03f48e504e63ad124f7f5cf74dd46af130', # release-candidate-4.0.0, Feb. 7 2023.  Remove it once we upgrade the default Kokkos version to 4.0
    '--download-kokkos-kernels-commit=697c4169f73ec9b316a30d7e0060013c86672d59', # release-candidate-4.0.0, Feb. 7 2023.
    '--with-strict-petscerrorcode',
  ]

  configure.petsc_configure(configure_options)
