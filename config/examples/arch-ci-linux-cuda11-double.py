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
    '--with-make-test-np=2',
    '--download-cmake', # kokkos needs 3.16+
    '--download-kokkos',
    '--download-kokkos-kernels',
    '--CUDAPPFLAGS=-std=c++14',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-cuda-dir=/usr/local/cuda-11.0',
  ]

  configure.petsc_configure(configure_options)
