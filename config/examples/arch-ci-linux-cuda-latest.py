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
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-precision=double',
    '--with-clanguage=c',
    '--with-mpi-dir=/software/mpich-43-main-0476502690-cuda130',
    '--with-cuda-dir=/usr/local/cuda-13.0',
    '--download-hypre=1',
    '--download-hypre-commit=b950aec946b3b8b85f2a66ce26f1cb13438ec903', # hypre-3.0 snapshot from Sep 12, 2025
    '--download-hypre-configure-arguments=--without-umpire',
    '--download-superlu_dist',
    '--with-cxx-dialect=17',
    '--with-strict-petscerrorcode',
  ]

  configure.petsc_configure(configure_options)
