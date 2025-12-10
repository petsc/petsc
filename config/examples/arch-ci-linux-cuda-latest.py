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
    '--with-mpi-dir=/software/mpich-5.0.0b1-cuda131',
    '--with-cuda-dir=/usr/local/cuda-13.1',
    #'--download-umpire', #'hypre' reserves 4G VRAM for each MPI process
    '--download-hypre=1',
    '--download-superlu_dist',
    '--with-cxx-dialect=17',
    '--with-strict-petscerrorcode',
  ]

  configure.petsc_configure(configure_options)
