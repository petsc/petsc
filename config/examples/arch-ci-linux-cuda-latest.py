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
    '--with-mpi-dir=/home/software/mpich-5.0.1-cuda133',
    '--with-cuda-dir=/home/software/spack-cuda/opt/spack/linux-nehalem/cuda-13.3.0-4vp2dphv6qkqtug72w67oghxthozvul4',
    '--download-umpire',
    '--with-cuda-arch=80,86',
    '--download-hypre=1',
    '--download-superlu_dist',
    '--with-cxx-dialect=17',
    '--with-strict-petscerrorcode',
  ]

  configure.petsc_configure(configure_options)
