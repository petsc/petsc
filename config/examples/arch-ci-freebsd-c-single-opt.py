#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  'CFLAGS=-std=c99 -pedantic -Wcast-function-type -Wno-long-long -Wno-overlength-strings',
  '--with-precision=single',
  '--with-debugging=0',
  '--with-mpi-dir=/home/svcpetsc/soft/mpich-4.2.2',
  '--download-superlu_dist',
  '--download-metis',
  '--download-parmetis',
  '--download-hypre',
  '--with-strict-petscerrorcode',
]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
