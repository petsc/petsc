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
    '--with-mpi-dir=/home/petsc/soft/openmpi-4.0.2-cuda',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-cuda=0',
    '--with-hip=1',
    '--with-hipcc=hipcc',
    '--with-hip-dir=/opt/rocm',
    '--with-precision=double',
    '--with-clanguage=c',
  ]

  configure.petsc_configure(configure_options)
