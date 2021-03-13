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
    '--with-cc=/scratch/soft/mpich/bin/mpicc',
    '--with-cxx=/scratch/soft/mpich/bin/mpicxx',
    '--with-fc=/scratch/soft/mpich/bin/mpif90',
    'LDFLAGS=-L/opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/lib -lquadmath',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-cuda=0',
    '--with-hip=1',
    '--with-hipc=hipcc',
    '--with-hip-dir=/opt/rocm',
    '--with-precision=double',
    '--with-clanguage=c',
    '--download-fblaslapack=1',
  ]

  configure.petsc_configure(configure_options)
