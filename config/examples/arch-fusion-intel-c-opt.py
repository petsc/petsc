#!/usr/bin/env python

configure_options = [
  '--with-blas-lapack-dir=/soft/mkl/10.2.2.025/lib/em64t',
  '--with-mpi-dir=/soft/mvapich2/1.4.1-intel-11.1.064/bin',
  '--with-debugging=0',
  '-COPTFLAGS=-O3 -xHost',
  '-CXXOPTFLAGS=-O3 -xHost',
  '-FOPTFLAGS=-O3 -xHost',
  ]

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)

