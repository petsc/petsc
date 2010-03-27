#!/usr/bin/env python

# python on cray-x1 is broken - this script is run
# on the linux-compile-node for x1 [robin]

configure_options = [
  '--with-cc=cc',
  '--with-fc=ftn',
  '--with-cxx=0',
  '--with-shared=0',
  '--with-debugging=0',
  '-COPTFLAGS=-O3',
  '-FOPTFLAGS=-O3',
  
  '--with-fortran-kernels=1',
  '--with-blas-lapack-lib=sci',
  
  '--with-batch=1',
  '--known-mpi-shared=0',
  '--known-sizeof-void-p=8',
  '--known-sizeof-char=1',
  '--known-memcmp-ok',
  '--known-sizeof-long=8',
  '--known-sizeof-size_t=8',
  '--known-sizeof-MPI_Comm=4',
  '--known-sizeof-double=8',
  '--known-sizeof-int=4',
  '--known-endian=big',
  '--known-bits-per-byte=8',
  '--known-sizeof-MPI_Fint=4',
  '--known-mpi-long-double=1',
  '--known-sizeof-long-long=8',
  '--known-sizeof-float=4',
  '--known-sizeof-short=2'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
