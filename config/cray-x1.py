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
  
  '--with-fortran-kernels=generic',
  '--with-blas-lapack-lib=sci',
  '--with-f90-interface=cray_x1',
  
  '--with-batch=1',
  '--sizeof_void_p=8',
  '--with-memcmp-ok',
  '--sizeof_long=8',
  '--sizeof_MPI_Comm=4',
  '--sizeof_double=8',
  '--sizeof_int=4',
  '--with-endian=big',
  '--bits_per_byte=8',
  '--sizeof_MPI_Fint=4',
  '--sizeof_long_long=8',
  '--sizeof_float=4',
  '--sizeof_short=2'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
