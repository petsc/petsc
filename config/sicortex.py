#!/usr/bin/env python
#
configure_options = [
  '--with-cc=scmpicc --pathscale',
  '--with-fc=scmpif90 --pathscale',
  '--LIBS=-lpathfstart -lpathfortran -lpathfstart',
  '--with-fortranlib-autodetect=0',

  '--with-batch=1',
  '--with-mpi-shared=0',

  '--with-memcmp-ok',
  '--sizeof_char=1',
  '--sizeof_void_p=8',
  '--sizeof_short=2',
  '--sizeof_int=4',
  '--sizeof_long=8',
  '--sizeof_long_long=8',
  '--sizeof_float=4',
  '--sizeof_double=8',
  '--bits_per_byte=8',
  '--sizeof_MPI_Comm=4',
  '--sizeof_MPI_Fint=4',

  '--download-f-blas-lapack=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
    
