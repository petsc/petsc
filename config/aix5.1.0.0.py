#!/usr/bin/env python

configure_options = [
  '--can-execute=0',
  '--sizeof_void_p=4',
  '--sizeof_short=2',
  '--sizeof_int=4',
  '--sizeof_long=4',
  '--sizeof_long_long=8',
  '--sizeof_float=4',
  '--sizeof_double=8',
  '--bits_per_byte=8',
  '--sizeof_MPI_Comm=4',
  '--sizeof_MPI_Fint=4',
  '--with-f90-header=f90_rs6000.h',
  '--with-f90-source=f90_rs6000.c'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
    
