#!/usr/bin/env python

# Notes:
#   --with-batch=1 is required for IBM MPI. However all batch test values are specified.

configure_options = [
  '--with-batch=1',
  '--with-endian=big',
  '--with-memcmp-ok',
  '--sizeof_char=1',
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
  '--with-f90-interface=rs6000'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
