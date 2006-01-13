#!/usr/bin/env python

# Notes:
#   --with-batch=1 is required for IBM MPI. However all batch test values are specified.
#   64 bit compilers/tools are specified for eg, with option: --with-cc='mpcc -q64'.

configure_options = [
  '--with-batch=1',
  '--with-memcmp-ok',
  '--sizeof_void_p=8',
  '--sizeof_char=1',
  '--sizeof_short=2',
  '--sizeof_int=4',
  '--sizeof_long=8',
  '--sizeof_long_long=8',
  '--sizeof_float=4',
  '--sizeof_double=8',
  '--bits_per_byte=8',
  '--sizeof_MPI_Comm=8',
  '--sizeof_MPI_Fint=4',
  '--with-f90-interface=rs6000',
  '--with-cc=mpcc -q64',
  '--with-fc=mpxlf -q64',
  '--with-ar=/usr/bin/ar -X64'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
