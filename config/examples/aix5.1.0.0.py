#!/usr/bin/env python

# Notes:
#   --with-batch=1 is required for IBM MPI. However all batch test values are specified.

configure_options = [
  '--with-batch=1',
  '--known-mpi-shared=0',
  '--known-endian=big',
  '--known-memcmp-ok',
  '--known-sizeof-char=1',
  '--known-sizeof-void-p=4',
  '--known-sizeof-short=2',
  '--known-sizeof-int=4',
  '--known-sizeof-long=4',
  '--known-sizeof-size_t=4',
  '--known-sizeof-long-long=8',
  '--known-sizeof-float=4',
  '--known-sizeof-double=8',
  '--known-bits-per-byte=8',
  '--known-sizeof-MPI_Comm=4',
  '--known-sizeof-MPI_Fint=4',
  '--known-mpi-long-double=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
