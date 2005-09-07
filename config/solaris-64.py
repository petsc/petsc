#!/bin/env python

configure_options = [
  '--with-64-bit-pointers',
  '--with-mpi-compilers=0',
  '--with-vendor-compilers=solaris',
  '--with-mpi-dir=/home/petsc/soft/solaris-9-64/mpich-1.2.5/'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
