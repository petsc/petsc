#!/usr/bin/env python

configure_options = [
  '--with-mpi-dir=/sandbox/petsc/soft/mpich-1.2.5.2',
  '--with-cxx=g++' # mpiCC does not work
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []


