#!/usr/bin/env python

# Note: Intel 7.1 Fortran cannot work with g++ 3.3
configure_options = [
  '--with-cc=gcc',
  '--with-fc=ifc',
  '--with-cxx=g++',
  '--with-scalar-type=complex',
  '--with-blas-lapack-dir=/home/petsc/soft/linux-rh73-intel/fblaslapack',
  '--with-mpi=0'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
