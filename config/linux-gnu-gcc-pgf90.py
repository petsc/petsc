#!/usr/bin/env python

# Note: Intel 7.1 Fortran cannot work with g++ 3.3
configure_options = [
  '--with-cc=gcc',
  '--with-fc=pgf90',
  '--with-cxx=0',
  '--download-mpich=1',
  '--download-mpich-pm=gforker',
  '--download-f-blas-lapack=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
