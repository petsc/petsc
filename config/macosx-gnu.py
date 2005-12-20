#!/usr/bin/env python

# using gfortran from http://gcc.gnu.org/wiki/GFortranBinariesMacOS
configure_options = [
  'CC=gcc',
  'FC=gfortran',
  '--with-python',
  '--with-shared=1',
  '--with-dynamic=1',
  '--download-mpich',
  '-download-mpich-pm=gforker'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []


