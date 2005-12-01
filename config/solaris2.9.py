#!/bin/env python

configure_options = [
  '--with-cc=cc',
  '--with-fc=f90',
  '--with-f90-header=f90_solaris.h',
  '--with-f90-source=f90_solaris.c',
  '--download-mpich=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
