#!/usr/bin/env python

configure_options = [
  '--with-cc=cc -64',
  '--with-fc=f90 -64',
  '--with-cxx=CC -64',
  '-ignoreWarnings',
  '-LDFLAGS=-Wl,-woff,84,-woff,85,-woff,113',
  '--with-f90-header=90_IRIX.h',
  '--with-f90-source=f90_IRIX.c',
  '--with-mpirun=mpirun'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
