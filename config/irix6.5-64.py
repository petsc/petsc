#!/usr/bin/env python

# ******* Currently not tested **********

configure_options = [
  '--with-cc=cc -64',
  '--with-fc=f90 -64',
  '--with-cxx=CC -64',
  '-ignoreWarnings',
  '-LDFLAGS=-Wl,-woff,84,-woff,85,-woff,113',
  '--with-f90-interface=IRIX',
  '--with-mpirun=mpirun'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
