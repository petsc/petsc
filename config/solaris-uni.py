#!/bin/env python

configure_options = [
  '--with-mpi=0',
  '--with-gnu-compilers=0'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
