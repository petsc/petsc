#!/usr/bin/env python

configure_options = [
  # -lg2c messes up shared libraries
  '--with-shared=0',
  '--with-mpi-dir=/home/petsc/soft/linux-ia64/mpich-1.2.5.2'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
