#!/usr/bin/env python
  
configure_options = [
  '--with-mpi-dir=/software/mpich-1.2.5.2'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
