#!/usr/bin/env python
  
configure_options = [
  '--with-cc=gcc -std=c89',
  '--download-mpich=1',
  '--download-hypre=1',
  '--download-spooles=1',
  '--download-superlu-dist=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
