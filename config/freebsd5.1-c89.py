#!/usr/bin/env python
  
configure_options = [
  '--with-cc=gcc -std=c89',
  '--with-fc=f77',
  '--download-mpich=1',
  '--download-mpich-pm=gforker',
  '--download-hypre=1',
  '--download-superlu-dist=1',
  '--with-shared=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
