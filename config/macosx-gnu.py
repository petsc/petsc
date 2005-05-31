#!/usr/bin/env python

configure_options = [
  '--with-python',
  '--with-shared=1',
  '--with-dynamic',
  '--download-mpich',
  '-download-mpich-pm=forker'  
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []


