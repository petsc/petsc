#!/usr/bin/env python

configure_options = [
  '--with-cc=gcc',
  '--with-fc=g95',
  '--with-fc=0',
  '--with-python',
  '--with-shared=1',
  '--with-dynamic=1',
  '--with-petsc-dynamic=1',
  '--download-mpich',
  '-download-mpich-pm=gforker'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []


