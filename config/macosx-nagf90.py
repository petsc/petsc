#!/usr/bin/env python

configure_options = [
  '--download-mpich=1',
  '--download-mpich-pm=forker',
  '--with-fc=f95 -w',
  '--with-cc=gcc',
  # c++ doesn't work yet
  '--with-cxx=0'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []


