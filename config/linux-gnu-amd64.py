#!/usr/bin/env python

configure_options = [
  '--download-mpich=1',
  '--download-mpich-pm=forker',
  '--download-f-blas-lapack=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = [
  '--with-debugging=0',
  '--with-lang=c++']
