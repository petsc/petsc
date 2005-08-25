#!/usr/bin/env python

configure_options = [
  '--with-shared=1',
  '--download-mpich=1',
  '--download-mpich-pm=gforker'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
