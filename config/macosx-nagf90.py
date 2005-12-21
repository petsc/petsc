#!/usr/bin/env python

# ******* Currently not tested **********

configure_options = [
  '--download-mpich=1',
  '--download-mpich-pm=gforker',
  '--with-fc=f95 -w',
  '--with-cc=gcc',
  # c++ doesn't work yet
  '--with-cxx=0'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
