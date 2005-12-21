#!/usr/bin/env python

# ******* Currently not tested **********

configure_options = [
  '--download-mpich=1',
  '--download-mpich-pm=gforker',  
  '--with-gnu-compilers=0',
  '--with-vendor-compilers=ibm',
  # c++ doesn't work yet
  '--with-cxx=0'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
