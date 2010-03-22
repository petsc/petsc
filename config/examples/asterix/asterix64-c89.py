#!/usr/bin/env python

# Test c89 std code compliance
  
configure_options = [
  '--with-cc=gcc -std=c89',
  '--with-fc=gfortran',
  '--download-mpich=1',
  '--download-mpich-pm=gforker',
  '--download-hypre=1',
  '--download-superlu-dist=1',
  '--download-plapack=1',
  '--with-shared=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
