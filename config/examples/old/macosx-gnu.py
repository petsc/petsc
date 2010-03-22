#!/usr/bin/env python

# using gfortran from http://gcc.gnu.org/wiki/GFortranBinariesMacOS on Mac OSX 10.4

configure_options = [
  'CC=gcc',
  'FC=gfortran',
  '--with-python',
  '--with-shared=1',
  '--with-dynamic=1',
  '--download-mpich',
  '-download-mpich-pm=gforker'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
