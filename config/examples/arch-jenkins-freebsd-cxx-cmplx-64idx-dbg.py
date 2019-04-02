#!/usr/bin/env python

configure_options = [
  '--with-cc=/home/petsc/soft/mpich-3.3b1/bin/mpicc',
  '--with-fc=/home/petsc/soft/mpich-3.3b1/bin/mpif90',
  '--with-cxx=/home/petsc/soft/mpich-3.3b1/bin/mpicxx',
  '--with-debugging=1',
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  '--with-clanguage=cxx',
  '--with-scalar-type=complex',
  '--with-cxx-dialect=C++11',
  '--with-64-bit-indices=1',
  'DATAFILESPATH=/home/petsc/datafiles',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
