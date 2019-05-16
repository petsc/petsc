#!/usr/bin/env python

configure_options = [
  'CFLAGS=-std=c89 -pedantic -Wno-long-long -Wno-overlength-strings',
  '--with-cc=/home/petsc/soft/mpich-3.3b1/bin/mpicc',
  '--with-fc=/home/petsc/soft/mpich-3.3b1/bin/mpif90',
  '--with-cxx=/home/petsc/soft/mpich-3.3b1/bin/mpicxx',
  '--with-precision=single',
  '--with-debugging=0',
  'DATAFILESPATH=/home/petsc/datafiles',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
