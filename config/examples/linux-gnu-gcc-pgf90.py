#!/usr/bin/env python

configure_options = [
  '--with-mpi-dir=/home/petsc/soft/linux-Ubuntu_8.04-ia32/mpich2-1.0.7-gcc-pgf90',
  '--download-f-blas-lapack=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
