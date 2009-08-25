#!/usr/bin/env python


configure_options = [
  '--with-mpi-dir=/home/petsc/soft/linux-rh73-mpich2/mpich2-0.971-CVS-200408131639',
  '--with-cxx=g++',
  '--with-matlab=0'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
