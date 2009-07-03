#!/usr/bin/env python

configure_options = [
  '--with-mpi-dir=/home/balay/soft/linux64/mpich2-1.1',
  '--with-shared=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
