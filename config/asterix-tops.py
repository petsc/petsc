#!/usr/bin/env python

configure_options = [
  '--with-mpi-dir=/home/balay/soft/linux-fc/mpich2-1.0.2p1',
  '--with-shared=1',
  '--with-mpi-shared=0',
  '--with-clanguage=c++',
  '--with-ccafe-dir=/home/balay/soft/cca-tools',
  '--with-babel-dir=/home/balay/soft/cca-tools'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
