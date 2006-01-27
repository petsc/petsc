#!/usr/bin/env python

configure_options = [
  '--with-mpi-dir=/home/balay/soft/linux-fc/mpich2-1.0.2p1',
  '--with-shared=1',
  '--with-clanguage=c++',
  '--with-ccafe-dir=/home/balay/soft/cca-tools-0.5.9',
  '--with-babel-dir=/home/balay/soft/cca-tools-0.5.9'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
