#!/usr/bin/env python

configure_options = [
  '--with-mpi-dir=/home/balay/soft/linux-fc-gcc4/mpich2-1.0.2p1',
  '--with-shared=1',
  '--with-debugging=0',
  '--with-log=0'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
