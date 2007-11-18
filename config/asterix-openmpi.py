#!/usr/bin/env python

configure_options = [
  '--with-mpi-dir=/home/balay/soft/linux-fc-openmpi/openmpi-1.2.1',
  '--with-clanguage=cxx',
  '--with-debugging=0',
  '--with-log=0',
  '--with-shared=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
