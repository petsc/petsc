#!/usr/bin/env python

configure_options = [
  '--with-mpi-dir=/home/balay/soft/linux-fc-intel/mpich2-1.0.2p1',
  '--with-shared=0',
  '--with-vendor-compilers=intel'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
