#!/bin/env python

configure_options = [
  '--with-cc=cc',
  '--with-fc=f90',
  '--with-cxx=CC',
  '--with-f90-header=f90_solaris.h',
  '--with-f90-source=f90_solaris.c',
  '--with-mpi-dir=/home/petsc/soft/solaris-9/mpich-1.2.5/'
  # '--with-mpi-include=/home/petsc/soft/solaris-9/mpich-1.2.5/include',
  # '--with-mpi-lib=[/home/petsc/soft/solaris-9/mpich-1.2.5/lib/libmpich.a,libsocket.a,libnsl.a,librt.a,libaio.a]'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
