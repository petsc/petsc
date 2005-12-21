#!/bin/env python

configure_options = [
  '--with-f90-header=f90_solaris.h',
  '--with-f90-source=f90_solaris.c',
  '--with-mpi-dir=/home/petsc/soft/solaris-9-lam/lam-6.5.8',
  '--with-mpirun=/sandbox/petsc/petsc-dev/bin/mpirun.lam'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
