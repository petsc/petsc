#!/usr/bin/env python

# ******* Currently not tested **********

configure_options = [
  '--with-mpi-dir=/home/petsc/soft/linux-alpha/mpich-1.2.6',
  '--with-blas-lapack-dir=/home/petsc/soft/linux-alpha/fblaslapack'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
