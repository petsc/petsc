#!/usr/bin/env python

configure_options = [
  '--with-mpi-dir=/home/petsc/soft/linux-debian_sarge-gcc-pgf90/mpich-1.2.6',
  '--download-f-blas-lapack=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
