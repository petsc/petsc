#!/usr/bin/env python

configure_options = [
  # NAG f90 doesn't compile NETLIB lapack - so use gcc compiled libraries
  '--with-blas-lapack-lib=[liblapack.a,libblas.a,libg2c.a]',
  '--with-mpi-dir=/home/petsc/soft/linux-debian_sarge-nagf90/mpich2-1.0.2p1',
  '--with-shared=0',
  '--with-scalar-type=complex',
  '--download-superlu_dist=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
