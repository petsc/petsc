#!/usr/bin/env python

configure_options = [
  '--with-blas-lapack-lib=[/home/petsc/soft/linux-debian_sarge-nagf90/fblaslapack/libflapack.a,libfblas.a,libg2c.a]',
  '--with-mpi-dir=/home/petsc/soft/linux-debian_sarge-nagf90/mpich2-1.0.2p1',
  '--with-shared=0',
  '--with-scalar-type=complex',
  '--download-superlu_dist=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
