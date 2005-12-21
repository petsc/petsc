#!/usr/bin/env python

# ******* Currently not tested **********

configure_options = [
  '--with-gnu-compilers=0',
  '--with-mpi-include=/home/petsc/software/mpich-1.2.2.3/alpha/include',
  '--with-mpi-lib=/home/petsc/software/mpich-1.2.2.3/alpha/lib/libmpich.a',
  '--with-mpirun=mpirun',
  '--with-blas-lib=/home/petsc/software/fblaslapack/alpha/libfblas.a',
  '--with-lapack-lib=/home/petsc/software/fblaslapack/alpha/libflapack.a'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
