#!/usr/bin/env python

configure_options = [
  '--with-cc=cc -n32',
  '--with-fc=f90 -n32',
  '--with-cxx=CC -n32',
  '-ignoreWarnings',
  '-LDFLAGS=-Wl,-woff,84,-woff,85,-woff,113',
  '--with-f90-header=f90_IRIX.h',
  '--with-f90-source=f90_IRIX.c',
  '--with-mpi-include=/home/petsc/software/mpich-1.2.0/IRIX/include',
  '--with-mpi-lib=/home/petsc/software/mpich-1.2.0/IRIX/lib/libmpich.a',
  '--with-mpirun=mpirun',
  '--with-lapack-lib=/home/petsc/software/blaslapack/IRIX/libflapack.a'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
