#!/usr/bin/env python

configure_options = [
  '--with-cc=gcc',
  '--with-fc=f90',
  '--with-cxx=g++',
  '--with-language=c++',
  '--with-blas-lib=/home/petsc/software/LAPACK/libblas_linux_absoft.a',
  '--with-lapack-lib=/home/petsc/software/LAPACK/liblapack_linux_absoft.a',
  '--with-mpi-dir=/home/petsc/software/mpich-1.2.0/linux_absoft',
  '--with-matlab=0'
  ]

if __name__ == '__main__':
    import configure
    configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
