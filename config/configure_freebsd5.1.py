#!/usr/bin/env python

if __name__ == '__main__':
  import configure
  
  configure_options = [
    '--with-mpi-dir=/software/mpich-1.2.5.2',
    '--with-blas=/usr/local/lib/libblas.a',
    '--with-lapack=/usr/local/lib/liblapack.a',
    '--with-PETSC_ARCH=freebsd5.1',
    '--with-PETSC_DIR=/home/petsc/petsc-test'
    ]

  configure.petsc_configure(configure_options)
