#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    # build on harley
    configure_options = [
    '--with-mpi',
    '--with-mpi-include=/Users/petsc/software/mpich-1.2.4/macx/include',
    '--with-mpi-lib=[/Users/petsc/software/mpich-1.2.4/macx/lib/libmpich.a,/Users/petsc/software/mpich-1.2.4/macx/lib/libpmpich.a]',
    '--with-mpirun=mpirun',
    '-PETSC_ARCH=darwin6.4',
    '-PETSC_DIR=/Users/petsc/petsc-test',
    '--with-blas=/Users/petsc/software/fblaslapack/macx/libfblas.a',
    '--with-lapack=/Users/petsc/software/fblaslapack/macx/libflapack.a',
    '--with-ranlib=ranlib -s -c'
    ]

    configure.petsc_configure(configure_options)
