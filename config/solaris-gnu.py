#!/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    '--with-vendor-compilers=0',
    '--with-mpi-compilers=0',
    '--with-blas=/home/petsc/soft/solaris-9-gnu/fblaslapack/libfblas.a',
    '--with-lapack=/home/petsc/soft/solaris-9-gnu/fblaslapack/libflapack.a',    
    '--with-mpi-dir=/home/petsc/soft/solaris-9-gnu/mpich-1.2.5'
    ]

    configure.petsc_configure(configure_options)
