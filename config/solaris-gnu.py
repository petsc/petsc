#!/bin/env python
import os
import sys

# Shared library target doesn't currently work with solaris & gnu
if __name__ == '__main__':
    import configure

    configure_options = [
    '--with-vendor-compilers=0',
    '--with-mpi-compilers=0',
    '--with-blas-lib=/home/petsc/soft/solaris-9-gnu/fblaslapack/libfblas.a',
    '--with-lapack-lib=/home/petsc/soft/solaris-9-gnu/fblaslapack/libflapack.a',    
    '--with-mpi-dir=/home/petsc/soft/solaris-9-gnu/mpich-1.2.5.2',
    '--with-shared=0'
    ]

    configure.petsc_configure(configure_options)
