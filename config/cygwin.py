#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    #with gcc + mpich autodetect
    '--with-mpi-compilers=0',
    '--with-blas-lib=/cygdrive/c/software/fblaslapack/win32_gnu/libfblas.a',
    '--with-lapack-lib=/cygdrive/c/software/fblaslapack/win32_gnu/libflapack.a'
    ]

    configure.petsc_configure(configure_options)
