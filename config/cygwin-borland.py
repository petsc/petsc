#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    # path set to $PETSC_DIR/bin/win32fe
    '--with-vendor-compilers=borland',
    '--with-fc=0',
    '--with-ranlib=true',
    '--with-blas-lapack-dir=/cygdrive/c/software/f2cblaslapack/win32_borland',
    '--with-mpi=0'
    ]

    configure.petsc_configure(configure_options)
