#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    # path set to $PETSC_DIR/bin/win32fe
    '--with-cc=win32fe bcc32 -w-8019 -w-8060 -w-8057 -w-8004 -w-8066',
    '--with-cxx=win32fe bcc32 -P -RT -w-8019 -w-8060 -w-8057 -w-8004 -w-8066',
    '--with-cpp=win32fe bcc32 --use cpp32',
    '--with-fc=0',
    '--with-ar=win32fe tlib -C -P512',
    '-AR_FLAGS=-u',
    '--with-ranlib=true',
    '--with-blas-lapack-dir=/cygdrive/c/software/f2cblaslapack/win32_borland',
    '--with-mpi=0'
    ]

    configure.petsc_configure(configure_options)
