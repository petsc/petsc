#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    '--with-mpi',
    '--with-mpi-dir=/Users/petsc/software/mpich-1.2.5',
    '-PETSC_ARCH=darwin6.6',
    '-PETSC_DIR=/Users/petsc/petsc-test',
    '--with-blas-lapack=/System/Library/Frameworks/vecLib.framework/vecLib',
    '--with-ranlib=ranlib -s -c'
    ]

    configure.petsc_configure(configure_options)
