#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    '--with-mpi-dir==/sandbox/petsc/soft/mpich-1.2.5.2-ibm',
    '--with-gnu-compilers=0',
    '--with-vendor-compilers=ibm',
    # c++ doesn't work yet
    '--with-cxx=0'
    ]

    configure.petsc_configure(configure_options)


