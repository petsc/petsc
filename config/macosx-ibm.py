#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    '--with-mpi=0',
    '--with-gnu-compilers=0',
    '--with-vendor-compilers=ibm'
    ]

    configure.petsc_configure(configure_options)


