#!/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    '--with-f90-header=f90impl/f90_solaris.h',
    '--with-f90-source=src/sys/src/f90/f90_solaris.c',
    '--with-mpi-dir=/home/petsc/soft/solaris-9-lam/lam-6.5.8'
    ]

    configure.petsc_configure(configure_options)
