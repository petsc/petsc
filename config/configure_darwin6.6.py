#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    '--with-mpi-dir=/Users/petsc/software/mpich-1.2.5',
    '-PETSC_ARCH=darwin6.6',
    '-PETSC_DIR=/Users/petsc/petsc-test'
    ]

    configure.petsc_configure(configure_options)
