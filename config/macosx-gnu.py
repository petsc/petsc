#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
        '--with-mpi-dir=/sandbox/petsc/soft/mpich-1.2.5.2',
        '--with-cxx=g++' # mpiCC does not work
        ]

    configure.petsc_configure(configure_options)


