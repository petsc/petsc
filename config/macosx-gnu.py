#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
        # mpi built with gcc/g77 (fink)
        '--with-cxx=0',
        '--with-mpi-dir=/home/petsc/soft/darwin-7/mpich-1.2.5.2'
        ]

    configure.petsc_configure(configure_options)


