#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    '--with-mpi=0',
    # Mac does not come with fortran compiler and fink g77 is out of sync with mac compilers
    '--with-fc=0'
    ]

    configure.petsc_configure(configure_options)


