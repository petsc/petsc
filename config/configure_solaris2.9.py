#!/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    # build on harley
    configure_options = [
    '--with-cc=cc',
    '--with-fc=f90',
    '--with-cxx=CC',
    '--with-f90-header=f90impl/f90_solaris.h',
    '--with-f90-source=src/sys/src/f90/f90_solaris.c',
    '--with-mpi-include=/home/petsc/soft/solaris-9/mpich-1.2.5/include',
    '--with-mpi-lib=[/home/petsc/soft/solaris-9/mpich-1.2.5/lib/libmpich.a,libsocket.a,libnsl.a,librt.a,libnsl.a,libaio.a]'
    ]

    configure.petsc_configure(configure_options)
