#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    # build on harley
    configure_options = [
    '--with-cc=cc -n32',
    '--with-fc=f90 -n32',
    '--with-cxx=CC -n32',
    '-ignoreWarnings',
    '-LDFLAGS=-Wl,-woff,84,-woff,85,-woff,113',
    '--with-f90-header=f90impl/f90_IRIX.h',
    '--with-f90-source=src/sys/src/f90/f90_IRIX.c',
    '--with-mpi-include=/home/petsc/software/mpich-1.2.0/IRIX/include',
    '--with-mpi-lib=/home/petsc/software/mpich-1.2.0/IRIX/lib/libmpich.a',
    '--with-mpirun=mpirun',
    '--with-lapack=/home/petsc/software/blaslapack/IRIX/libflapack.a',
    '-PETSC_ARCH='+configure.getarch()
    ]

    configure.petsc_configure(configure_options)
