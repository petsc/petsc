#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    # build on harley
    configure_options = [
    '--with-cc=cc -64',
    '--with-fc=f90 -64',
    '--with-cxx=CC -64',
    '-ignoreWarnings',
    '-LDFLAGS=-Wl,-woff,84,-woff,85,-woff,113',
    '--with-f90-header=f90impl/f90_IRIX.h',
    '--with-f90-source=src/sys/src/f90/f90_IRIX.c',
    '--with-mpirun=mpirun',
    '-PETSC_ARCH='+configure.getarch()
    ]

    configure.petsc_configure(configure_options)
