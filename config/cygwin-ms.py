#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    # Autodetect MPICH & Intel MKL
    # path set to $PETSC_DIR/bin/win32fe
    '--with-cc=win32fe cl',
    '--with-cxx=win32fe cl',
    '--with-fc=win32fe f90'
    ]

    configure.petsc_configure(configure_options)
