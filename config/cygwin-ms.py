#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    # Using MPICH for Windows 2000/NT available from http://www.mcs.anl.gov/mpi/mpich
    '--with-mpi-dir=/software/MPI/mpich-nt.1.2.5/SDK',
    '--with-mpi-compilers=0',
    '--with-blas-lapack-dir=/software/BLAS/MKL',
    # Using Microsoft C/C++ compiler
    '--with-cc=win32fe cl',
    '--with-cxx=win32fe cl',
    # Using Compaq FORTRAN Compiler
    '--with-fc=win32fe f90'
    ]

    configure.petsc_configure(configure_options)
