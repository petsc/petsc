#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    # Using MPICH for Windows 2000/NT available from http://www.mcs.anl.gov/mpi/mpich
    '--with-mpi-dir=/software/MPI/mpich-nt.1.2.5/SDK',
    '--with-mpi-compilers=0',
    '--with-blas-lapack=/software/BLAS/MKL/ia32/lib/mkl_c_dll.lib',
    # Using Microsoft C/C++ compiler
    '--with-cc=win32fe cl',
    '--with-cxx=win32fe cl',
    # Using Compaq FORTRAN Compiler
    '--with-fc=win32fe f90',
    '-PETSC_ARCH='+configure.getarch()
    ]

    configure.petsc_configure(configure_options)
