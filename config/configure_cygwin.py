#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    # Using mpiuni
    #'--with-mpi=0',
    # Using MPICH for Windows 2000/NT available from http://www.mcs.anl.gov/mpi/mpich
    '--with-mpi-dir=/software/MPI/mpich-nt.1.2.5/SDK.gcc',
    '--with-mpi-compilers=0',
    # Using reference BLAS/LAPACK available from http://www.netlib.org/lapack
    '--with-blas-lapack-dir=/software/BLAS/LAPACK/LIB/win32_gnu_local_O',
    '-PETSC_ARCH='+configure.getarch()
    ]

    configure.petsc_configure(configure_options)
