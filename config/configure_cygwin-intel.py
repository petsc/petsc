#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    # Using MPICH for Windows 2000/NT available from http://www.mcs.anl.gov/mpi/mpich
    '--with-mpi-dir=/software/MPI/mpich-nt.1.2.5/SDK',
    '--with-mpi-compilers=0',
    # Using Intel's MKL available from http://www.intel.com
    '--with-blas-lapack=/software/BLAS/MKL/ia32/lib/mkl_c_dll.lib',
    '--with-vendor-compilers=intel'
    ]

    configure.petsc_configure(configure_options)
