#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    '-PETSC_ARCH=cygwin-intel',
    '-PETSC_DIR=/home/Kris/petsc/petsc-dev',
    # Using MPICH for Windows 2000/NT available from http://www.mcs.anl.gov/mpi/mpich
    '--with-mpi-dir=/software/MPI/mpich-nt.1.2.5/SDK',
    # Using Intel's MKL available from http://www.intel.com
    '--with-blas-lapack=/software/BLAS/MKL/ia32/lib/mkl_c_dll.lib',
    '--with-cc=win32fe icl',
    '--CFLAGS=--nodetect -MT',
    '-CFLAGS_g=-Z7',
    '-CFLAGS_O=-O3 -QxW',
    '--with-fc=win32fe ifl',
    '--FFLAGS=--nodetect -MT -fpp',
    '-FFLAGS_g=-Z7',
    '-FFLAGS_O=-O3 -QxW',
    '--with-cxx=win32fe icl',
    '--CXXFLAGS=--nodetect -MT -GX -GR',
    '-CXXFLAGS_g=-TP -Z7',
    '-CXXFLAGS_O=-TP -O3 -QxW'
    ]

    configure.petsc_configure(configure_options)
