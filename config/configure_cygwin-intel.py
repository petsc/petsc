#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    '-PETSC_ARCH=cygwin-intel',
    '-PETSC_DIR=/home/Kris/petsc/petsc-2',
    # Using MPICH for Windows 2000/NT available from http://www.mcs.anl.gov/mpi/mpich
    '--with-mpi-include=/software/MPI/mpich-nt.1.2.5/SDK/include',
    '--with-mpi-lib=[/software/MPI/mpich-nt.1.2.5/SDK/lib/mpich.lib,ws2_32.lib]',
    '--with-blas-lapack=/software/BLAS/MKL/ia32/lib/mkl_c_dll.lib',
    '--with-cc=win32fe icl',
    '--CFLAGS=--nodetect -MT -W3',
    '-CFLAGS_g=-Z7',
    '-CFLAGS_O=-O3 -QxW',
    '--with-fc=win32fe ifl',
    '--FFLAGS=--nodetect -MT -fpp -W0',
    '-FFLAGS_g=-Z7',
    '-FFLAGS_O=-O3 -QxW',
    '--with-cxx=win32fe icl',
    '--CXXFLAGS=--nodetect -MT -W3 -GX -GR',
    '-CXXFLAGS_g=-TP -Z7',
    '-CXXFLAGS_O=-TP -O3 -QxW'
    ]

    configure.petsc_configure(configure_options)
