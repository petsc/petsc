#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    '-PETSC_ARCH=cygwin-ms',
    '-PETSC_DIR=/home/Kris/petsc/petsc-2',
    # Using MPICH for Windows 2000/NT available from http://www.mcs.anl.gov/mpi/mpich
    '--with-mpi-include=/software/MPI/mpich-nt.1.2.5/SDK/include',
    '--with-mpi-lib=[/software/MPI/mpich-nt.1.2.5/SDK/lib/mpich.lib,ws2_32.lib]',
    '--with-blas-lapack=/software/BLAS/MKL/ia32/lib/mkl_c_dll.lib',
    '--with-cc=win32fe cl',
    '--CFLAGS=-MT -W3',
    '-CFLAGS_g=-Z7',
    '-CFLAGS_O=-O3',
    '--with-fc=win32fe f90',
    '-FFLAGS_g=-threads -debug:full',
    '-FFLAGS_O=-threads -optimize:5 -fast',
    '--with-cxx=win32fe cl',
    '--CXXFLAGS=-MT -W3 -GX -GR',
    '-CXXFLAGS_g=-TP -Z7',
    '-CXXFLAGS_O=-TP -O3 -QxW'
    ]

    configure.petsc_configure(configure_options)
