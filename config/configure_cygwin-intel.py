#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    import configure

    configure_options = [
    '-PETSC_ARCH=cygwin-intel',
    '-PETSC_DIR=/home/Kris/petsc/petsc-tmp',
    # Using MPICH for Windows 2000/NT available from http://www.mcs.anl.gov/mpi/mpich
    '--with-mpi-include=/software/MPI/mpich-nt.1.2.5/SDK/include',
    '--with-mpi-lib=[/software/MPI/mpich-nt.1.2.5/SDK/lib/mpich.lib,ws2_32.lib]',
    '--with-blas-lapack=/software/BLAS/MKL/ia32/lib/mkl_c_dll.lib',
    '--with-cc=win32fe icl --nodetect',
    '-CFLAGS_g=-MT -W3 -Z7',
    '-CFLAGS_O=-MT -W3 -O3 -QxW',
    '--with-fc=win32fe ifl --nodetect',
    '-FFLAGS_g=-MT -W0 -fpp -Z7',
    '-FFLAGS_O=-MT -W0 -fpp -O3 -QxW',
    '--with-cxx=win32fe icl --nodetect',
    '-CXXFLAGS_g=-MT -W3 -TP -GX -GR -Z7',
    '-CXXFLAGS_O=-MT -W3 -TP -GX -GR -O3 -QxW'
    ]

    configure.petsc_configure(configure_options)
