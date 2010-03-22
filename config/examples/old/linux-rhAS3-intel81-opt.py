#!/usr/bin/env python

# Build PETSc on jazz cluster, with intel compilers

configure_options = [
  '--with-shared=1',
  '--with-is-color-value-type=short',
  '--with-mpi-dir=/soft/apps/packages/mpich-gm-1.2.6..13b-intel-8.1-2',
  '--with-blas-lapack-dir=/soft/com/packages/mkl_7.2/mkl72/lib/32',
  
  '-COPTFLAGS=-O3 -march=pentium4 -mcpu=pentium4',
  '-FOPTFLAGS=-O3 -march=pentium4 -mcpu=pentium4',
  '-CXXOPTFLAGS=-O3 -march=pentium4 -mcpu=pentium4',
  '--with-debugging=0',

  '--download-hypre=1',
  '--download-spooles=1',
  '--download-superlu=1',
  '--download-parmetis=1',
  '--download-superlu_dist=1',
  '--download-blacs=1',
  '--download-scalapack=1',
  '--download-mumps=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
