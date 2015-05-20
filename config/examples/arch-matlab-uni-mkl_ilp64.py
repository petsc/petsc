#!/usr/bin/python

# This test is done on cg.mcs.anl.gov. It uses IPL64 MKL/BLAS packaged
# with MATLAB.

# Note: regular BLAS [with 32bit integers] conflict wtih
# MATLAB BLAS - hence requring -known-64-bit-blas-indices=1

# Note: MATLAB build requires petsc shared libraries

# export LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/4.6/libgfortran.so

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-blas-lapack-dir=/soft/com/packages/MATLAB/R2013a',
    '--with-matlab-engine=1',
    '--with-matlab=1',
    '--with-mpi=0',
    '--with-shared-libraries=1',
    '-known-64-bit-blas-indices=1',
  ]
  configure.petsc_configure(configure_options)
