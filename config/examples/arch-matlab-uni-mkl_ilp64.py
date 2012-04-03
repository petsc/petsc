#!/usr/bin/python

# This test is done on cg.mcs.anl.gov. It uses IPL64 MKL/BLAS packaged
# with MATLAB.

# Note: regular BLAS [with 32bit integers] conflict wtih
# MATLAB BLAS - hence requring -known-64-bit-blas-indices=1

# Note: MATLAB build requires petsc shared libraries

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-blas-lapack-dir=/soft/com/packages/MATLAB/R2012a',
    '--with-matlab-engine=1',
    '--with-matlab=1',
    '--with-mpi=0',
    '--with-shared-libraries=1',
    '-known-64-bit-blas-indices=1',
  ]
  configure.petsc_configure(configure_options)
