#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    'CC=icc',
    'CXX=icpc',
    'FC=ifort',
    '--with-blas-lapack-dir=/soft/com/packages/intel/13/update5/mkl/',
    '--with-mkl_pardiso-dir=/soft/com/packages/intel/13/update5/mkl/',
    '--download-mpich=1',
  ]
  configure.petsc_configure(configure_options)
