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
    '--with-blas-lapack-dir='+os.environ['MKL_HOME'],
    '--with-mkl_pardiso-dir='+os.environ['MKL_HOME'],
    '--download-mpich=1',
  ]
  configure.petsc_configure(configure_options)
