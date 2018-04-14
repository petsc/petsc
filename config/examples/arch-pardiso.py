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
    '--with-blaslapack-dir='+os.environ['MKL_HOME'],
    '--with-mkl_pardiso-dir='+os.environ['MKL_HOME'],
    '--with-mkl_sparse_optimize=0',
    # using mpich-3.2 as default mpich-3.1.3 does not build with ifort-16
    '--download-mpich=http://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpich-3.2.tar.gz',
  ]
  configure.petsc_configure(configure_options)
