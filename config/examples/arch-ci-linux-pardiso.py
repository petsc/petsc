#!/usr/bin/python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    'CC=icc',
    'CXX=icpc',
    'FC=ifort',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-blaslapack-dir='+os.environ['MKL_HOME'],
    '--with-mkl_pardiso-dir='+os.environ['MKL_HOME'],
    '--with-mkl_sparse_optimize=0',
    '--download-mpich=1',
    '--download-triangle=1',
    '--download-ctetgen=1',
    '--download-tetgen=1',
    '--download-p4est=1',
    '--download-zlib=1',
    '--download-codipack=1',
    '--download-adblaslapack=1',
  ]
  configure.petsc_configure(configure_options)
