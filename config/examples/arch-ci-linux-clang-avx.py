#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--download-f2cblaslapack',
    '--download-blis',
    '--download-mpich',
    '--with-cc=clang',
    '--with-cxx=clang++',
    '--with-fc=0',
    'CFLAGS=-mavx',
    'COPTFLAGS=-g -O',
    #'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--download-codipack=1',
    '--download-adblaslapack=1',
    '--with-mpi-f90module-visibility=0',
  ]
  configure.petsc_configure(configure_options)
