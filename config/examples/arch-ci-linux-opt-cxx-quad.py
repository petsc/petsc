#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-debugging=0',
  'CC=gcc',
  'CXX=g++',
  'FC=gfortran',
  '--with-mpi-include=/usr/include/x86_64-linux-gnu/mpich',
  '--with-mpi-lib=-L/usr/lib/x86_64-linux-gnu -lmpichfort -lmpich',
  '--download-f2cblaslapack=1',
  '--download-slepc=1',
  '--download-hpddm=1',
  '--with-precision=__float128',
  '--with-clanguage=cxx',
  '--with-mpi-f90module-visibility=0',
  '--with-strict-petscerrorcode',
  '--with-coverage',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
