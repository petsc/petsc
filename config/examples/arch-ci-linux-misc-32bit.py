#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-cc=gcc -m32',
  '--with-cxx=g++ -m32',
  '--with-fc=gfortran -m32',

  '--with-64-bit-indices=1',
  '--with-log=0',
  '--with-info=0',
  '--with-ctable=0',
  '--with-is-color-value-type=short',
  '--with-single-library=0',
  '--with-strict-petscerrorcode',

  '--with-c2html=0',
  '--with-x=0',
  '--download-mpich',
  '--download-fblaslapack',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
