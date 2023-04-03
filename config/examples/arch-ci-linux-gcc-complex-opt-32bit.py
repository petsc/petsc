#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-cc=gcc -m32',
  '--with-cxx=g++ -m32',
  '--with-fc=gfortran -m32',
  '--with-clanguage=c',
  '--with-shared-libraries=yes',
  '--with-debugging=no',
  '--with-scalar-type=complex',
  '--with-64-bit-indices=no',
  '--with-precision=double',
  '--download-mpich',
  '--download-fblaslapack',
  '--with-strict-petscerrorcode',
  '--with-coverage',
]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
