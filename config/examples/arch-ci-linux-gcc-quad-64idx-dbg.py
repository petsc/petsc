#!/usr/bin/env python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-debugging=yes',
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  '--with-clanguage=c',
  '--with-shared-libraries=no',
  '--with-scalar-type=real',
  '--with-64-bit-indices=yes',
  '--with-precision=__float128',
  '--download-mpich',
  '--download-f2cblaslapack',
]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
