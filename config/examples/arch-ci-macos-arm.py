#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-mpi-dir=/Users/petsc/soft/mpich-4.3.0-p2-ofi',
  '--with-64-bit-indices=1',
  '--with-clanguage=cxx',
  'CXXFLAGS=-Wall -Wwrite-strings -Wshorten-64-to-32 -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fno-stack-check -Wno-deprecated -fvisibility=hidden',
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  '--with-petsc4py=1',
  '--download-mpi4py=1',
  '--download-make=1',
  '--download-slepc=1',
  '--download-f2cblaslapack=1',
  '--with-f2cblaslapack-fp16-bindings=1',
  #'--with-coverage',
  '--with-strict-petscerrorcode',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
