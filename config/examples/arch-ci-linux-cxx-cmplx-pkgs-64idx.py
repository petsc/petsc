#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-cc=clang',
  '--with-fc=gfortran',
  '--with-cxx=clang++',

  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',

  '--with-clanguage=cxx',
  'CXXFLAGS=-Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -Wno-deprecated',
  '--with-scalar-type=complex',
  '--with-64-bit-indices=1',

  '--with-log=0',
  '--with-info=0',  

  '--download-hypre=1',
  '--download-mpich=1',
  '--download-cmake=1',
  '--download-make=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-pastix=1',
  '--download-hwloc',
  '--download-ptscotch=1',
  '--download-superlu_dist=1',
  '--download-elemental=1',
  '--download-p4est=1',
  '--with-zlib=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
