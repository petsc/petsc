#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-cc=gcc',
  '--with-fc=gfortran',
  '--with-cxx=g++',

  'COPTFLAGS=-g -O0',
  'FOPTFLAGS=-g -O0',
  'CXXOPTFLAGS=-g -O0',

  '--with-clanguage=cxx',
  '--with-scalar-type=complex',
  '--with-64-bit-indices=1',

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
  '--download-ptscotch',
  '--download-scalapack',
  '--download-strumpack',
  '--with-zlib=1',
  '--with-coverage=1',
  '--with-strict-petscerrorcode',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
