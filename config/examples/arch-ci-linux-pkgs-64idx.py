#!/usr/bin/env python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')
if not os.path.isdir(petsc_hash_pkgs): os.mkdir(petsc_hash_pkgs)

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  '--with-64-bit-indices=1',
  '--download-openmpi=1', #download-mpich works - but system mpich gives wierd errors with superlu_dist+parmeits [with shared/64-bit-indices]?
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-pastix=1',
  '--download-ptscotch=1',
  '--download-hypre=1',
  '--download-hypre-configure-arguments=--enable-bigint=no --enable-mixedint=yes', # HYPRE with mixed integers
  '--download-superlu_dist=1',
  '--donwload-suitesparse=1',
  '--download-cmake',  # superlu_dist requires a newer cmake
  '--download-p4est=1',
  '--with-zlib=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
