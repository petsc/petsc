#!/usr/bin/python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')
if not os.path.isdir(petsc_hash_pkgs): os.mkdir(petsc_hash_pkgs)

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--download-f2cblaslapack',
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
  ]
  configure.petsc_configure(configure_options)
