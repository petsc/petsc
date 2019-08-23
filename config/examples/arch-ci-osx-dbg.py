#!/usr/bin/env python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')
if not os.path.isdir(petsc_hash_pkgs): os.mkdir(petsc_hash_pkgs)

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-cc=gcc',
  '--with-fc=gfortran', # https://brew.sh/
  '--with-cxx=g++',
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  '--download-mpich=1',
  '--download-mpich-device=ch3:nemesis', #for some reason runex174_2_elemental takes very long with ch3:p4
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-elemental=1',
  '--download-metis',
  '--download-parmetis',
  '--download-ptscotch',
  '--download-scalapack',
  '--download-strumpack',
  '--download-fblaslapack', #vecLib has incomplete lapack - so unuseable by strumpack
  '--download-codipack=1',
  '--download-adblaslapack=1',
  '--download-libpng=1',
  '--download-libjpeg=1',
  '--with-zlib=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
