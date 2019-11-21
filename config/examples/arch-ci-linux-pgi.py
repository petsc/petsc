#!/usr/bin/env python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  # the machine this runs on does not support PGI std-C++11 functionality
  '--with-cxx-dialect=cxx11',
  'CC=pgcc',
  'CXX=pgc++',
  'FC=pgf90',
  #'COPTFLAGS=-g -O', #-O gives compile errors with fblaslapack? so disabling for now
  #'FOPTFLAGS=-g -O',
  #'CXXOPTFLAGS=-g -O',
  '--with-hwloc=0', # ubuntu -lhwloc requires -lnuma - which conflicts with -lnuma from pgf90
  '--download-mpich=1',
  '--download-fblaslapack=1',
  '--download-codipack=1',
  '--download-adblaslapack=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
