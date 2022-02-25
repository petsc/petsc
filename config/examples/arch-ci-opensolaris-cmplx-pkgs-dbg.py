#!/usr/bin/env python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  'CC=cc',
  'CXX=CC',
  'FC=f90',
  '--with-scalar-type=complex',
  'FFLAGS=-ftrap=%none',
  '--with-c2html=0',
  '--download-mpich',
  '--download-metis',
  '--download-parmetis',
  '--download-triangle',
  '--download-superlu',
  '--download-fblaslapack',
  '--download-scalapack',
  '--download-mumps',
  '--download-hdf5',
  # requires C++ compiler
  #'--download-suitesparse',
  '--download-chaco',
  # opensolaris throws warning
  # CC: Warning: Option -std=c++03 passed to ld, if ld is invoked, ignored otherwise
  # to stderr, so don't use flag at all
  '--with-cxx-dialect=0',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
