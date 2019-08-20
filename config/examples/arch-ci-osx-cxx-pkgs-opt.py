#!/usr/bin/env python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')
if not os.path.isdir(petsc_hash_pkgs): os.mkdir(petsc_hash_pkgs)

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-clanguage=cxx',
  '--with-cxx-dialect=C++11',
  '--with-debugging=0',
  'CXXFLAGS=-Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -Wno-deprecated',
  '--with-visibility=0', # CXXFLAGS disables this option
  '--download-mpich',
  '--download-metis',
  '--download-parmetis',
  '--download-ptscotch',
  '--download-triangle',
  '--download-superlu',
  '--download-superlu_dist',
  '--download-scalapack',
  '--download-mumps',
  '--download-parms',
  '--download-hdf5',
  '--download-sundials',
  '--download-hypre',
  '--download-suitesparse',
  '--download-chaco',
  '--download-spai',
  '--download-saws',
  '--download-revolve',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
