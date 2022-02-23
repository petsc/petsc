#!/usr/bin/env python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-cc=cc',
  '--with-cxx=CC',
  '--with-fc=f90',
  #'--with-debugger=/bin/true',
  '--with-debugging=0',
  'FFLAGS=-ftrap=%none',

  '--download-mpich=1',
  '--download-mpich-device=ch3:sock',

  '--with-c2html=0',

  #'-download-fblaslapack=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-triangle=1',
  '--download-superlu=1',
  #'--download-superlu_dist=1', now requires C++11
  '--download-fblaslapack=1', # -lsunperf is insufficient for scalapack
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-parms=1',
  #'--download-elemental=1', breaks with solaris compilers
  #'--download-hdf5',
  #'--download-sundials2=1', breaks when built via ssh - but not on terminal?
  # requires C++ compiler
  #'--download-hypre=1',
  #'--download-suitesparse=1',
  '--download-chaco=1',
  '--download-spai=1',
  '--with-mpi-f90module-visibility=0',
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
