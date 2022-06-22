#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-debugging=0',

  '--useThreads=0', # for some reason cmake hangs when invoked from configure on bsd?
  #'-download-fblaslapack=1',
  '--download-mpich=1',
  '--download-mpich-device=ch3:sock',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-triangle=1',
  #'--download-superlu=1',
  #'--download-superlu_dist=1', disabled as superlu_dist now requires gnumake - and this build tests freebsd-make
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-parms=1',
  # no with-cxx-dialect=C++11 support '--download-elemental=1',
  #'--download-hdf5',
  '--download-sundials2=1',
  '--download-hypre=1',
  #'--download-suitesparse=1', requires gnumake
  '--download-chaco=1',
  '--download-spai=1',
  '--download-concurrencykit=1',
  '--download-revolve=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
