#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-debugging=0',

  '--useThreads=0', # for some reason cmake hangs when invoked from configure on bsd?
  '--download-netlib-lapack=1',
  '--with-netlib-lapack-c-bindings=1',
  '--download-pastix=1',
  '--download-hwloc=1',
  '--with-mpi-dir=/home/svcpetsc/soft/mpich-4.2.2',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-triangle=1',
  '--download-triangle-build-exec=1',
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
  '--with-strict-petscerrorcode',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
