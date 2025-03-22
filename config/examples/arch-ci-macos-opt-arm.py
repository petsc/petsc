#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  'FPPFLAGS=-fno-underscoring', # test PETSC_HAVE_FORTRAN_UNDERSCORE
  '--download-triangle', # for TRIANGLE_HDF5 test (with trilinos chaco)
  '--download-triangle-build-exec',
  '--download-mpich=1',
  '--download-mpich-device=ch3:sock',
  '--with-debugging=0',
  '--download-metis=1',
  '--download-suitesparse=1',
  '--with-shared-libraries=0',
  '--download-eigen',
  #'--with-coverage',
  '--with-strict-petscerrorcode=0',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
