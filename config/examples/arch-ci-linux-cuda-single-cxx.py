#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-make-test-np=15',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-log=0',
    '--with-info=0',
    '--with-cuda=1',
    '--with-precision=single',
    '--with-clanguage=cxx',
    '--with-single-library=0',
    '--with-visibility=1',
    '--download-hpddm',
    '--with-strict-petscerrorcode',
  ]

  configure.petsc_configure(configure_options)
