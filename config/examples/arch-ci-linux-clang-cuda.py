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
    '--with-debugging=0',
    '--with-cuda=1',
    '--with-cudac=clang++',
    '--with-cuda-dialect=17',
    '--with-cc=clang',
    '--with-cxx=clang++',
    '--download-openmpi',
    #'--with-coverage',
  ]

  configure.petsc_configure(configure_options)
