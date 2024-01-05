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
    '--with-cudac=0',
    '--with-hipc=0',
    '--download-mpich',
    '--with-cc=clang',
    '--with-cxx=clang++',
    '--with-fc=0',
    '--with-debugging=1',
    'CFLAGS=-fsanitize=address,undefined',
    'CXXFLAGS=-fsanitize=address,undefined',
    'LDFLAGS=-lubsan',
    '--with-strict-petscerrorcode',
  ]
  configure.petsc_configure(configure_options)

