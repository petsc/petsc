#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  'CFLAGS=-std=c99 -pedantic -Wno-long-long -Wno-overlength-strings',
  '--with-precision=single',
  '--with-debugging=0',
  '--download-mpich',
  '--download-mpich-device=ch3:sock',
  '--download-superlu_dist',
  '--download-metis',
  '--download-parmetis',
  '--download-cmake'  # needed by metis/parmetis
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
