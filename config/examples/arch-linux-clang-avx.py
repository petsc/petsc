#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-f2cblaslapack',
    '--download-mpich',
    '--with-cc=clang',
    '--with-cxx=0',
    '--with-fc=0',
    'CFLAGS=-mavx',
    'PETSC_ARCH=arch-linux-clang-avx',
  ]
  configure.petsc_configure(configure_options)
