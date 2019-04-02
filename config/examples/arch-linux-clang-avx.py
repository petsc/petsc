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
    '--with-cxx=clang++',
    '--with-fc=0',
    'CFLAGS=-mavx',
    'COPTFLAGS=-g -O',
    #'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-cxx-dialect=C++11',
    '--download-codipack=1',
    '--download-adblaslapack=1',
  ]
  configure.petsc_configure(configure_options)
