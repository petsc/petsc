#!/usr/bin/env python3

configure_options = [
  '--download-xsdk',
  '--download-fblaslapack=1',
  '--download-mpich=1',
  '--download-cmake=1',
  '--with-clanguage=C++',
  '--with-debugging=1',
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  '--with-shared-libraries=0',

  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
