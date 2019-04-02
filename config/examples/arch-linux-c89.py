#!/usr/bin/env python

configure_options = [
  'CFLAGS=-std=c89 -pedantic -Wno-long-long -Wno-overlength-strings',
  #'COPTFLAGS=-g -O',
  #'FOPTFLAGS=-g -O',
  #'CXXOPTFLAGS=-g -O',
  '--with-shared-libraries=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
