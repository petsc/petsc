#!/usr/bin/env python

configure_options = [
  '--with-cc=gcc',
  '--with-fc=g95',
  '--with-python',
  '--with-shared=1',
  '--with-dynamic=1',
  '--download-mpich',
  '-download-mpich-pm=gforker'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
