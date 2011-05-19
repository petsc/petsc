#!/usr/bin/env python

configure_options = [
  '--download-mpich=1',
  #'--with-gcov=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
