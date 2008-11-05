#!/bin/env python

configure_options = [
  '--with-cc=cc',
  '--with-fc=f90',
  '--with-f90-interface=solaris',
  '--download-mpich=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
