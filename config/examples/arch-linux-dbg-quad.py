#!/usr/bin/env python

configure_options = [
  '--with-debugging=1',
  '--download-f2cblaslapack=1',
  '--with-precision=__float128',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
