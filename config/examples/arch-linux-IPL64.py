#!/usr/bin/env python

configure_options = [
  '--with-64-bit-indices=1',
  '--download-openblas',
  '--download-openblas-64-bit-blas-indices=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
