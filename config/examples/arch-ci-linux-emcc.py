#!/usr/bin/env python3

import os

configure_options = [
  '--with-cc=emcc',
  '--with-cxx=0',
  '--with-fc=0',
  '--with-ranlib=emranlib',
  '--with-ar=emar',
  '--with-shared-libraries=0',
  '--download-f2cblaslapack=1',
  '--with-mpi=0',
  '--with-batch',
  '--with-strict-petscerrorcode',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
