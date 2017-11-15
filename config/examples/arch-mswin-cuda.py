#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=win32fe cl',
    '--with-cxx=win32fe cl',
    '--with-fc=0',
    '--with-mpi=0',
    '--download-f2cblaslapack=1',

    '--with-cudac=win32fe nvcc',
    '--with-cuda=1',
    '--with-cusp=1',
    '--with-thrust=1',
    '--with-cuda-arch=sm_12',
    '--with-precision=single',
  ]
  configure.petsc_configure(configure_options)

