#!/usr/bin/python
#
# This test build is with Cuda 5.0, with default thrust, and cusp-v0.3.1 separately installed.
# [using default mpich from ubuntu 12.04]
#
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cuda=1',
    '--with-cusp=1',
    '-with-cusp-dir=/home/balay/soft/cusp-v0.3.1',
    '--with-thrust=1',
    '--with-precision=double',
    '--with-clanguage=c',
    '--with-cuda-arch=sm_13'

  ]
  configure.petsc_configure(configure_options)
