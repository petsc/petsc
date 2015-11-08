#!/usr/bin/python
#
# This test build is with Cuda 5.5, with default thrust, and cusplibrary-0.4.0 separately installed.
#
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cuda=1',
    '--with-cusp=1',
    '-with-cusp-dir=/home/balay/soft/cusplibrary-0.4.0',
    '--with-thrust=1',
    '--with-precision=double',
    '--with-clanguage=c',
    '--with-cuda-arch=sm_13'

  ]
  configure.petsc_configure(configure_options)
