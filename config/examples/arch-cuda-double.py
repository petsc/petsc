#!/usr/bin/python
#
# This test build is with Cuda 4.1, with default thrust, and cusp-v0.3.1 separately installed.
# Also enable txpetscgpu.
#
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=gcc',
    '--with-cxx=g++',
    '--download-mpich=1',
    '--with-cuda=1',
    '--with-cusp=1',
    '-with-cusp-dir=/usr/local/cusp-v0.3.1',
    '--with-thrust=1',
    '--download-txpetscgpu=1',
    'PETSC_ARCH=arch-cuda-double',
    '--with-precision=double',
    '--with-fc=0',
    '--with-clanguage=c',
    '--with-cuda-arch=sm_13'

  ]
  configure.petsc_configure(configure_options)
