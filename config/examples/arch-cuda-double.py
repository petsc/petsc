#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=gcc',
    '--with-cxx=g++',
    '--with-mpi=0',
    '--with-cuda=1',
    '--with-cusp=1',
    '--with-thrust=1',
    'PETSC_ARCH=arch-cuda-double',
    '--with-precision=double',
    '--with-fc=0',
    '--with-clanguage=c',
    '--with-cuda-arch=sm_13'

  ]
  configure.petsc_configure(configure_options)
