#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=nvcc',
    '--with-cxx=nvcc',
    '--with-mpi=0',
    '--with-cuda',
    'PETSC_ARCH=arch-cuda-double',
    '--with-precision=double',
    '--with-fc=0',
    '--with-clanguage=c++',
    '--CXX_CXXFLAGS=-x cu -arch sm_13'
  ]
  configure.petsc_configure(configure_options)
