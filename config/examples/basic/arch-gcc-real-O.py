#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-f-blas-lapack=1',
    '--download-mpich=1',
    '--with-cc=gcc',
    '--with-clanguage=c',
    '--with-dynamic-loading=1',
    '--with-shared-libraries=1',
    '--with-python=1',
    '--with-debugging=0',
    '--PETSC_ARCH=arch-gcc-real-O'
    ]
  configure.petsc_configure(configure_options)
