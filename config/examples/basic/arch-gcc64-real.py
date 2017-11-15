#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=gcc',
    '--with-clanguage=c',
    '--with-64-bit-indices=1',
    '--with-scalar-type=real',
    '--with-dynamic-loading=1',
    '--with-shared-libraries=1',
    '--download-f-blas-lapack=1',
    '--download-mpich=1',
    '--with-python=1',
    '--PETSC_ARCH=arch-gcc64-real'
    ]
  configure.petsc_configure(configure_options)
