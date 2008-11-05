#!/usr/bin/env python

configure_options = [
  # with gnu compilers + mpich autodetect
  '--with-blas-lib=/cygdrive/c/software/fblaslapack/win32_gnu/libfblas.a',
  '--with-lapack-lib=/cygdrive/c/software/fblaslapack/win32_gnu/libflapack.a',
  'DATAFILESPATH=/home/balay/datafiles',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
