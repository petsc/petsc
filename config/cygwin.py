#!/usr/bin/env python

configure_options = [
  #with gcc + mpich autodetect
  '--with-blas-lib=/cygdrive/c/software/fblaslapack/win32_gnu/libfblas.a',
  '--with-lapack-lib=/cygdrive/c/software/fblaslapack/win32_gnu/libflapack.a'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
