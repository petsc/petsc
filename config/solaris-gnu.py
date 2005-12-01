#!/bin/env python

# Shared library target doesn't currently work with solaris & gnu
configure_options = [
  '--with-vendor-compilers=0',
  '--with-blas-lib=/home/petsc/soft/solaris-9-gnu/fblaslapack/libfblas.a',
  '--with-lapack-lib=/home/petsc/soft/solaris-9-gnu/fblaslapack/libflapack.a',    
  '--download-mpich=1',
  '--with-shared=0'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
