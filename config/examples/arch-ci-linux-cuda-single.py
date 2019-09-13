#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-cuda=1',
    '--with-precision=single',
    '--download-openblas', # default ATLAS blas on Ubuntu 14.04 breaks runex76 in src/mat/examples/tests
    '--download-openblas-make-options=CORE=GENERIC TARGET=GENERIC',
    '--with-clanguage=c',
    # Note: If using nvcc with a host compiler other than the CUDA SDK default for your platform (GCC on Linux, clang
    # on Mac OS X, MSVC on Windows), you must set -ccbin appropriately in CUDAFLAGS, as in the example for PGI below:
    # 'CUDAFLAGS=-ccbin pgc++',
  ]
  configure.petsc_configure(configure_options)
