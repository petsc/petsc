#!/usr/bin/python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')
if not os.path.isdir(petsc_hash_pkgs): os.mkdir(petsc_hash_pkgs)

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-cuda=1',
    '--with-precision=single',
    '--download-openblas', # default ATLAS blas on Ubuntu 14.04 breaks runex76 in src/mat/examples/tests
    '--download-openblas-make-options=TARGET=GENERIC',
    '--with-clanguage=c',
    '--with-single-library=0',
    '--with-visibility=1',
    # Note: If using nvcc with a host compiler other than the CUDA SDK default for your platform (GCC on Linux, clang
    # on Mac OS X, MSVC on Windows), you must set -ccbin appropriately in CUDAFLAGS, as in the example for PGI below:
    # 'CUDAFLAGS=-ccbin pgc++',
  ]
  import platform
  if platform.node() == 'p1':
    configure_options.append('--with-make-test-np=4')

  configure.petsc_configure(configure_options)
