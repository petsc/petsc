#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-make-test-np=15',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-log=0',
    '--with-info=0',
    '--with-cuda=1',
    '--with-precision=single',
    '--with-clanguage=cxx',
    '--with-single-library=0',
    '--with-visibility=1',
    '--download-hpddm',
    # 28/02/2023, nvc/nvc++ would produce strange segmentation violation errors in C++20
    # mode that could not be reproduced with any other compiler. Normally the first
    # response would be 'there must be a bug in the code' -- and that may still be true --
    # but no less than 3 other CI jobs reproduce the same package/env without these seg
    # faults. So limiting to C++17 because maybe it *is* the compilers fault this time.
    '--with-cxx-dialect=17',
    # Note: If using nvcc with a host compiler other than the CUDA SDK default for your platform (GCC on Linux, clang
    # on Mac OS X, MSVC on Windows), you must set -ccbin appropriately in CUDAFLAGS, as in the example for PGI below:
    # 'CUDAFLAGS=-ccbin pgc++',
    '--with-strict-petscerrorcode',
  ]

  configure.petsc_configure(configure_options)
