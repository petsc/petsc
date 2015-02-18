#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=clang',
    '--with-fc=gfortran',
    '--with-cxx=clang++',
    '--with-clanguage=c++',
    '--with-shared-libraries=yes',
    '--with-debugging=yes',
    '--with-scalar-type=complex',
    '--with-64-bit-indices=no',
    '--with-precision=single',
    ]
  configure.petsc_configure(configure_options)
