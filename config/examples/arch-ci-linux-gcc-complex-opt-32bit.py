#!/usr/bin/env python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  'CFLAGS=-m32 -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas',
  'CXXFLAGS=-m32 -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas',
  'FFLAGS=-m32 -fPIC -Wall -ffree-line-length-0 -Wno-unused-dummy-argument',
  '--with-clanguage=c',
  '--with-shared-libraries=yes',
  '--with-debugging=no',
  '--with-scalar-type=complex',
  '--with-64-bit-indices=no',
  '--with-precision=double',
  '--download-mpich',
  '--download-fblaslapack',
]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
