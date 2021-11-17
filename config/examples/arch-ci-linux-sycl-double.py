#!/usr/bin/python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-mpi-dir=/opt/intel/inteloneapi/mpi/latest',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-cuda=0',
    '--with-sycl=1',
    '--with-syclc=dpcpp',
    '--with-sycl-dir=/opt/intel/inteloneapi/compiler/latest/linux',
    '--with-precision=double',
    '--with-clanguage=c',
  ]

  configure.petsc_configure(configure_options)
