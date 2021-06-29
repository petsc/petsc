#!/usr/tce/packages/python/python-3.7.2/bin/python3

# Tested 2021-06-13 with
# $ module list
#
# Currently Loaded Modules:
#   1) StdEnv (S)   2) clang/ibm-11.0.1   3) spectrum-mpi/rolling-release   4) cuda/11.2.0

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-blaslapack-lib=/usr/tcetmp/packages/essl/essl-6.3.0/lib64/liblapackforessl.so /usr/tcetmp/packages/essl/essl-6.3.0/lib64/libessl.so',
    '--with-cuda=1',
    '--with-debugging=0',
    '--with-fc=0',
    '--with-mpi-dir=/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-ibm-11.0.1',
    'COPTFLAGS=-O3 -mcpu=native -ffp-contract=fast',
    'CUDAFLAGS=--gpu-architecture=sm_70 -ccbin clang++',
    'CXXOPTFLAGS=-O3 -mcpu=native -ffp-contract=fast',
    'PETSC_ARCH=lassen-clang-essl-opt',
  ]
  configure.petsc_configure(configure_options)
