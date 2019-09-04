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
    '--download-viennacl',
    '--with-opencl-include=/usr/local/cuda/include',
    '--with-opencl-lib=-L/usr/local/cuda/lib64 -lOpenCL',
    '--download-codipack=1',
    '--download-adblaslapack=1',
  ]
  configure.petsc_configure(configure_options)
