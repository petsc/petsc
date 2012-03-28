#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cuda=1',
    '--with-cudac=/usr/local/cuda/bin/nvcc',
    '--with-cuda-arch=sm_20',
    '--with-thrust=1',
    '--with-cusp=1',
    '--with-cusp-dir=/sandbox/soft/cusp-v0.3.1',
    '--download-txpetscgpu=1',
    '--with-debugging=0',
    'COPTFLAGS=-O3',
    'CXXOPTFLAGS=-O3',
    'FOPTFLAGS=-O3',
    'PETSC_ARCH=arch-cuda-cg-opt',
  ]
  configure.petsc_configure(configure_options)
