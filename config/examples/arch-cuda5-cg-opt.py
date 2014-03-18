#!/usr/bin/python
#
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/sandbox/soft/cuda-5.0/lib64
#
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cuda=1',
    '--with-cuda-dir=/sandbox/soft/cuda-5.0',
    '--with-cudac=/sandbox/soft/cuda-5.0/bin/nvcc',
    '--with-cuda-arch=sm_20',
    '--with-thrust=1',
    '--with-cusp=1',
    '--with-cusp-dir=/sandbox/soft/cusp-v0.3.1',
    '--with-debugging=0',
    'COPTFLAGS=-O3',
    'CXXOPTFLAGS=-O3',
    'FOPTFLAGS=-O3',
  ]
  configure.petsc_configure(configure_options)
