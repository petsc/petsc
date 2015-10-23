#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-viennacl',
    '--with-opencl-include=/home/balay/soft/cuda-5.5/include',
    '--with-opencl-lib=-lOpenCL'
  ]
  configure.petsc_configure(configure_options)
