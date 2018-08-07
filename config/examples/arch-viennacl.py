#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-viennacl',
    '--with-opencl-include=/usr/local/cuda/include',
    '--with-opencl-lib=-L/usr/local/cuda/lib64 -lOpenCL'
  ]
  configure.petsc_configure(configure_options)
