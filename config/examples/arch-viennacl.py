#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-viennacl=yes',
    '--with-opencl-include=/usr/local/cuda-5.0/include'
    '--with-opencl-lib=/usr/lib/libOpenCL.so'
  ]
  configure.petsc_configure(configure_options)
