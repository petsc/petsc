#!/usr/bin/env python

configure_options = [
  '--download-xsdk',
  '--download-triangle', # for TRIANGLE_HDF5 test (with trilinos chaco)
  '--download-mpich=1',
  '--download-cmake=1',
  '--with-debugging=0',
  '--with-shared-libraries=0',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
