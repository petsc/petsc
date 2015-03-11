#!/usr/bin/python
#
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-chaco',
    '--download-ctetgen',
    '--download-exodusii',
    '--download-fiat',
    '--download-generator',
    '--download-hdf5',
    '--download-metis',
    '--download-ml',
    '--download-netcdf',
    '--download-parmetis',
    '--download-scientificpython',
    '--download-triangle',
    '--with-cuda',
    '--with-cuda-only',
    '--with-shared-libraries',
  ]
  configure.petsc_configure(configure_options)
