#!/usr/bin/python
#
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-suitesparse',
    '--download-mumps',
    '--download-scalapack',
    '--download-chaco',
    '--download-ctetgen',
    '--download-exodusii',
    '--download-cmake',   # for exodus as it breaks with cmake version 2.8.12.2
    '--download-pnetcdf',
    '--download-generator',
    '--download-hdf5',
    '--download-zlib=1',
    '--download-metis',
    '--download-ml',
    '--download-netcdf',
    '--download-parmetis',
    '--download-triangle',
    '--with-cuda',
    '--with-shared-libraries',
  ]
  configure.petsc_configure(configure_options)
