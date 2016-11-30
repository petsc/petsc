#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-f2cblaslapack',
    '--with-cc=win32fe cl',
    '--with-shared-libraries=1',
    '--with-cxx=0',
    '--with-fc=0',
    '--with-mpi=0',
    'PETSC_ARCH=arch-mswin-uni',
  ]
  configure.petsc_configure(configure_options)

