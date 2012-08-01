#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-mpich',
    '--with-fc=0',
    '--with-shared-libraries',
    '-download-mpich-shared=0',
    'PETSC_ARCH=arch-framework',
  ]
  configure.petsc_configure(configure_options)
