#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
# make sure this has the same options as arch-osx.py
  configure_options = [
    '--download-mpich',
    '--with-fc=0',
    '--with-shared-libraries',
    '--download-mpich-shared=0',
    '--with-valgrind=0',
    '--with-hwloc=0',
    '--with-debugging=0',
    '--download-sowing=1',
    '--download-c2html=1',
    'PETSC_ARCH=arch-osx-release',
  ]
  configure.petsc_configure(configure_options)
