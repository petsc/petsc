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
    '--download-mpich-shared=0',
    '--with-valgrind=0',            # valgrind is ok but then must also provide -I/opt/local/include to compiler
    'PETSC_ARCH=arch-osx',
  ]
  configure.petsc_configure(configure_options)
