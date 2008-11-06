#!/usr/bin/env python

configure_options = [
  # Autodetect MPICH & Intel MKL
  # path set to $PETSC_DIR/bin/win32fe
  '--with-cc=win32fe cl',
  '--with-cxx=win32fe cl',
  '--with-fc=win32fe f90',
  'DATAFILESPATH=/home/balay/datafiles',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
