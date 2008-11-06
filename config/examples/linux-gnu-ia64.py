#!/usr/bin/env python

configure_options = [
  '--with-shared=1',
  '--download-mpich=1',
  'DATAFILESPATH=/home/balay/datafiles',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
