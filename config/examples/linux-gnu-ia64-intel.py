#!/usr/bin/env python

configure_options = [
  '--with-vendor-compilers=intel',
  '--with-mpi-dir=/home/balay/soft/mpich2-1.0.4-intel', 
  '--download-parmetis=1',
  'DATAFILESPATH=/home/balay/datafiles',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
