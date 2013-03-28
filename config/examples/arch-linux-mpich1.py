#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-mpi-dir=/home/petsc/soft/linux-Ubuntu_12.04-x86_64/mpich-1.2.7p1', #intel
    '--with-cxx=0',
    '--with-shared-libraries=0',
  ]
  configure.petsc_configure(configure_options)

