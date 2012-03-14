#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-mpi-dir=/sandbox/petsc/software/mpich-1.2.7p1',
  ]
  configure.petsc_configure(configure_options)

