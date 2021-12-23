#!/usr/bin/env python

import os

configure_options = [
  '--with-mpi-dir=/Volumes/Scratch/svcpetsc/soft/mpich-3.4.3',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
