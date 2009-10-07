#!/usr/bin/env python

configure_options = [
  'CC=gcc',
  'FC=pgf90',
  '--download-mpich=1',
  '--download-f-blas-lapack=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
