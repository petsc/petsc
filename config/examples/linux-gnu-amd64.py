#!/usr/bin/env python

configure_options = [
  '--download-mpich=1',
  '--download-mpich-pm=gforker',
  '--download-f-blas-lapack=1',
  '--download-prometheus=1',
  '--download-parmetis=1',
  '--with-debugging=0'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
