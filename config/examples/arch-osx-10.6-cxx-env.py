#!/usr/bin/env python

configure_options = [
  '--download-mpich=1',
  '--with-environment-variables=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  os.environ['CC']='gcc'
  os.environ['CXX']='g++'
  os.environ['FC']='gfortran'
  configure.petsc_configure(configure_options)
