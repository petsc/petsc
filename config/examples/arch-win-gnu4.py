#!/usr/bin/env python

configure_options = [
  # Blas autodetec with cygwin blas at /usr/lib/liblapack,a,libblas.a
  '--with-cc=gcc-4',
  '--with-fc=gfortran-4',
  '--with-cxx=g++-4',
  'DATAFILESPATH=/home/sbalay/datafiles',
  '--download-mpich=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
