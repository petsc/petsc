#!/usr/bin/env python

configure_options = [
  '--with-cc=gcc',
  '--with-fc=gfortran', # http://brew.sh
  '--with-cxx=g++',
  '--download-mpich=1',
  '--download-mpich-device=ch3:nemesis', #for some reason runex174_2_elemental takes very long with ch3:p4
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-elemental=1',
  '--with-cxx-dialect=C++11',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
