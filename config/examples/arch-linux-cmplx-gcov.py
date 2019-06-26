#!/usr/bin/env python

configure_options = [
  '--with-clanguage=cxx',
  '--with-scalar-type=complex',
  '--with-gcov=1',
  '--download-mpich=1',
  '--download-metis',
  '--download-parmetis',
  '--download-ptscotch',
  '--download-scalapack',
  '--download-strumpack'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
