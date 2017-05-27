#!/usr/bin/env python

configure_options = [
  '--download-xsdk',
  '--download-fblaslapack=1',
  '--download-mpich=1',
  '--download-cmake=1',
  '--with-clanguage=C++',
  '--with-debugging=0',
  '--with-shared-libraries=1',
  '--download-triangle=1', # for TRIANGLE_HDF5 test (with trilinos chaco)
  #test latest snapshot
  '--download-hypre-commit=origin/master',
  '--download-superlu_dist-commit=origin/master',
  '--download-trilinos-commit=origin/master',
  '--download-xsdktrilinos-commit=origin/master',

  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
