#!/usr/bin/env python

# moab breaks with --with-cxx-dialect=C++11 - so disable elemental?
# moab appears to break with -with-visibility=1 - so disable it

configure_options = [
  '--with-cc=gcc',
  '--with-fc=gfortran', # http://brew.sh/
  '--with-cxx=g++',

  '--with-clanguage=cxx',
  '--with-debugging=0',
  '--with-visibility=0',

  #'-download-fblaslapack=1',
  '--download-mpich=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-triangle=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-parms=1',
  #'--download-hdf5',
  '--download-sundials=1',
  '--download-hypre=1',
  '--download-suitesparse=1',
  '--download-chaco=1',
  '--download-spai=1',
  '--download-moab=1',
  '--download-saws',
  '--download-revolve=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
