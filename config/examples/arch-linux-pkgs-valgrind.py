#!/usr/bin/env python

configure_options = [
  '--download-mpich=1',
  '--download-fblaslapack=1',
  '--download-hypre=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-suitesparse=1',
  '--download-triangle=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-elemental=1',
  '--with-cxx-dialect=C++11',
  #'--download-spai=1', valgrind leaks here will probably not get fixed in the near future
  '--download-parms=1',
  '--download-moab=1',
  '--download-chaco=1',
  '--download-revolve=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
