#!/usr/bin/env python

configure_options = [
  '--with-serialize-functions=1',
  '--download-mpich=1',
  '--download-openblas=1',
  '--download-openblas-make-options=TARGET=CORE2 DYNAMIC_ARCH=0',
  '--with-fortran-datatypes=1',
  '--with-fortran-interfaces=0',
  '--download-hypre=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-suitesparse=1',
  '--download-triangle=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  #'--download-scalapack=1',
  #'--download-mumps=1', disabled as some mumps tests are fortran tests - that don't work with fortran-datatypes
  '--download-elemental=1',
  '--with-cxx-dialect=C++11',
  '--download-spai=1',
  '--download-parms=1',
  '--download-chaco=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
