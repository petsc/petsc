#!/usr/bin/env python

configure_options = [
  '--with-clanguage=cxx',
  '--with-cxx-dialect=C++11',
  '--with-debugging=0',
  'DATAFILESPATH=/Users/petsc/datafiles',
  'CXXFLAGS=-Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -Wno-deprecated',
  '--with-visibility=0', # CXXFLAGS disables this option
  '--with-mpi-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-metis-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-parmetis-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-ptscotch-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-triangle-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-superlu-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-superlu_dist-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-scalapack-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-mumps-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-parms-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-hdf5-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-sundials-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-hypre-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-suitesparse-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-chaco-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-spai-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-saws-dir=/Users/glci/soft/cxx-pkgs-opt',
  '--with-revolve-dir=/Users/glci/soft/cxx-pkgs-opt',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
