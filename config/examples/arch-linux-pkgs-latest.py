#!/usr/bin/env python

configure_options = [
  '--with-debugging=0',
  #prereq packages
  '--download-mpich=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-triangle=1', # for hdf5 tests
  '--download-chaco=1', # for hdf5 tests
  '--download-hdf5=1',
  #test latest snapshot
  '--download-superlu=1',
  '--download-superlu-commit=origin/master',
  '--download-superlu_dist=1',
  '--download-superlu_dist-commit=origin/master',
  '--download-saws',
  '--download-saws-commit=origin/master',
  '--download-chombo=1',
  '--download-chombo-commit=origin/master',
  '--download-hypre=1',
  '--download-hypre-commit=origin/master',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
