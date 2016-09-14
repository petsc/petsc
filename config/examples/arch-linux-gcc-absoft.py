#!/usr/bin/env python

configure_options = [
  '--with-cc=gcc',
  '--with-fc=f90',
  'FFLAGS=-M1643', #suppress warnings about unused 'parameter' variables defined in fortran includes
  '--with-cxx=g++',
  '--with-clanguage=c++',
  '--with-shared-libraries=0', # /soft/com/packages/absoft11.0/lib64/libafio.a is not -fPIC compiled
  '--download-fblaslapack=1',
  '--download-mpich=1',
  '--download-mpich-pm=gforker',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--with-matlab=0'
  ]

if __name__ == '__main__':
    import sys,os
    sys.path.insert(0,os.path.abspath('config'))
    import configure
    configure.petsc_configure(configure_options)
