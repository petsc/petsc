#!/usr/bin/env python

configure_options = [
  # use MPICH from PGI
  #'--with-cc=pgcc',
  #'--with-fc=pgfortran',
  '--with-cxx=0', # osx PGI does not have c++? And autodetect code messes up -L "foo bar" paths
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  #'CXXOPTFLAGS=-g -O',
  #'--download-mpich=1',
  #'--download-mpich-device=ch3:nemesis', # socket code gives 'Error from ioctl = 6; Error is: : Device not configured'
  #'--download-cmake=1', #use from brew
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-triangle=1',
  #'--download-superlu=1',
  #'--download-superlu_dist=1', disable due to error: PGC-S-0039-Use of undeclared variable FLT_ROUNDS (smach.c: 71)
  '--download-scalapack=1',
  '--download-mumps=1',
  #'--download-parms=1',
  #'--download-hdf5',
  '--download-sundials=1',
  #'--download-hypre=1',
  '--download-suitesparse=1',
  '--download-chaco=1',
  '--download-spai=1',
  #'--download-moab=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
