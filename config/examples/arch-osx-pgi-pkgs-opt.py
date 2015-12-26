#!/usr/bin/env python

# Note: /Applications/Free PGI.app/Contents/Resources/pgi/osx86-64/14.3/include/va_list.h
# is edited to worarround duplicate 'typedef' warnings. And the following to avoid link warning
# cd "/Applications/Free PGI.app/Contents/Resources/pgi/osx86-64/14.3" && ln -s lib libso

configure_options = [
  '--with-cc=pgcc',
  '--with-fc=pgfortran',
  '--with-cxx=0', # osx PGI does not have c++? And autodetect code messes up -L "foo bar" paths

  '--download-mpich=1',
  '--download-mpich-device=ch3:nemesis', # socket code gives 'Error from ioctl = 6; Error is: : Device not configured'
  '--download-cmake=1',
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
