#!/usr/bin/env python

# Note: Intel 7.1 Fortran cannot work with g++ 3.3
configure_options = [
  # cannot use download-mpich with fortranlib-autodetect=0 so disabling
  #'--with-cc=gcc',
  #'--with-fc=ifort',
  #'--with-cxx=g++',
  #'--download-mpich=1',
  #'--download-mpich-pm=gforker',
  '--with-mpi-dir=/homes/petsc/soft/linux-Ubuntu_12.04-x86_64/mpich-3.1.3-gcc-ifc',

  '--with-clib-autodetect=0',
  '--with-fortranlib-autodetect=0',
  '--with-cxxlib-autodetect=0',
  'LIBS=-L/soft/com/packages/intel/13/079/composer_xe_2013.0.079/compiler/lib/intel64 -lifcore -ldl -limf -lirc -L/homes/petsc/soft/linux-Ubuntu_12.04-x86_64/mpich-3.1.3-gcc-ifc/lib -lmpifort -lmpi',
  '--with-scalar-type=complex',
  '--download-hdf5',
  '--download-fblaslapack=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
