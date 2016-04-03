#!/usr/bin/env python

# find the ifort libs location
import os
import distutils.spawn
ifort_lib_dir=os.path.join(os.path.dirname(os.path.dirname(distutils.spawn.find_executable('ifort'))),'lib','intel64')
mpich_install_dir='/homes/petsc/soft/linux-Ubuntu_14.04-x86_64/mpich-3.2-gcc-ifc'
mpich_lib_dir=os.path.join(mpich_install_dir,'lib')

configure_options = [
  # cannot use download-mpich with fortranlib-autodetect=0 so disabling
  #'--with-cc=gcc',
  #'--with-fc=ifort',
  #'--with-cxx=g++',
  #'--download-mpich=1',
  #'--download-mpich-pm=gforker',
  '--with-mpi-dir='+mpich_install_dir,

  'LIBS=-L'+ifort_lib_dir+' -lifcore -ldl -limf -lirc -L'+mpich_lib_dir+' -lmpifort -lmpi',
  '--with-scalar-type=complex',
  '--download-hdf5',
  '--with-zlib=1',
  '--download-fblaslapack=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
