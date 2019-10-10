#!/usr/bin/env python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')
if not os.path.isdir(petsc_hash_pkgs): os.mkdir(petsc_hash_pkgs)

# find the ifort libs location
import os
import distutils.spawn
ifort_lib_dir=os.path.join(os.path.dirname(os.path.dirname(distutils.spawn.find_executable('ifort'))),'lib','intel64')
mpich_install_dir='/homes/petsc/soft/linux-Ubuntu_14.04-x86_64/mpich-3.2-gcc-ifc'
mpich_lib_dir=os.path.join(mpich_install_dir,'lib')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  # cannot use download-mpich with fortranlib-autodetect=0 so disabling
  #'--with-cc=gcc',
  #'--with-fc=ifort',
  #'--with-cxx=g++',
  #'--download-mpich=1',
  #'--download-mpich-pm=gforker',
  '--with-mpi-dir='+mpich_install_dir,

  'LIBS=-L'+ifort_lib_dir+' -lifcore -ldl -limf -lirc -L'+mpich_lib_dir+' -lmpifort -lmpi',

  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',

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
